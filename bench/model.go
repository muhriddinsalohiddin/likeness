package main

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

// Spec describes how a model expects its input. The engine stays
// model-agnostic: everything model-specific lives here.
type Spec struct {
	File string
	Kind string // "scrfd", "yunet", "rec"
	Size int    // square input side
	Pre  Pre
}

var specs = map[string]Spec{
	"det_500m":      {File: "det_500m.onnx", Kind: "scrfd", Size: 640, Pre: Pre{Mean: 127.5, Std: 128}},
	"det_10g":       {File: "det_10g.onnx", Kind: "scrfd", Size: 640, Pre: Pre{Mean: 127.5, Std: 128}},
	"yunet_2023mar": {File: "yunet_2023mar.onnx", Kind: "yunet", Size: 640, Pre: Pre{Mean: 0, Std: 1, BGR: true}},
	"w600k_mbf":     {File: "w600k_mbf.onnx", Kind: "rec", Size: 112, Pre: Pre{Mean: 127.5, Std: 127.5}},
	"w600k_r50":     {File: "w600k_r50.onnx", Kind: "rec", Size: 112, Pre: Pre{Mean: 127.5, Std: 127.5}},
	"sface_2021dec": {File: "sface_2021dec.onnx", Kind: "rec", Size: 112, Pre: Pre{Mean: 0, Std: 1}},
}

// specFor resolves "name" or "name@size" (detector input side override).
func specFor(name string) (Spec, error) {
	base, size, hasSize := strings.Cut(name, "@")
	s, ok := specs[base]
	if !ok {
		return Spec{}, fmt.Errorf("unknown model %q", base)
	}
	if hasSize {
		n, err := strconv.Atoi(size)
		if err != nil {
			return Spec{}, fmt.Errorf("bad size in %q: %w", name, err)
		}
		s.Size = n
	}
	return s, nil
}

// Model is one ONNX session shared by all goroutines; Run is thread-safe.
type Model struct {
	Name      string
	Spec      Spec
	LoadTime  time.Duration
	sess      *ort.DynamicAdvancedSession
	inShape   ort.Shape
	outShapes []ort.Shape
	// autoOut: the model file declares output shapes for another input size
	// (InsightFace detectors are exported at 640), and ORT rejects
	// pre-allocated outputs that contradict the declaration. Let ORT allocate
	// the (small) outputs on every run instead.
	autoOut bool
}

// Binding holds the pre-allocated input/output tensors of one worker, so the
// hot path performs no allocation.
type Binding struct {
	m    *Model
	In   *ort.Tensor[float32]
	outs []ort.Value
	Out  [][]float32
}

func LoadModel(dir, name string, threads int, spin bool) (*Model, error) {
	spec, err := specFor(name)
	if err != nil {
		return nil, err
	}
	path := filepath.Join(dir, spec.File)
	start := time.Now()

	ins, outs, err := ort.GetInputOutputInfo(path)
	if err != nil {
		return nil, fmt.Errorf("read model info %s: %w", path, err)
	}
	if len(ins) != 1 {
		return nil, fmt.Errorf("%s: expected 1 input, got %d", name, len(ins))
	}
	shape := append(ort.Shape(nil), ins[0].Dimensions...)
	if len(shape) != 4 {
		return nil, fmt.Errorf("%s: expected NCHW input, got %v", name, shape)
	}
	if shape[0] < 1 {
		shape[0] = 1
	}
	if shape[2] < 1 || shape[3] < 1 {
		shape[2], shape[3] = int64(spec.Size), int64(spec.Size)
	} else {
		// static model: its own size wins
		spec.Size = int(shape[2])
	}
	outNames := make([]string, len(outs))
	for i, o := range outs {
		outNames[i] = o.Name
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()
	if err := opts.SetIntraOpNumThreads(threads); err != nil {
		return nil, err
	}
	if err := opts.SetInterOpNumThreads(1); err != nil {
		return nil, err
	}
	if err := opts.SetExecutionMode(ort.ExecutionModeSequential); err != nil {
		return nil, err
	}
	if err := opts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll); err != nil {
		return nil, err
	}
	if !spin {
		// Idle pool threads block instead of busy-waiting, so they do not
		// steal cores from the next stage of the pipeline.
		if err := opts.AddSessionConfigEntry("session.intra_op.allow_spinning", "0"); err != nil {
			return nil, err
		}
	}

	sess, err := ort.NewDynamicAdvancedSession(path, []string{ins[0].Name}, outNames, opts)
	if err != nil {
		return nil, fmt.Errorf("create session %s: %w", name, err)
	}
	m := &Model{Name: name, Spec: spec, sess: sess, inShape: shape}

	// One probing run with auto-allocated outputs tells us the concrete
	// output shapes for this input size.
	in, err := ort.NewEmptyTensor[float32](shape)
	if err != nil {
		return nil, err
	}
	defer in.Destroy()
	probe := make([]ort.Value, len(outNames))
	if err := sess.Run([]ort.Value{in}, probe); err != nil {
		return nil, fmt.Errorf("probe run %s: %w", name, err)
	}
	for i, v := range probe {
		got := append(ort.Shape(nil), v.GetShape()...)
		m.outShapes = append(m.outShapes, got)
		if decl := outs[i].Dimensions; len(decl) == len(got) {
			for j := range decl {
				if decl[j] > 0 && decl[j] != got[j] {
					m.autoOut = true
				}
			}
		}
		v.Destroy()
	}
	m.LoadTime = time.Since(start)
	return m, nil
}

func (m *Model) NewBinding() (*Binding, error) {
	in, err := ort.NewEmptyTensor[float32](m.inShape)
	if err != nil {
		return nil, err
	}
	b := &Binding{m: m, In: in, Out: make([][]float32, 0, len(m.outShapes))}
	if m.autoOut {
		b.outs = make([]ort.Value, len(m.outShapes))
		b.Out = make([][]float32, len(m.outShapes))
	}
	for _, s := range m.outShapes {
		if m.autoOut {
			break
		}
		t, err := ort.NewEmptyTensor[float32](s)
		if err != nil {
			return nil, err
		}
		b.outs = append(b.outs, t)
		b.Out = append(b.Out, t.GetData())
	}
	// letterbox padding is black *before* normalisation
	pad := (0 - m.Spec.Pre.Mean) / m.Spec.Pre.Std
	data := in.GetData()
	for i := range data {
		data[i] = pad
	}
	return b, nil
}

func (b *Binding) Run() error {
	if !b.m.autoOut {
		return b.m.sess.Run([]ort.Value{b.In}, b.outs)
	}
	for i, o := range b.outs {
		if o != nil {
			o.Destroy()
			b.outs[i] = nil
		}
	}
	if err := b.m.sess.Run([]ort.Value{b.In}, b.outs); err != nil {
		return err
	}
	for i, o := range b.outs {
		t, ok := o.(*ort.Tensor[float32])
		if !ok {
			return fmt.Errorf("%s: output %d is not a float32 tensor", b.m.Name, i)
		}
		b.Out[i] = t.GetData()
	}
	return nil
}

func (b *Binding) Destroy() {
	b.In.Destroy()
	for _, o := range b.outs {
		if o != nil {
			o.Destroy()
		}
	}
}
