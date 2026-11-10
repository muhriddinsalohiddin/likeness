// Package detect finds faces and their landmarks with an ONNX detector.
package detect

import (
	"errors"
	"fmt"
	"path/filepath"
	"sync"

	"github.com/muhriddinsalohiddin/likeness/internal/align"
	"github.com/muhriddinsalohiddin/likeness/internal/models"
	"github.com/muhriddinsalohiddin/likeness/internal/ortx"
	"github.com/muhriddinsalohiddin/likeness/internal/vision"
)

// Face is one detection in upright picture coordinates.
type Face struct {
	// Box is x1, y1, x2, y2.
	Box   [4]float32
	Score float32
	// Landmarks are left eye, right eye, nose tip, left mouth corner, right
	// mouth corner, as seen in the picture.
	Landmarks [5]align.Point
}

// Width of the box in pixels.
func (f Face) Width() float32 { return f.Box[2] - f.Box[0] }

// Height of the box in pixels.
func (f Face) Height() float32 { return f.Box[3] - f.Box[1] }

// Options tune one detection call. The zero value uses the model defaults.
type Options struct {
	// Size is the side of the square network input. Smaller is faster and
	// misses small faces. Models with a fixed input ignore it.
	Size int
	// Threshold is the minimum score; 0 means 0.5.
	Threshold float32
	// NMS is the overlap above which two boxes count as the same face; 0
	// means the model's usual value.
	NMS float32
	// NoAntiAlias samples the picture the way reference implementations do,
	// without a pre-filter. Used by parity tests.
	NoAntiAlias bool
}

const (
	DefaultThreshold = 0.5
	minSize          = 64
	maxSize          = 2048
)

// Detector runs one detection model. It is safe for concurrent use.
type Detector struct {
	m      *models.Manifest
	sess   *ortx.Session
	static int // fixed input side, 0 when the model accepts any
	yunet  map[string]int
	pool   sync.Pool
}

type worker struct {
	tensors map[int]*ortx.Tensor
	out     ortx.Outputs
	faces   []Face
}

// New loads the detector described by m from dir.
func New(rt *ortx.Runtime, m *models.Manifest, dir string) (*Detector, error) {
	if !m.Kind.Detector() {
		return nil, fmt.Errorf("detect: %s is a %s model, not a detector", m.Name, m.Kind)
	}
	sess, err := rt.NewSessionFromFile(filepath.Join(dir, m.File), ortx.SessionOptions{})
	if err != nil {
		return nil, err
	}
	d := &Detector{m: m, sess: sess}
	if err := d.check(); err != nil {
		sess.Close()
		return nil, fmt.Errorf("detect: %s: %w", m.Name, err)
	}
	return d, nil
}

func (d *Detector) check() error {
	in := d.sess.Inputs()
	if len(in) != 1 || len(in[0].Dims) != 4 || in[0].Dims[1] != 3 || in[0].Type != ortx.Float32 {
		return fmt.Errorf("expected one float32 NCHW input with 3 channels, got %+v", in)
	}
	if h, w := in[0].Dims[2], in[0].Dims[3]; h > 0 || w > 0 {
		if h != w || h < minSize {
			return fmt.Errorf("unsupported fixed input %dx%d", w, h)
		}
		d.static = int(h)
	}
	switch d.m.Kind {
	case models.YuNet:
		d.yunet = map[string]int{}
		for i, o := range d.sess.Outputs() {
			d.yunet[o.Name] = i
		}
		for _, name := range yunetOutputs {
			if _, ok := d.yunet[name]; !ok {
				return fmt.Errorf("output %q missing", name)
			}
		}
	case models.SCRFD:
		if n := len(d.sess.Outputs()); n != 9 {
			return fmt.Errorf("expected 9 outputs, got %d", n)
		}
	}
	return nil
}

// Manifest returns the model description.
func (d *Detector) Manifest() *models.Manifest { return d.m }

// InputSize returns the network input side that o would use.
func (d *Detector) InputSize(o Options) int {
	if d.static > 0 {
		return d.static
	}
	if o.Size <= 0 {
		return d.m.Input
	}
	return min(max(o.Size, minSize), maxSize) &^ 31 // strides divide 32
}

// Close releases the model. It must not overlap with Detect.
func (d *Detector) Close() { d.sess.Close() }

// Detect finds the faces in img, best score first.
func (d *Detector) Detect(img *vision.Image, o Options) ([]Face, error) {
	if img == nil {
		return nil, errors.New("detect: nil image")
	}
	size := d.InputSize(o)
	thr := o.Threshold
	if thr <= 0 {
		thr = DefaultThreshold
	}
	iou := o.NMS
	if iou <= 0 {
		iou = defaultNMS(d.m.Kind)
	}

	w, _ := d.pool.Get().(*worker)
	if w == nil {
		w = &worker{tensors: map[int]*ortx.Tensor{}}
	}
	defer d.pool.Put(w)
	t, ok := w.tensors[size]
	if !ok {
		var err error
		if t, err = ortx.NewFloat32(1, 3, int64(size), int64(size)); err != nil {
			return nil, err
		}
		w.tensors[size] = t
	}
	toPicture := img.Letterbox(t.Float32(), size, size, d.m.Norm, vision.Options{NoAntiAlias: o.NoAntiAlias})
	if err := d.sess.Run([]*ortx.Tensor{t}, &w.out); err != nil {
		return nil, err
	}

	var faces []Face
	switch d.m.Kind {
	case models.SCRFD:
		faces = decodeSCRFD(&w.out, size, thr, w.faces[:0])
	case models.YuNet:
		faces = decodeYuNet(&w.out, d.yunet, size, thr, w.faces[:0])
	}
	w.faces = faces
	faces = nms(faces, iou)
	out := make([]Face, len(faces))
	for i, f := range faces {
		out[i] = toPictureCoords(f, toPicture)
	}
	return out, nil
}

func defaultNMS(k models.Kind) float32 {
	if k == models.YuNet {
		return 0.3
	}
	return 0.4
}

func toPictureCoords(f Face, m align.Affine) Face {
	tl := m.Apply(align.Point{X: f.Box[0], Y: f.Box[1]})
	br := m.Apply(align.Point{X: f.Box[2], Y: f.Box[3]})
	f.Box = [4]float32{tl.X, tl.Y, br.X, br.Y}
	for i := range f.Landmarks {
		f.Landmarks[i] = m.Apply(f.Landmarks[i])
	}
	return f
}
