// Package ortx is a small binding to the ONNX Runtime C API: enough to load a
// model and run it with tensors that live outside the Go heap.
//
// The runtime library is opened at run time, so a binary built with this
// package has no link-time dependency on onnxruntime and can fetch the
// library itself.
package ortx

/*
#cgo CFLAGS: -O2 -Wall
#cgo linux LDFLAGS: -ldl
#include <stdlib.h>
#include "ortx.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync"
	"unsafe"
)

// apiVersion is the oldest C API level that has every call this package
// makes; onnxruntime 1.16 (September 2023) and everything after serve it.
const apiVersion = 16

// LogLevel selects how talkative the runtime is on stderr.
type LogLevel int

const (
	LogError LogLevel = iota
	LogFatal
	LogWarning
	LogInfo
	LogVerbose
)

// ort severity values differ from the Go ones so that the zero value is
// "errors only": the runtime's warnings are about model export details
// that a server operator cannot act on.
var ortSeverity = map[LogLevel]C.int{LogVerbose: 0, LogInfo: 1, LogWarning: 2, LogError: 3, LogFatal: 4}

// Config says how to load the runtime and how it may use the CPU.
type Config struct {
	// Library is the path of the onnxruntime shared library.
	Library string
	// Threads is the number of threads one inference may use; 0 means the
	// number of CPUs.
	Threads int
	// PerSessionPools gives every session its own thread pool instead of the
	// one shared pool the environment owns. Shared is faster when several
	// models run back to back, which is what a detect-then-embed pipeline
	// does; the option exists so that both can be measured.
	PerSessionPools bool
	// NoSpin stops idle pool threads from busy-waiting for the next task.
	// Spinning shortens the gap between two kernels at the price of burning a
	// core while a request is in flight.
	NoSpin bool
	Log    LogLevel
}

// Runtime is the loaded library and its environment. A process has one.
type Runtime struct {
	cfg     Config
	env     *C.OrtEnv
	version string
}

var (
	mu      sync.Mutex
	current *Runtime
)

func takeError(msg *C.char) error {
	if msg == nil {
		return nil
	}
	err := errors.New("onnxruntime: " + C.GoString(msg))
	C.ortx_free(unsafe.Pointer(msg))
	return err
}

// Load opens the runtime library and creates the environment. Calling it
// again with the same configuration returns the same Runtime; a different
// configuration is an error until the first Runtime is closed.
func Load(cfg Config) (*Runtime, error) {
	if cfg.Library == "" {
		return nil, errors.New("ortx: no library path given")
	}
	if cfg.Threads <= 0 {
		cfg.Threads = runtime.NumCPU()
	}
	if _, err := os.Stat(cfg.Library); err != nil {
		return nil, fmt.Errorf("ortx: %w", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if current != nil {
		if current.cfg == cfg {
			return current, nil
		}
		return nil, errors.New("ortx: the runtime is already loaded with a different configuration")
	}
	path := C.CString(cfg.Library)
	defer C.free(unsafe.Pointer(path))
	if err := takeError(C.ortx_load(path, apiVersion)); err != nil {
		return nil, err
	}
	spin := C.int(1)
	if cfg.NoSpin {
		spin = 0
	}
	global := C.int(1)
	if cfg.PerSessionPools {
		global = 0
	}
	rt := &Runtime{cfg: cfg, version: C.GoString(C.ortx_version())}
	if err := takeError(C.ortx_env_create(ortSeverity[cfg.Log], global, C.int(cfg.Threads), 1, spin, &rt.env)); err != nil {
		return nil, err
	}
	current = rt
	return rt, nil
}

// Version is the version string of the loaded library.
func (rt *Runtime) Version() string { return rt.version }

// Config returns the configuration the runtime was loaded with.
func (rt *Runtime) Config() Config { return rt.cfg }

// Close releases the environment. Every session must be closed first.
func (rt *Runtime) Close() {
	mu.Lock()
	defer mu.Unlock()
	if rt.env != nil {
		C.ortx_env_release(rt.env)
		rt.env = nil
	}
	if current == rt {
		current = nil
	}
}

// ElementType is the ONNX tensor element type.
type ElementType int

const (
	Undefined ElementType = 0
	Float32   ElementType = 1
	Uint8     ElementType = 2
	Int8      ElementType = 3
	Int32     ElementType = 6
	Int64     ElementType = 7
	Float16   ElementType = 10
	Float64   ElementType = 11
)

func (t ElementType) String() string {
	switch t {
	case Float32:
		return "float32"
	case Uint8:
		return "uint8"
	case Int8:
		return "int8"
	case Int32:
		return "int32"
	case Int64:
		return "int64"
	case Float16:
		return "float16"
	case Float64:
		return "float64"
	case Undefined:
		return "undefined"
	}
	return fmt.Sprintf("type(%d)", int(t))
}

// IOInfo describes one input or output of a model. A dimension of -1 is
// symbolic: the model accepts any size there.
type IOInfo struct {
	Name string
	Dims []int64
	Type ElementType
}

// Static reports whether every dimension is fixed.
func (i IOInfo) Static() bool {
	for _, d := range i.Dims {
		if d < 0 {
			return false
		}
	}
	return true
}

// Optimization is the graph optimisation level of a session.
type Optimization int

const (
	OptimizeAll Optimization = iota
	OptimizeNone
	OptimizeBasic
	OptimizeExtended
)

var ortOptimization = map[Optimization]C.int{OptimizeAll: 99, OptimizeNone: 0, OptimizeBasic: 1, OptimizeExtended: 2}

// SessionOptions tune one session. The zero value is right for inference.
type SessionOptions struct {
	Optimization Optimization
	Log          LogLevel
}

// Session is a loaded model. Run may be called from several goroutines at
// once; Close must not overlap with Run.
type Session struct {
	rt       *Runtime
	s        *C.OrtSession
	inputs   []IOInfo
	outputs  []IOInfo
	inNames  **C.char
	outNames **C.char
}

// NewSession loads a model from its serialised ONNX bytes. The slice is not
// retained.
func (rt *Runtime) NewSession(model []byte, o SessionOptions) (*Session, error) {
	if len(model) == 0 {
		return nil, errors.New("ortx: empty model")
	}
	spin, global := C.int(1), C.int(1)
	if rt.cfg.NoSpin {
		spin = 0
	}
	if rt.cfg.PerSessionPools {
		global = 0
	}
	cfg := C.ortx_session_config{
		graph_optimization: ortOptimization[o.Optimization],
		log_severity:       ortSeverity[o.Log],
		global_pool:        global,
		intra_threads:      C.int(rt.cfg.Threads),
		spin:               spin,
	}
	s := &Session{rt: rt}
	if err := takeError(C.ortx_session_create(rt.env, unsafe.Pointer(&model[0]), C.size_t(len(model)), &cfg, &s.s)); err != nil {
		return nil, err
	}
	if err := s.describe(); err != nil {
		s.Close()
		return nil, err
	}
	return s, nil
}

// NewSessionFromFile loads a model from disk.
func (rt *Runtime) NewSessionFromFile(path string, o SessionOptions) (*Session, error) {
	model, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return rt.NewSession(model, o)
}

func (s *Session) describe() error {
	var nin, nout C.size_t
	if err := takeError(C.ortx_session_count(s.s, &nin, &nout)); err != nil {
		return err
	}
	var err error
	if s.inputs, s.inNames, err = s.ioInfo(0, int(nin)); err != nil {
		return err
	}
	s.outputs, s.outNames, err = s.ioInfo(1, int(nout))
	return err
}

func (s *Session) ioInfo(output C.int, n int) ([]IOInfo, **C.char, error) {
	infos := make([]IOInfo, n)
	names := (**C.char)(C.calloc(C.size_t(max(n, 1)), C.size_t(unsafe.Sizeof(uintptr(0)))))
	slots := unsafe.Slice(names, max(n, 1))
	for i := 0; i < n; i++ {
		var name *C.char
		var elem C.int
		var dims *C.int64_t
		var ndims C.size_t
		if err := takeError(C.ortx_session_info(s.s, output, C.size_t(i), &name, &elem, &dims, &ndims)); err != nil {
			freeNames(names, n)
			return nil, nil, err
		}
		slots[i] = name // owned by the session until Close
		infos[i] = IOInfo{Name: C.GoString(name), Type: ElementType(elem)}
		if ndims > 0 {
			infos[i].Dims = append([]int64(nil), unsafe.Slice((*int64)(unsafe.Pointer(dims)), int(ndims))...)
			C.ortx_free(unsafe.Pointer(dims))
		}
	}
	return infos, names, nil
}

func freeNames(names **C.char, n int) {
	if names == nil {
		return
	}
	for _, p := range unsafe.Slice(names, max(n, 1)) {
		if p != nil {
			C.ortx_free(unsafe.Pointer(p))
		}
	}
	C.free(unsafe.Pointer(names))
}

// Inputs describes the model inputs, in the order Run expects them.
func (s *Session) Inputs() []IOInfo { return s.inputs }

// Outputs describes the model outputs, in the order Run fills them.
func (s *Session) Outputs() []IOInfo { return s.outputs }

// Close releases the session. Runs still in flight must have finished.
func (s *Session) Close() {
	if s.s != nil {
		C.ortx_session_release(s.s)
		s.s = nil
	}
	freeNames(s.inNames, len(s.inputs))
	freeNames(s.outNames, len(s.outputs))
	s.inNames, s.outNames = nil, nil
}

// Outputs receives the results of Run. Its slices are reused between runs,
// so a caller that keeps one Outputs per worker allocates nothing after the
// first run.
type Outputs struct {
	Data [][]float32
	Dims [][]int64
}

// Run executes the model. inputs are matched to Inputs by position and must
// be float32 with compatible dimensions. Every output is copied into out.
func (s *Session) Run(inputs []*Tensor, out *Outputs) error {
	if s.s == nil {
		return errors.New("ortx: session is closed")
	}
	if len(inputs) != len(s.inputs) {
		return fmt.Errorf("ortx: model has %d inputs, got %d", len(s.inputs), len(inputs))
	}
	for i, t := range inputs {
		if t == nil || t.v == nil {
			return fmt.Errorf("ortx: input %d is nil or closed", i)
		}
		if err := checkDims(s.inputs[i], t.dims); err != nil {
			return err
		}
	}
	ins := make([]*C.OrtValue, len(inputs))
	for i, t := range inputs {
		ins[i] = t.v
	}
	outs := make([]*C.OrtValue, max(len(s.outputs), 1))
	err := takeError(C.ortx_run(s.s, s.inNames, &ins[0], C.size_t(len(ins)), s.outNames, &outs[0], C.size_t(len(s.outputs))))
	runtime.KeepAlive(inputs)
	defer func() {
		for _, v := range outs {
			C.ortx_value_release(v)
		}
	}()
	if err != nil {
		return err
	}

	n := len(s.outputs)
	if cap(out.Data) < n {
		out.Data = append(out.Data[:0], make([][]float32, n)...)
		out.Dims = append(out.Dims[:0], make([][]int64, n)...)
	}
	out.Data, out.Dims = out.Data[:n], out.Dims[:n]
	for i := 0; i < n; i++ {
		var elem C.int
		var dims *C.int64_t
		var ndims, count C.size_t
		var data unsafe.Pointer
		if err := takeError(C.ortx_value_read(outs[i], &elem, &dims, &ndims, &data, &count)); err != nil {
			return err
		}
		out.Dims[i] = out.Dims[i][:0]
		if ndims > 0 {
			out.Dims[i] = append(out.Dims[i], unsafe.Slice((*int64)(unsafe.Pointer(dims)), int(ndims))...)
			C.ortx_free(unsafe.Pointer(dims))
		}
		if ElementType(elem) != Float32 {
			return fmt.Errorf("ortx: output %q is %v, only float32 is supported", s.outputs[i].Name, ElementType(elem))
		}
		out.Data[i] = append(out.Data[i][:0], unsafe.Slice((*float32)(data), int(count))...)
	}
	return nil
}

func checkDims(info IOInfo, dims []int64) error {
	if len(dims) != len(info.Dims) {
		return fmt.Errorf("ortx: input %q wants %d dimensions, got %v", info.Name, len(info.Dims), dims)
	}
	for j, d := range info.Dims {
		if d >= 0 && d != dims[j] {
			return fmt.Errorf("ortx: input %q wants dimensions %v, got %v", info.Name, info.Dims, dims)
		}
	}
	return nil
}

// Tensor is a float32 tensor whose memory is allocated outside the Go heap
// and handed to the runtime without copying.
type Tensor struct {
	v    *C.OrtValue
	data unsafe.Pointer
	n    int
	dims []int64
}

// NewFloat32 allocates a zeroed tensor of the given dimensions.
func NewFloat32(dims ...int64) (*Tensor, error) {
	if len(dims) == 0 {
		return nil, errors.New("ortx: a tensor needs at least one dimension")
	}
	n := int64(1)
	for _, d := range dims {
		if d <= 0 || n > (1<<31)/d {
			return nil, fmt.Errorf("ortx: bad tensor dimensions %v", dims)
		}
		n *= d
	}
	cdims := make([]C.int64_t, len(dims))
	for i, d := range dims {
		cdims[i] = C.int64_t(d)
	}
	t := &Tensor{n: int(n), dims: append([]int64(nil), dims...)}
	if err := takeError(C.ortx_tensor_create(C.int(Float32), &cdims[0], C.size_t(len(cdims)), C.size_t(n*4), &t.data, &t.v)); err != nil {
		return nil, err
	}
	return t, nil
}

// Float32 returns the tensor's elements in row-major order. The slice is
// valid until Close.
func (t *Tensor) Float32() []float32 {
	return unsafe.Slice((*float32)(t.data), t.n)
}

// Dims returns the tensor's dimensions.
func (t *Tensor) Dims() []int64 { return t.dims }

// Close frees the tensor.
func (t *Tensor) Close() {
	if t.v != nil {
		C.ortx_value_release(t.v)
		t.v = nil
	}
	if t.data != nil {
		C.ortx_free_aligned(t.data)
		t.data = nil
	}
}
