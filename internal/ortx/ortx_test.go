package ortx

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/muhriddinsalohiddin/likeness/internal/home"
)

var rt *Runtime

// childMode selects what a re-executed copy of this test binary does; see
// TestCallsBeforeLoadFailCleanly and TestClosedRuntimeRefusesSessions.
const childMode = "LIKENESS_ORTX_TEST_CHILD"

func TestMain(m *testing.M) {
	lib := home.LibraryPath()
	if _, err := os.Stat(lib); err == nil && os.Getenv(childMode) == "" {
		r, err := Load(Config{Library: lib, Threads: 2})
		if err != nil {
			panic(err)
		}
		rt = r
	}
	code := m.Run()
	if rt != nil {
		rt.Close()
	}
	os.Exit(code)
}

func runtimeOrSkip(t *testing.T) *Runtime {
	t.Helper()
	if rt == nil {
		t.Skipf("onnxruntime library not found at %s (run `likeness setup`, or set LIKENESS_ORT_LIBRARY)", home.LibraryPath())
	}
	return rt
}

func session(t *testing.T, name string) *Session {
	t.Helper()
	s, err := runtimeOrSkip(t).NewSessionFromFile(filepath.Join("testdata", name), SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func tensor(t *testing.T, values []float32, dims ...int64) *Tensor {
	t.Helper()
	x, err := NewFloat32(dims...)
	if err != nil {
		t.Fatal(err)
	}
	copy(x.Float32(), values)
	t.Cleanup(func() { x.Close() })
	return x
}

func TestLoadRejectsMissingLibrary(t *testing.T) {
	if _, err := Load(Config{Library: filepath.Join(t.TempDir(), "nope.so")}); err == nil {
		t.Fatal("a missing library must be an error")
	}
	if _, err := Load(Config{}); err == nil {
		t.Fatal("an empty path must be an error")
	}
}

func TestLoadIsIdempotent(t *testing.T) {
	r := runtimeOrSkip(t)
	again, err := Load(r.Config())
	if err != nil || again != r {
		t.Fatalf("second Load: %v, same=%v", err, again == r)
	}
	other := r.Config()
	other.Threads++
	if _, err := Load(other); err == nil {
		t.Fatal("a different configuration must be refused while the runtime is loaded")
	}
	if v := r.Version(); !strings.HasPrefix(v, "1.") {
		t.Fatalf("unexpected version %q", v)
	}
}

func TestAdd(t *testing.T) {
	s := session(t, "add.onnx")
	in, out := s.Inputs(), s.Outputs()
	if len(in) != 2 || in[0].Name != "a" || in[1].Name != "b" || len(out) != 1 || out[0].Name != "y" {
		t.Fatalf("unexpected signature: in=%v out=%v", in, out)
	}
	if !in[0].Static() || in[0].Type != Float32 || len(in[0].Dims) != 2 || in[0].Dims[1] != 3 {
		t.Fatalf("unexpected input info %+v", in[0])
	}
	a := tensor(t, []float32{1, 2, 3}, 1, 3)
	b := tensor(t, []float32{10, 20, 30}, 1, 3)
	var res Outputs
	for round := 0; round < 3; round++ {
		if err := s.Run([]*Tensor{a, b}, &res); err != nil {
			t.Fatal(err)
		}
		if len(res.Data) != 1 || len(res.Dims) != 1 {
			t.Fatalf("got %d outputs", len(res.Data))
		}
		if d := res.Dims[0]; len(d) != 2 || d[0] != 1 || d[1] != 3 {
			t.Fatalf("output dims %v", d)
		}
		if got := res.Data[0]; got[0] != 11+float32(round) || got[1] != 22 || got[2] != 33 {
			t.Fatalf("round %d: got %v", round, got)
		}
		a.Float32()[0] += 1 // the tensor memory is live: the next run sees the change
	}
	if res.Data[0][0] != 13 {
		t.Fatalf("input edits were not observed: %v", res.Data[0])
	}
}

func TestDynamicShapes(t *testing.T) {
	s := session(t, "dyn.onnx")
	in := s.Inputs()[0]
	if in.Static() || len(in.Dims) != 4 || in.Dims[1] != 3 || in.Dims[2] != -1 || in.Dims[3] != -1 {
		t.Fatalf("expected symbolic H and W, got %+v", in)
	}
	var res Outputs
	for _, hw := range [][2]int64{{4, 5}, {2, 2}, {7, 3}} {
		n := 3 * hw[0] * hw[1]
		values := make([]float32, n)
		var want float32
		for i := range values {
			values[i] = float32(i%5) - 2 // -2..2
			want += values[i]
		}
		x := tensor(t, values, 1, 3, hw[0], hw[1])
		if err := s.Run([]*Tensor{x}, &res); err != nil {
			t.Fatal(err)
		}
		relu, sum := res.Data[0], res.Data[1]
		if d := res.Dims[0]; len(d) != 4 || d[2] != hw[0] || d[3] != hw[1] {
			t.Fatalf("relu dims %v for %v", d, hw)
		}
		if int64(len(relu)) != n {
			t.Fatalf("relu has %d values, want %d", len(relu), n)
		}
		for i, v := range values {
			if want := max(v, 0); relu[i] != want {
				t.Fatalf("relu[%d] = %v, want %v", i, relu[i], want)
			}
		}
		if len(sum) != 1 || sum[0] != want {
			t.Fatalf("sum = %v, want %v", sum, want)
		}
	}
}

func TestInputValidation(t *testing.T) {
	s := session(t, "add.onnx")
	var res Outputs
	a := tensor(t, []float32{1, 2, 3}, 1, 3)
	if err := s.Run([]*Tensor{a}, &res); err == nil {
		t.Fatal("too few inputs must be rejected")
	}
	wrong := tensor(t, []float32{1, 2}, 1, 2)
	if err := s.Run([]*Tensor{a, wrong}, &res); err == nil || !strings.Contains(err.Error(), "dimensions") {
		t.Fatalf("mismatched dimensions must be rejected before running, got %v", err)
	}
	closed, _ := NewFloat32(1, 3)
	closed.Close()
	if err := s.Run([]*Tensor{a, closed}, &res); err == nil {
		t.Fatal("a closed tensor must be rejected")
	}
	if err := s.Run([]*Tensor{a, nil}, &res); err == nil {
		t.Fatal("a nil tensor must be rejected")
	}
}

func TestUnsupportedOutputType(t *testing.T) {
	s := session(t, "int64out.onnx")
	if got := s.Outputs()[0].Type; got != Int64 {
		t.Fatalf("output type %v, want int64", got)
	}
	x := tensor(t, []float32{1, 2, 3, 4}, 2, 2)
	var res Outputs
	err := s.Run([]*Tensor{x}, &res)
	if err == nil || !strings.Contains(err.Error(), "int64") {
		t.Fatalf("want an error naming the int64 output, got %v", err)
	}
}

func TestBadModel(t *testing.T) {
	r := runtimeOrSkip(t)
	if _, err := r.NewSession([]byte("this is not a model"), SessionOptions{}); err == nil {
		t.Fatal("garbage must not load")
	}
	if _, err := r.NewSession(nil, SessionOptions{}); err == nil {
		t.Fatal("an empty model must not load")
	}
	if _, err := r.NewSessionFromFile(filepath.Join(t.TempDir(), "missing.onnx"), SessionOptions{}); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing file: got %v", err)
	}
}

func TestConcurrentRuns(t *testing.T) {
	s := session(t, "dyn.onnx")
	const workers, rounds = 8, 40
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			x, err := NewFloat32(1, 3, 6, 6)
			if err != nil {
				errs <- err
				return
			}
			defer x.Close()
			for i := range x.Float32() {
				x.Float32()[i] = float32(w + 1)
			}
			var res Outputs
			for r := 0; r < rounds; r++ {
				if err := s.Run([]*Tensor{x}, &res); err != nil {
					errs <- err
					return
				}
				if want := float32((w + 1) * 3 * 36); res.Data[1][0] != want {
					errs <- errors.New("wrong sum under concurrency")
					return
				}
			}
		}(w)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
}

// inChild re-runs the calling test in a fresh process, where TestMain leaves
// the runtime alone, and reports whether this process is that child.
func inChild(t *testing.T, mode string) bool {
	t.Helper()
	if os.Getenv(childMode) == mode {
		return true
	}
	cmd := exec.Command(os.Args[0], "-test.run=^"+t.Name()+"$", "-test.v")
	cmd.Env = append(os.Environ(), childMode+"="+mode)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("child process failed: %v\n%s", err, out)
	}
	return false
}

// A process that never loaded the runtime (the library is missing, or the
// caller forgot) must get errors, not a crash.
func TestCallsBeforeLoadFailCleanly(t *testing.T) {
	if !inChild(t, "unloaded") {
		return
	}
	x, err := NewFloat32(1, 3)
	if err == nil || !strings.Contains(err.Error(), "not loaded") {
		t.Fatalf("NewFloat32 before Load: tensor=%v err=%v", x, err)
	}
	// releasing things that never came to life is harmless
	(&Tensor{}).Close()
	(&Session{}).Close()
	(&Runtime{}).Close()
	if err := (&Session{}).Run(nil, &Outputs{}); err == nil {
		t.Fatal("running a session that was never created must be an error")
	}
}

func TestClosedRuntimeRefusesSessions(t *testing.T) {
	lib := home.LibraryPath()
	if _, err := os.Stat(lib); err != nil {
		t.Skipf("onnxruntime library not found at %s", lib)
	}
	if !inChild(t, "closed") {
		return
	}
	r, err := Load(Config{Library: lib, Threads: 1})
	if err != nil {
		t.Fatal(err)
	}
	model, err := os.ReadFile(filepath.Join("testdata", "add.onnx"))
	if err != nil {
		t.Fatal(err)
	}
	r.Close()
	if _, err := r.NewSession(model, SessionOptions{}); err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("NewSession on a closed runtime: %v", err)
	}
	r.Close() // idempotent
	// the library stays mapped, so a new environment can be created
	again, err := Load(Config{Library: lib, Threads: 1})
	if err != nil {
		t.Fatal(err)
	}
	s, err := again.NewSession(model, SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	s.Close()
	again.Close()
}

func TestTensor(t *testing.T) {
	runtimeOrSkip(t)
	for _, dims := range [][]int64{nil, {0}, {-1, 2}, {1 << 20, 1 << 20}} {
		if _, err := NewFloat32(dims...); err == nil {
			t.Fatalf("dimensions %v must be rejected", dims)
		}
	}
	x, err := NewFloat32(2, 3, 4)
	if err != nil {
		t.Fatal(err)
	}
	if len(x.Float32()) != 24 || len(x.Dims()) != 3 {
		t.Fatalf("len %d dims %v", len(x.Float32()), x.Dims())
	}
	for _, v := range x.Float32() {
		if v != 0 {
			t.Fatal("a new tensor must be zeroed")
		}
	}
	x.Close()
	x.Close() // idempotent
}

func TestSessionCloseIsIdempotent(t *testing.T) {
	s := session(t, "add.onnx")
	s.Close()
	s.Close()
}
