package likeness

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/muhriddinsalohiddin/likeness/internal/fixtures"
	"github.com/muhriddinsalohiddin/likeness/internal/home"
)

func TestPresets(t *testing.T) {
	names := Presets()
	if len(names) < 2 || names[0] != "default" {
		t.Fatalf("presets %v: the default one comes first", names)
	}
}

// These need neither the runtime nor any model.
func TestOpenFailsCleanly(t *testing.T) {
	if _, err := Open(Config{Preset: "resnet-9000"}); err == nil || !strings.Contains(err.Error(), "default") {
		t.Errorf("unknown preset: %v", err)
	}

	empty := t.TempDir()
	t.Setenv("LIKENESS_ORT_LIBRARY", "")
	if _, err := Open(Config{Home: empty}); !errors.Is(err, os.ErrNotExist) || !strings.Contains(err.Error(), "ONNX Runtime") {
		t.Errorf("no library: %v", err)
	}
	if _, err := Open(Config{Library: filepath.Join(empty, "nowhere.so")}); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("explicit library path: %v", err)
	}

	// a library file is there, the models are not
	lib := filepath.Join(empty, "lib", home.LibraryName())
	if err := os.MkdirAll(filepath.Dir(lib), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(lib, []byte("placeholder"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(Config{Home: empty}); !errors.Is(err, os.ErrNotExist) || !strings.Contains(err.Error(), "yunet_2023mar") {
		t.Errorf("no models: %v", err)
	}

	// a model file with the right name and the wrong content
	model := filepath.Join(empty, "models", "face_detection_yunet_2023mar.onnx")
	if err := os.MkdirAll(filepath.Dir(model), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(model, []byte("these are not the weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(Config{Home: empty}); err == nil || !strings.Contains(err.Error(), "checksum") {
		t.Errorf("corrupt model: %v", err)
	}
}

func TestInfo(t *testing.T) {
	e := open(t, "default")
	got := e.Info()
	if got.Preset != "default" || got.Detector != "yunet_2023mar" || got.Embedder != "sface_2021dec" || got.Dim != 128 || got.Threads != testThreads {
		t.Fatalf("%+v", got)
	}
	if !strings.HasPrefix(got.Fingerprint, "sface:sface_2021dec:") || got.Runtime == "" {
		t.Fatalf("%+v", got)
	}
	other := open(t, "buffalo_s")
	if other.Info().Fingerprint == got.Fingerprint || other.Info().Dim != 512 {
		t.Fatalf("%+v", other.Info())
	}
}

func TestEnginesShareTheRuntime(t *testing.T) {
	first := open(t, "default")
	pic, _ := fixtures.Closeup("cassada")

	second, err := Open(Config{Threads: testThreads})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Open(Config{Threads: testThreads + 1}); err == nil || !strings.Contains(err.Error(), "different configuration") {
		t.Errorf("an engine with other thread settings must be refused while one is open: %v", err)
	}
	second.Close()
	second.Close() // closing twice is harmless
	if _, err := second.Analyze(context.Background(), pic, ForSearch()); !errors.Is(err, ErrClosed) {
		t.Errorf("a closed engine answered: %v", err)
	}
	// the first engine still holds the runtime
	if _, reason := analyze(t, first, pic, ForSearch()).Best(); reason != Accepted {
		t.Errorf("the surviving engine: %q", reason)
	}
}

func TestRuntimeIsReleasedWithTheLastEngine(t *testing.T) {
	e := open(t, "default")
	e.Close()
	if sharedRuntime.refs != 0 || sharedRuntime.rt != nil {
		t.Fatalf("runtime still held: %d references", sharedRuntime.refs)
	}
	// and a fresh engine, with other settings, can start
	again, err := Open(Config{Threads: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer again.Close()
	pic, _ := fixtures.Closeup("hathaway")
	if _, reason := analyze(t, again, pic, ForEnrolment()).Best(); reason != Accepted {
		t.Fatalf("%q", reason)
	}
	if again.Info().Threads != 2 {
		t.Fatalf("threads %d", again.Info().Threads)
	}
}
