// Package testenv gives tests access to the ONNX Runtime library and the
// model files installed on the machine, skipping when they are absent.
package testenv

import (
	"os"
	"sync"
	"testing"

	"github.com/muhriddinsalohiddin/likeness/internal/home"
	"github.com/muhriddinsalohiddin/likeness/internal/models"
	"github.com/muhriddinsalohiddin/likeness/internal/ortx"
)

var (
	once sync.Once
	rt   *ortx.Runtime
	err  error
)

// Runtime loads the runtime once per test binary, or skips the test when
// the library is not installed.
func Runtime(t testing.TB) *ortx.Runtime {
	t.Helper()
	once.Do(func() {
		lib := home.LibraryPath()
		if _, statErr := os.Stat(lib); statErr != nil {
			err = statErr
			return
		}
		rt, err = ortx.Load(ortx.Config{Library: lib, Threads: 4})
	})
	if err != nil {
		t.Skipf("onnxruntime is not available: %v (run `likeness setup`, or set LIKENESS_ORT_LIBRARY)", err)
	}
	return rt
}

// Model returns the models directory after checking that the named model is
// installed there, or skips the test.
func Model(t testing.TB, name string) (*models.Manifest, string) {
	t.Helper()
	m, ok := models.Lookup(name)
	if !ok {
		t.Fatalf("no model named %q", name)
	}
	dir := home.ModelsDir()
	if _, err := models.Installed(dir, m); err != nil {
		t.Skipf("model %s is not installed: %v", name, err)
	}
	return m, dir
}
