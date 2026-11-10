package home

import (
	"path/filepath"
	"testing"
)

func TestPathsFollowEnvironment(t *testing.T) {
	t.Setenv("LIKENESS_HOME", "/opt/likeness-data")
	t.Setenv("LIKENESS_ORT_LIBRARY", "")
	if Dir() != "/opt/likeness-data" || ModelsDir() != filepath.Join("/opt/likeness-data", "models") {
		t.Fatalf("Dir=%s ModelsDir=%s", Dir(), ModelsDir())
	}
	if got := LibraryPath(); got != filepath.Join("/opt/likeness-data", "lib", LibraryName()) {
		t.Fatalf("LibraryPath=%s", got)
	}
	t.Setenv("LIKENESS_ORT_LIBRARY", "/usr/lib/libonnxruntime.so")
	if got := LibraryPath(); got != "/usr/lib/libonnxruntime.so" {
		t.Fatalf("LibraryPath override=%s", got)
	}
	t.Setenv("LIKENESS_HOME", "")
	if d := Dir(); filepath.Base(d) != "likeness" {
		t.Fatalf("default Dir=%s", d)
	}
	if n := LibraryName(); n == "" {
		t.Fatal("empty library name")
	}
}
