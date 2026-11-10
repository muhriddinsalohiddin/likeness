// Package home locates the directory where likeness keeps downloaded
// artifacts: the ONNX Runtime library and model files.
package home

import (
	"os"
	"path/filepath"
	"runtime"
)

// Dir returns $LIKENESS_HOME, or a "likeness" directory inside the user's
// cache directory. Everything in it can be downloaded again.
func Dir() string {
	if d := os.Getenv("LIKENESS_HOME"); d != "" {
		return d
	}
	base, err := os.UserCacheDir()
	if err != nil {
		base = os.TempDir()
	}
	return filepath.Join(base, "likeness")
}

// ModelsDir is where model files live.
func ModelsDir() string { return filepath.Join(Dir(), "models") }

// LibraryDir is where the runtime library lives.
func LibraryDir() string { return filepath.Join(Dir(), "lib") }

// LibraryName is the file name of the ONNX Runtime shared library on this
// platform.
func LibraryName() string {
	switch runtime.GOOS {
	case "darwin":
		return "libonnxruntime.dylib"
	case "windows":
		return "onnxruntime.dll"
	}
	return "libonnxruntime.so"
}

// LibraryPath returns $LIKENESS_ORT_LIBRARY or the library's place under
// LibraryDir. The file may not exist yet.
func LibraryPath() string {
	if p := os.Getenv("LIKENESS_ORT_LIBRARY"); p != "" {
		return p
	}
	return filepath.Join(LibraryDir(), LibraryName())
}
