package models

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

var payload = bytes.Repeat([]byte("likeness model bytes "), 2000)

func digest(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func makeZip(t *testing.T, members map[string][]byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for name, data := range members {
		w, err := zw.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		w.Write(data)
	}
	zw.Close()
	return buf.Bytes()
}

func makeTgz(t *testing.T, members map[string][]byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	for name, data := range members {
		if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0o644, Size: int64(len(data)), Typeflag: tar.TypeReg}); err != nil {
			t.Fatal(err)
		}
		tw.Write(data)
	}
	tw.Close()
	gz.Close()
	return buf.Bytes()
}

// serve returns a server for the given files and a counter of the requests
// it has answered.
func serve(t *testing.T, files map[string][]byte) (*httptest.Server, *int) {
	t.Helper()
	requests := new(int)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*requests++
		b, ok := files[r.URL.Path]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Write(b)
	}))
	t.Cleanup(srv.Close)
	return srv, requests
}

func TestFetchPlainZipAndTgz(t *testing.T) {
	other := []byte("another member that must be ignored")
	srv, requests := serve(t, map[string][]byte{
		"/plain.onnx": payload,
		"/pack.zip":   makeZip(t, map[string][]byte{"other.onnx": other, "wanted.onnx": payload}),
		"/pack.tgz":   makeTgz(t, map[string][]byte{"pack-1.0/lib/other.so": other, "pack-1.0/lib/wanted.so": payload}),
	})
	want := Artifact{SHA256: digest(payload), Size: int64(len(payload))}
	cases := map[string]Artifact{
		"plain": {URL: srv.URL + "/plain.onnx"},
		"zip":   {URL: srv.URL + "/pack.zip", Member: "wanted.onnx"},
		"tgz":   {URL: srv.URL + "/pack.tgz", Member: "pack-1.0/lib/wanted.so"},
	}
	for name, a := range cases {
		a.SHA256, a.Size = want.SHA256, want.Size
		dir := t.TempDir()
		dst := filepath.Join(dir, "sub", "model.bin")
		var calls int
		if err := Fetch(context.Background(), nil, a, dst, func(done, total int64) { calls++ }); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if got, _ := os.ReadFile(dst); !bytes.Equal(got, payload) {
			t.Fatalf("%s: wrong content", name)
		}
		if calls == 0 {
			t.Fatalf("%s: progress never reported", name)
		}
		if err := Verify(dst, a); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		entries, _ := os.ReadDir(filepath.Dir(dst))
		if len(entries) != 1 {
			t.Fatalf("%s: temporary files left behind: %v", name, entries)
		}
		// a second fetch must not touch the network
		before := *requests
		if err := Fetch(context.Background(), nil, a, dst, nil); err != nil {
			t.Fatalf("%s: refetch of a verified file: %v", name, err)
		}
		if *requests != before {
			t.Fatalf("%s: a verified file was downloaded again", name)
		}
	}
}

func TestFetchRejectsBadDigestAndKeepsNothing(t *testing.T) {
	srv, _ := serve(t, map[string][]byte{"/m.onnx": payload})
	dst := filepath.Join(t.TempDir(), "m.onnx")
	a := Artifact{URL: srv.URL + "/m.onnx", SHA256: strings.Repeat("0", 64), Size: int64(len(payload))}
	err := Fetch(context.Background(), nil, a, dst, nil)
	if !errors.Is(err, ErrChecksum) {
		t.Fatalf("got %v, want a checksum error", err)
	}
	if entries, _ := os.ReadDir(filepath.Dir(dst)); len(entries) != 0 {
		t.Fatalf("files left behind after a failed fetch: %v", entries)
	}
	a.SHA256, a.Size = digest(payload), 1
	if err := Fetch(context.Background(), nil, a, dst, nil); !errors.Is(err, ErrChecksum) {
		t.Fatalf("size mismatch: got %v", err)
	}
}

func TestFetchErrors(t *testing.T) {
	srv, _ := serve(t, map[string][]byte{
		"/pack.zip": makeZip(t, map[string][]byte{"a": payload}),
		"/pack.tgz": makeTgz(t, map[string][]byte{"a": payload}),
		"/pack.rar": payload,
	})
	ok := Artifact{SHA256: digest(payload), Size: int64(len(payload))}
	for name, a := range map[string]Artifact{
		"404":            {URL: srv.URL + "/missing"},
		"no such member": {URL: srv.URL + "/pack.zip", Member: "b"},
		"no such tgz":    {URL: srv.URL + "/pack.tgz", Member: "b"},
		"unknown format": {URL: srv.URL + "/pack.rar", Member: "a"},
		"bad url":        {URL: "http://127.0.0.1:1/x"},
	} {
		a.SHA256, a.Size = ok.SHA256, ok.Size
		if err := Fetch(context.Background(), nil, a, filepath.Join(t.TempDir(), "m"), nil); err == nil {
			t.Fatalf("%s: expected an error", name)
		}
	}
}

func TestFetchHonoursContext(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(payload[:100])
		w.(http.Flusher).Flush()
		<-release
	}))
	t.Cleanup(func() { close(release); srv.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	dst := filepath.Join(t.TempDir(), "m")
	err := Fetch(ctx, nil, Artifact{URL: srv.URL, SHA256: digest(payload), Size: int64(len(payload))}, dst, nil)
	if err == nil || !errors.Is(err, context.DeadlineExceeded) && !strings.Contains(err.Error(), "deadline") {
		t.Fatalf("got %v, want a deadline error", err)
	}
	if _, err := os.Stat(dst + ".part"); err == nil {
		t.Fatal("partial file left behind")
	}
}

func TestManifestsAreConsistent(t *testing.T) {
	seen := map[string]bool{}
	for _, m := range All {
		if seen[m.Name] {
			t.Fatalf("duplicate model name %s", m.Name)
		}
		seen[m.Name] = true
		if len(m.Artifact.SHA256) != 64 || m.Artifact.Size <= 0 || m.Artifact.URL == "" || m.File == "" {
			t.Fatalf("%s: incomplete artifact", m.Name)
		}
		if m.Kind.Detector() != (m.Embedding == 0) {
			t.Fatalf("%s: detector/embedding mismatch", m.Name)
		}
		if strings.Contains(m.Artifact.URL, ".zip") == (m.Artifact.Member == "") {
			t.Fatalf("%s: archives need a member and plain files must not have one", m.Name)
		}
		fp := m.Fingerprint()
		if !strings.HasPrefix(fp, string(m.Kind)+":"+m.Name+":") {
			t.Fatalf("fingerprint %q", fp)
		}
		if got, ok := Lookup(m.Name); !ok || got != m {
			t.Fatalf("Lookup(%s) failed", m.Name)
		}
	}
	for _, p := range Presets {
		if !p.Detector.Kind.Detector() || p.Embedder.Kind.Detector() {
			t.Fatalf("preset %s pairs the wrong kinds", p.Name)
		}
		if got, ok := LookupPreset(p.Name); !ok || got.Name != p.Name {
			t.Fatalf("LookupPreset(%s) failed", p.Name)
		}
	}
	if def, ok := LookupPreset("default"); !ok || def.NonCommercial() {
		t.Fatal("the default preset must use permissively licensed weights")
	}
	for _, platform := range []string{"linux/amd64", "linux/arm64", "darwin/arm64", "windows/amd64"} {
		a, ok := Runtime(platform)
		if !ok || a.Member == "" || len(a.SHA256) != 64 || a.Size == 0 {
			t.Fatalf("runtime artifact for %s is incomplete", platform)
		}
	}
	if _, ok := Runtime("plan9/mips"); ok {
		t.Fatal("unknown platforms must not have a runtime")
	}
}
