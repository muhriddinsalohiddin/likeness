package models

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// ErrChecksum means a downloaded or installed file does not match its
// manifest.
var ErrChecksum = errors.New("models: checksum mismatch")

// Verify checks that the file at path has the size and digest the artifact
// promises.
func Verify(path string, a Artifact) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return err
	}
	if a.Size > 0 && st.Size() != a.Size {
		return fmt.Errorf("%w: %s is %d bytes, want %d", ErrChecksum, path, st.Size(), a.Size)
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	if got := hex.EncodeToString(h.Sum(nil)); !strings.EqualFold(got, a.SHA256) {
		return fmt.Errorf("%w: %s has digest %s, want %s", ErrChecksum, path, got, a.SHA256)
	}
	return nil
}

// Installed returns the path of a model inside dir, or an error if the file
// is missing or corrupt.
func Installed(dir string, m *Manifest) (string, error) {
	path := filepath.Join(dir, m.File)
	if err := Verify(path, m.Artifact); err != nil {
		return "", err
	}
	return path, nil
}

// Progress reports download progress. total is -1 when unknown.
type Progress func(done, total int64)

// Fetch downloads an artifact to dst, verifying the digest before the file
// appears under its final name. A file that already verifies is left alone.
// A nil client means http.DefaultClient.
func Fetch(ctx context.Context, client *http.Client, a Artifact, dst string, progress Progress) error {
	if err := Verify(dst, a); err == nil {
		return nil
	}
	if client == nil {
		client = http.DefaultClient
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, a.URL, nil)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("models: download %s: %w", a.URL, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("models: download %s: HTTP %s", a.URL, resp.Status)
	}
	body := io.Reader(&progressReader{r: resp.Body, total: resp.ContentLength, fn: progress})

	part := dst + ".part"
	defer os.Remove(part)
	out, err := os.Create(part)
	if err != nil {
		return err
	}
	digest := sha256.New()
	switch {
	case a.Member == "":
		_, err = io.Copy(io.MultiWriter(out, digest), body)
	case strings.HasSuffix(a.URL, ".zip"):
		err = extractZip(body, a.Member, dst+".archive", out, digest)
	case strings.HasSuffix(a.URL, ".tgz") || strings.HasSuffix(a.URL, ".tar.gz"):
		err = extractTgz(body, a.Member, out, digest)
	default:
		err = fmt.Errorf("models: do not know how to unpack %s", a.URL)
	}
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return err
	}
	if got := hex.EncodeToString(digest.Sum(nil)); !strings.EqualFold(got, a.SHA256) {
		return fmt.Errorf("%w: %s downloaded with digest %s, want %s", ErrChecksum, a.URL, got, a.SHA256)
	}
	if err := Verify(part, a); err != nil {
		return err
	}
	return os.Rename(part, dst)
}

type progressReader struct {
	r     io.Reader
	done  int64
	total int64
	fn    Progress
}

func (p *progressReader) Read(b []byte) (int, error) {
	n, err := p.r.Read(b)
	p.done += int64(n)
	if p.fn != nil && n > 0 {
		p.fn(p.done, p.total)
	}
	return n, err
}

// extractZip needs random access, so the archive is spooled to disk first.
func extractZip(body io.Reader, member, spool string, out io.Writer, digest hash.Hash) error {
	f, err := os.Create(spool)
	if err != nil {
		return err
	}
	defer os.Remove(spool)
	defer f.Close()
	size, err := io.Copy(f, body)
	if err != nil {
		return err
	}
	zr, err := zip.NewReader(f, size)
	if err != nil {
		return fmt.Errorf("models: %s: %w", spool, err)
	}
	for _, zf := range zr.File {
		if zf.Name != member {
			continue
		}
		rc, err := zf.Open()
		if err != nil {
			return err
		}
		defer rc.Close()
		_, err = io.Copy(io.MultiWriter(out, digest), rc)
		return err
	}
	return fmt.Errorf("models: archive has no member %q", member)
}

func extractTgz(body io.Reader, member string, out io.Writer, digest hash.Hash) error {
	gz, err := gzip.NewReader(body)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			return fmt.Errorf("models: archive has no member %q", member)
		}
		if err != nil {
			return err
		}
		if strings.TrimPrefix(hdr.Name, "./") != member || hdr.Typeflag != tar.TypeReg {
			continue
		}
		_, err = io.Copy(io.MultiWriter(out, digest), tr)
		return err
	}
}
