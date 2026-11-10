// Package embed turns an aligned face into a fixed-length vector with an
// ONNX embedding model.
package embed

import (
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"sync"

	"github.com/muhriddinsalohiddin/likeness/internal/align"
	"github.com/muhriddinsalohiddin/likeness/internal/models"
	"github.com/muhriddinsalohiddin/likeness/internal/ortx"
	"github.com/muhriddinsalohiddin/likeness/internal/vision"
)

// Options tune one embedding call.
type Options struct {
	// NoAntiAlias samples the picture without a pre-filter, as reference
	// implementations do. Used by parity tests.
	NoAntiAlias bool
}

// Embedder runs one embedding model. It is safe for concurrent use.
type Embedder struct {
	m    *models.Manifest
	sess *ortx.Session
	size int
	dim  int
	pool sync.Pool
}

type worker struct {
	in   *ortx.Tensor
	crop []float32
	out  ortx.Outputs
}

// New loads the embedder described by m from dir.
func New(rt *ortx.Runtime, m *models.Manifest, dir string) (*Embedder, error) {
	if m.Kind.Detector() {
		return nil, fmt.Errorf("embed: %s is a %s model, not an embedder", m.Name, m.Kind)
	}
	sess, err := rt.NewSessionFromFile(filepath.Join(dir, m.File), ortx.SessionOptions{})
	if err != nil {
		return nil, err
	}
	e := &Embedder{m: m, sess: sess, size: m.Input, dim: m.Embedding}
	if err := e.check(); err != nil {
		sess.Close()
		return nil, fmt.Errorf("embed: %s: %w", m.Name, err)
	}
	return e, nil
}

func (e *Embedder) check() error {
	in, out := e.sess.Inputs(), e.sess.Outputs()
	if len(in) != 1 || len(in[0].Dims) != 4 || in[0].Dims[1] != 3 || in[0].Type != ortx.Float32 {
		return fmt.Errorf("expected one float32 NCHW input with 3 channels, got %+v", in)
	}
	if h, w := in[0].Dims[2], in[0].Dims[3]; h > 0 && (h != w || int(h) != e.size) {
		return fmt.Errorf("model input is %dx%d, manifest says %d", w, h, e.size)
	}
	if len(out) != 1 || len(out[0].Dims) != 2 || out[0].Type != ortx.Float32 {
		return fmt.Errorf("expected one float32 [1, n] output, got %+v", out)
	}
	if d := out[0].Dims[1]; d > 0 && int(d) != e.dim {
		return fmt.Errorf("model emits %d values, manifest says %d", d, e.dim)
	}
	return nil
}

// Manifest returns the model description.
func (e *Embedder) Manifest() *models.Manifest { return e.m }

// Dim is the length of the vectors this model produces.
func (e *Embedder) Dim() int { return e.dim }

// Fingerprint identifies the weights; vectors from different fingerprints
// must never be compared.
func (e *Embedder) Fingerprint() string { return e.m.Fingerprint() }

// Close releases the model. It must not overlap with Embed.
func (e *Embedder) Close() { e.sess.Close() }

// CropSize is the side of the square aligned crop the model takes.
func (e *Embedder) CropSize() int { return e.size }

// Crop writes the aligned face as raw R, G and B planes on the 0..255 scale,
// 3*CropSize()*CropSize() values. The crop is independent of the model's own
// normalisation, so its quality can be judged before it is embedded.
func (e *Embedder) Crop(img *vision.Image, landmarks [5]align.Point, dst []float32, o Options) error {
	if img == nil {
		return errors.New("embed: nil image")
	}
	if len(dst) != 3*e.size*e.size {
		return fmt.Errorf("embed: crop buffer has length %d, want %d", len(dst), 3*e.size*e.size)
	}
	toCrop, err := align.Crop(landmarks, e.size)
	if err != nil {
		return fmt.Errorf("embed: %w", err)
	}
	if err := img.Warp(dst, e.size, toCrop, vision.RawRGB, vision.Options{NoAntiAlias: o.NoAntiAlias}); err != nil {
		return fmt.Errorf("embed: %w", err)
	}
	return nil
}

// EmbedCrop runs the model on a crop made by Crop and writes the
// L2-normalised vector to dst, which must have length Dim.
func (e *Embedder) EmbedCrop(crop, dst []float32) error {
	if len(crop) != 3*e.size*e.size {
		return fmt.Errorf("embed: crop has length %d, want %d", len(crop), 3*e.size*e.size)
	}
	if len(dst) != e.dim {
		return fmt.Errorf("embed: destination has length %d, want %d", len(dst), e.dim)
	}
	w, err := e.worker()
	if err != nil {
		return err
	}
	defer e.pool.Put(w)
	return e.run(w, crop, dst)
}

// Embed aligns the face whose landmarks are given, runs the model and writes
// the L2-normalised vector to dst, which must have length Dim.
func (e *Embedder) Embed(img *vision.Image, landmarks [5]align.Point, dst []float32, o Options) error {
	if len(dst) != e.dim {
		return fmt.Errorf("embed: destination has length %d, want %d", len(dst), e.dim)
	}
	w, err := e.worker()
	if err != nil {
		return err
	}
	defer e.pool.Put(w)
	if err := e.Crop(img, landmarks, w.crop, o); err != nil {
		return err
	}
	return e.run(w, w.crop, dst)
}

func (e *Embedder) worker() (*worker, error) {
	if w, _ := e.pool.Get().(*worker); w != nil {
		return w, nil
	}
	t, err := ortx.NewFloat32(1, 3, int64(e.size), int64(e.size))
	if err != nil {
		return nil, err
	}
	return &worker{in: t, crop: make([]float32, 3*e.size*e.size)}, nil
}

func (e *Embedder) run(w *worker, crop, dst []float32) error {
	e.m.Norm.Apply(w.in.Float32(), crop, e.size*e.size)
	if err := e.sess.Run([]*ortx.Tensor{w.in}, &w.out); err != nil {
		return err
	}
	if len(w.out.Data[0]) != e.dim {
		return fmt.Errorf("embed: model returned %d values, want %d", len(w.out.Data[0]), e.dim)
	}
	Normalize(dst, w.out.Data[0])
	return nil
}

// Normalize writes src scaled to unit length into dst. A zero vector stays
// zero.
func Normalize(dst, src []float32) {
	var sum float64
	for _, v := range src {
		sum += float64(v) * float64(v)
	}
	scale := float32(0)
	if sum > 0 {
		scale = float32(1 / math.Sqrt(sum))
	}
	for i, v := range src {
		dst[i] = v * scale
	}
}
