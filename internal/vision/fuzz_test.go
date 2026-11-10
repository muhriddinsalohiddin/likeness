package vision

import (
	"encoding/binary"
	"image"
	"math/rand"
	"os"
	"testing"

	"github.com/muhriddinsalohiddin/likeness/internal/align"
)

// Run with: go test ./internal/vision -fuzz FuzzDecode -fuzztime 1m
func FuzzDecode(f *testing.F) {
	rng := rand.New(rand.NewSource(1))
	jpg := encodeJPEG(f, smoothRGBA(rng, 48, 40), 80)
	f.Add(jpg)
	f.Add(withOrientation(jpg, 6, binary.LittleEndian))
	f.Add(encodeJPEG(f, smoothGray(rng, 33, 47), 60))
	f.Add(encodeJPEG(f, smoothYCbCr(rng, 40, 40, image.YCbCrSubsampleRatio444), 95))
	for _, name := range []string{"gradient.png", "gradient_lossy.webp", "gradient_lossless.webp"} {
		b, err := os.ReadFile("testdata/" + name)
		if err != nil {
			f.Fatal(err)
		}
		f.Add(b)
	}
	f.Add(pngHeader(60000, 60000))

	lim := Limits{MaxPixels: 1 << 20, MaxSide: 2048}
	f.Fuzz(func(t *testing.T, data []byte) {
		m, err := Decode(data, lim)
		if err != nil {
			return
		}
		w, h := m.Size()
		if w < 16 || h < 16 || w > 2048 || h > 2048 || w*h > 1<<20 {
			t.Fatalf("decoded %dx%d despite the limits", w, h)
		}
		// whatever decoded must be safe to sample, including far outside
		det := make([]float32, 3*32*32)
		m.Letterbox(det, 32, 32, raw, Options{})
		crop := make([]float32, 3*16*16)
		if err := m.Warp(crop, 16, align.Affine{A: 0.3, B: -0.2, C: -5, D: 0.2, E: 0.3, F: 7}, raw, Options{}); err != nil {
			t.Fatal(err)
		}
	})
}

func FuzzExifOrientation(f *testing.F) {
	jpg := encodeJPEG(f, image.NewGray(image.Rect(0, 0, 16, 16)), 80)
	f.Add(jpg)
	f.Add(withOrientation(jpg, 5, binary.BigEndian))
	f.Add(withOrientation(jpg, 8, binary.LittleEndian)[:40])
	f.Fuzz(func(t *testing.T, data []byte) {
		if o := exifOrientation(data); o < 1 || o > 8 {
			t.Fatalf("orientation %d out of range", o)
		}
	})
}
