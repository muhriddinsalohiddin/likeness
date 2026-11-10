package vision

import (
	"image"
	"math/rand"
	"testing"

	"github.com/muhriddinsalohiddin/likeness/internal/align"
)

var scrfdNorm = Uniform(127.5, 128, false)

func frame720p(b *testing.B) (*image.YCbCr, []byte) {
	img := smoothYCbCr(rand.New(rand.NewSource(1)), 1280, 720, image.YCbCrSubsampleRatio420)
	return img, encodeJPEG(b, img, 88)
}

func BenchmarkDecodeJPEG720p(b *testing.B) {
	_, jpg := frame720p(b)
	b.SetBytes(int64(len(jpg)))
	for b.Loop() {
		if _, err := Decode(jpg, Limits{}); err != nil {
			b.Fatal(err)
		}
	}
}

// Every request decodes a fresh picture, so the reduced copy is rebuilt each
// time: wrap the pixels anew to keep the cache from flattering the numbers.
func benchLetterbox(b *testing.B, side int, o Options) {
	img, _ := frame720p(b)
	dst := make([]float32, 3*side*side)
	for b.Loop() {
		FromImage(img, 1).Letterbox(dst, side, side, scrfdNorm, o)
	}
}

func BenchmarkLetterbox720pTo320(b *testing.B) { benchLetterbox(b, 320, Options{}) }
func BenchmarkLetterbox720pTo320NoAA(b *testing.B) {
	benchLetterbox(b, 320, Options{NoAntiAlias: true})
}
func BenchmarkLetterbox720pTo640(b *testing.B)       { benchLetterbox(b, 640, Options{}) }
func BenchmarkLetterbox720pTo320Serial(b *testing.B) { benchLetterbox(b, 320, Options{Workers: 1}) }
func BenchmarkLetterboxRotated720pTo320(b *testing.B) { // phone portrait shot: general path
	img, _ := frame720p(b)
	dst := make([]float32, 3*320*320)
	for b.Loop() {
		FromImage(img, 6).Letterbox(dst, 320, 320, scrfdNorm, Options{})
	}
}

func BenchmarkWarpFace112(b *testing.B) {
	img, _ := frame720p(b)
	// a face about 400 px wide in the middle of the frame
	var lm [5]align.Point
	for i, r := range align.Reference112 {
		lm[i] = align.Point{X: (r.X-56)*3.6 + 640, Y: (r.Y-56)*3.6 + 360}
	}
	toCrop, err := align.Crop(lm, 112)
	if err != nil {
		b.Fatal(err)
	}
	dst := make([]float32, 3*112*112)
	for b.Loop() {
		if err := FromImage(img, 1).Warp(dst, 112, toCrop, Uniform(127.5, 127.5, false), Options{}); err != nil {
			b.Fatal(err)
		}
	}
}
