package vision

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"math"
	"math/rand"
	"testing"

	"github.com/muhriddinsalohiddin/likeness/internal/align"
)

var raw = Uniform(0, 1, false) // tensors hold plain 0..255 RGB values

func smooth(rng *rand.Rand, x, y int, phase float64) uint8 {
	v := 128 + 70*math.Sin(float64(x)*0.11+phase) + 40*math.Cos(float64(y)*0.07-phase) + rng.NormFloat64()*2
	return uint8(math.Min(math.Max(v, 0), 255))
}

func smoothYCbCr(rng *rand.Rand, w, h int, ratio image.YCbCrSubsampleRatio) *image.YCbCr {
	img := image.NewYCbCr(image.Rect(0, 0, w, h), ratio)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Y[img.YOffset(x, y)] = smooth(rng, x, y, 0)
			c := img.COffset(x, y)
			img.Cb[c] = smooth(rng, x, y, 1.3)
			img.Cr[c] = smooth(rng, x, y, 2.9)
		}
	}
	return img
}

func smoothGray(rng *rand.Rand, w, h int) *image.Gray {
	img := image.NewGray(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Pix[img.PixOffset(x, y)] = smooth(rng, x, y, 0.4)
		}
	}
	return img
}

func smoothRGBA(rng *rand.Rand, w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetRGBA(x, y, color.RGBA{smooth(rng, x, y, 0), smooth(rng, x, y, 2), smooth(rng, x, y, 4), 255})
		}
	}
	return img
}

// uniqueRGBA gives every pixel its own colour so that a misplaced pixel
// cannot go unnoticed.
func uniqueRGBA(w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetRGBA(x, y, color.RGBA{uint8(x * 11), uint8(y * 13), uint8(x + y*w), 255})
		}
	}
	return img
}

func encodeJPEG(t testing.TB, img image.Image, quality int) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: quality}); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func maxDiff(a, b []float32) (float32, int) {
	var worst float32
	at := -1
	for i := range a {
		if d := float32(math.Abs(float64(a[i] - b[i]))); d > worst {
			worst, at = d, i
		}
	}
	return worst, at
}

// naiveSample is sample() written straight from the definition: float64
// blending, every tap bounds-checked, no fast paths.
func naiveSample(m *Image, tw, th, cw, ch int, toLuma align.Affine, replicate bool, n Norm) []float32 {
	out := make([]float32, 3*tw*th)
	nz := n.prepare()
	tap := func(pix []uint8, stride, step, off, w, h, x, y int, outside float64) float64 {
		if x < 0 || y < 0 || x >= w || y >= h {
			return outside
		}
		return float64(pix[y*stride+x*step+off])
	}
	bilinear := func(pix []uint8, stride, step, off, w, h int, x, y float32, outside float64) float64 {
		fx, fy := float64(x), float64(y)
		if replicate {
			fx = math.Min(math.Max(fx, 0), float64(w-1))
			fy = math.Min(math.Max(fy, 0), float64(h-1))
		}
		x0, y0 := math.Floor(fx), math.Floor(fy)
		ax, ay := fx-x0, fy-y0
		ix, iy := int(x0), int(y0)
		top := tap(pix, stride, step, off, w, h, ix, iy, outside)*(1-ax) + tap(pix, stride, step, off, w, h, ix+1, iy, outside)*ax
		bot := tap(pix, stride, step, off, w, h, ix, iy+1, outside)*(1-ax) + tap(pix, stride, step, off, w, h, ix+1, iy+1, outside)*ax
		return top*(1-ay) + bot*ay
	}
	clamp := func(v float64) float64 { return math.Min(math.Max(v, 0), 255) }

	for v := 0; v < ch; v++ {
		for u := 0; u < cw; u++ {
			x := toLuma.A*float32(u) + toLuma.B*float32(v) + toLuma.C
			y := toLuma.D*float32(u) + toLuma.E*float32(v) + toLuma.F
			var r, g, b float64
			switch m.kind {
			case kindYCbCr:
				cwid, chei := m.chromaSize()
				cx, cy := (x+0.5)/float32(m.hdiv)-0.5, (y+0.5)/float32(m.vdiv)-0.5
				l := bilinear(m.y, m.yStride, 1, 0, m.w, m.h, x, y, 0)
				cb := bilinear(m.cb, m.cStride, 1, 0, cwid, chei, cx, cy, 128) - 128
				cr := bilinear(m.cr, m.cStride, 1, 0, cwid, chei, cx, cy, 128) - 128
				r, g, b = clamp(l+1.402*cr), clamp(l-0.344136*cb-0.714136*cr), clamp(l+1.772*cb)
			case kindGray:
				l := bilinear(m.y, m.yStride, 1, 0, m.w, m.h, x, y, 0)
				r, g, b = l, l, l
			case kindRGBA:
				r = bilinear(m.pix, m.stride, 4, 0, m.w, m.h, x, y, 0)
				g = bilinear(m.pix, m.stride, 4, 1, m.w, m.h, x, y, 0)
				b = bilinear(m.pix, m.stride, 4, 2, m.w, m.h, x, y, 0)
			}
			i := v*tw + u
			out[nz.plane[0]*tw*th+i] = (float32(r) - nz.mean[0]) * nz.inv[0]
			out[nz.plane[1]*tw*th+i] = (float32(g) - nz.mean[1]) * nz.inv[1]
			out[nz.plane[2]*tw*th+i] = (float32(b) - nz.mean[2]) * nz.inv[2]
		}
	}
	return out
}
