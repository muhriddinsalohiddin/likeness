// Package vision turns encoded pictures into model input tensors.
//
// Pixels stay in the layout the decoder produced (planar YCbCr for JPEG) and
// are converted to normalised RGB only for the samples a tensor needs, so a
// request never pays for a full-resolution colour conversion.
package vision

import (
	"image"
	"image/draw"
	"sync"

	"github.com/muhriddinsalohiddin/likeness/internal/align"
)

type kind uint8

const (
	kindYCbCr kind = iota
	kindGray
	kindRGBA
)

// Image is a decoded picture. Coordinates passed to its methods refer to the
// upright picture, after the EXIF orientation has been applied.
type Image struct {
	kind   kind
	w, h   int // stored size, before orientation
	orient int // EXIF orientation, 1..8

	y, cb, cr        []uint8 // kindGray uses y only
	yStride, cStride int
	hdiv, vdiv       int // chroma subsampling factors

	pix    []uint8 // kindRGBA: 4 bytes per pixel, alpha ignored
	stride int

	mu     sync.Mutex
	levels map[int]*Image // box-filtered copies, keyed by reduction factor
}

// FromImage wraps a decoded image. orientation is the EXIF orientation tag;
// values outside 1..8 mean upright.
func FromImage(src image.Image, orientation int) *Image {
	if orientation < 1 || orientation > 8 {
		orientation = 1
	}
	b := src.Bounds()
	m := &Image{w: b.Dx(), h: b.Dy(), orient: orientation}
	origin := b.Min == image.Point{}

	switch s := src.(type) {
	case *image.YCbCr:
		if hd, vd, ok := chromaDiv(s.SubsampleRatio); ok && origin {
			m.kind, m.y, m.cb, m.cr = kindYCbCr, s.Y, s.Cb, s.Cr
			m.yStride, m.cStride, m.hdiv, m.vdiv = s.YStride, s.CStride, hd, vd
			return m
		}
	case *image.Gray:
		if origin {
			m.kind, m.y, m.yStride = kindGray, s.Pix, s.Stride
			return m
		}
	case *image.RGBA:
		if origin {
			m.kind, m.pix, m.stride = kindRGBA, s.Pix, s.Stride
			return m
		}
	case *image.NRGBA:
		if origin {
			m.kind, m.pix, m.stride = kindRGBA, s.Pix, s.Stride
			return m
		}
	}
	rgba := image.NewRGBA(image.Rect(0, 0, m.w, m.h))
	draw.Draw(rgba, rgba.Bounds(), src, b.Min, draw.Src)
	m.kind, m.pix, m.stride = kindRGBA, rgba.Pix, rgba.Stride
	return m
}

func chromaDiv(r image.YCbCrSubsampleRatio) (hdiv, vdiv int, ok bool) {
	switch r {
	case image.YCbCrSubsampleRatio444:
		return 1, 1, true
	case image.YCbCrSubsampleRatio422:
		return 2, 1, true
	case image.YCbCrSubsampleRatio420:
		return 2, 2, true
	case image.YCbCrSubsampleRatio440:
		return 1, 2, true
	case image.YCbCrSubsampleRatio411:
		return 4, 1, true
	case image.YCbCrSubsampleRatio410:
		return 4, 2, true
	}
	return 0, 0, false
}

// Size returns the width and height of the upright picture.
func (m *Image) Size() (w, h int) {
	if m.orient >= 5 {
		return m.h, m.w
	}
	return m.w, m.h
}

// storedFromUpright maps upright coordinates to stored pixel coordinates.
func (m *Image) storedFromUpright() align.Affine {
	w, h := float32(m.w-1), float32(m.h-1)
	switch m.orient {
	case 2: // mirrored horizontally
		return align.Affine{A: -1, C: w, E: 1}
	case 3: // rotated 180
		return align.Affine{A: -1, C: w, E: -1, F: h}
	case 4: // mirrored vertically
		return align.Affine{A: 1, E: -1, F: h}
	case 5: // transposed
		return align.Affine{B: 1, D: 1}
	case 6: // stored picture must turn 90 degrees clockwise to be upright
		return align.Affine{B: 1, D: -1, F: h}
	case 7: // transversed
		return align.Affine{B: -1, C: w, D: -1, F: h}
	case 8: // stored picture must turn 90 degrees counter-clockwise
		return align.Affine{B: -1, C: w, D: 1}
	}
	return align.Identity
}

func (m *Image) chromaSize() (w, h int) {
	return (m.w + m.hdiv - 1) / m.hdiv, (m.h + m.vdiv - 1) / m.vdiv
}
