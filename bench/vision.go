package main

import (
	"bytes"
	"image"
	"image/jpeg"
	"math"
	"math/rand"
	"sync"
)

// Pre is the pixel normalisation a model expects: (v - Mean) / Std, in RGB
// channel order unless BGR is set.
type Pre struct {
	Mean, Std float32
	BGR       bool
}

func chromaDiv(r image.YCbCrSubsampleRatio) (int, int) {
	switch r {
	case image.YCbCrSubsampleRatio422:
		return 2, 1
	case image.YCbCrSubsampleRatio420:
		return 2, 2
	case image.YCbCrSubsampleRatio440:
		return 1, 2
	case image.YCbCrSubsampleRatio411:
		return 4, 1
	case image.YCbCrSubsampleRatio410:
		return 4, 2
	}
	return 1, 1
}

type axisTab struct {
	i0, i1 []int
	w      []float32
}

// mkAxis maps dstN output samples onto a plane that is the source axis
// subsampled by div. scale is source pixels per output pixel.
func mkAxis(dstN, srcN int, scale float64, div int) axisTab {
	pn := (srcN + div - 1) / div
	t := axisTab{make([]int, dstN), make([]int, dstN), make([]float32, dstN)}
	for d := 0; d < dstN; d++ {
		p := ((float64(d)+0.5)*scale)/float64(div) - 0.5
		if p < 0 {
			p = 0
		}
		if p > float64(pn-1) {
			p = float64(pn - 1)
		}
		i0 := int(p)
		i1 := i0 + 1
		if i1 > pn-1 {
			i1 = pn - 1
		}
		t.i0[d], t.i1[d], t.w[d] = i0, i1, float32(p-float64(i0))
	}
	return t
}

func clamp255(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return v
}

// resizeYCbCr resamples src straight into a CHW float32 tensor (tw x th),
// filling the top-left cw x ch area. Colour conversion and normalisation are
// fused into the same pass: only output pixels are ever converted to RGB.
func resizeYCbCr(src *image.YCbCr, dst []float32, tw, th, cw, ch int, p Pre, workers int) {
	W, H := src.Rect.Dx(), src.Rect.Dy()
	hd, vd := chromaDiv(src.SubsampleRatio)
	sx, sy := float64(W)/float64(cw), float64(H)/float64(ch)
	xl, xc := mkAxis(cw, W, sx, 1), mkAxis(cw, W, sx, hd)
	yl, yc := mkAxis(ch, H, sy, 1), mkAxis(ch, H, sy, vd)
	plane := tw * th
	cR, cB := 0, 2
	if p.BGR {
		cR, cB = 2, 0
	}
	inv := 1 / p.Std

	rows := func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			ya, yb, wy := src.Y[yl.i0[y]*src.YStride:], src.Y[yl.i1[y]*src.YStride:], yl.w[y]
			cba, cbb := src.Cb[yc.i0[y]*src.CStride:], src.Cb[yc.i1[y]*src.CStride:]
			cra, crb := src.Cr[yc.i0[y]*src.CStride:], src.Cr[yc.i1[y]*src.CStride:]
			wc := yc.w[y]
			o := y * tw
			for x := 0; x < cw; x++ {
				x0, x1, wx := xl.i0[x], xl.i1[x], xl.w[x]
				t := float32(ya[x0]) + (float32(ya[x1])-float32(ya[x0]))*wx
				b := float32(yb[x0]) + (float32(yb[x1])-float32(yb[x0]))*wx
				Y := t + (b-t)*wy

				u0, u1, wu := xc.i0[x], xc.i1[x], xc.w[x]
				t = float32(cba[u0]) + (float32(cba[u1])-float32(cba[u0]))*wu
				b = float32(cbb[u0]) + (float32(cbb[u1])-float32(cbb[u0]))*wu
				cb := t + (b-t)*wc - 128
				t = float32(cra[u0]) + (float32(cra[u1])-float32(cra[u0]))*wu
				b = float32(crb[u0]) + (float32(crb[u1])-float32(crb[u0]))*wu
				cr := t + (b-t)*wc - 128

				dst[cR*plane+o+x] = (clamp255(Y+1.402*cr) - p.Mean) * inv
				dst[plane+o+x] = (clamp255(Y-0.344136*cb-0.714136*cr) - p.Mean) * inv
				dst[cB*plane+o+x] = (clamp255(Y+1.772*cb) - p.Mean) * inv
			}
		}
	}
	if workers <= 1 {
		rows(0, ch)
		return
	}
	var wg sync.WaitGroup
	step := (ch + workers - 1) / workers
	for y0 := 0; y0 < ch; y0 += step {
		y1 := min(y0+step, ch)
		wg.Add(1)
		go func() { defer wg.Done(); rows(y0, y1) }()
	}
	wg.Wait()
}

// arcfaceTemplate is the canonical 5-point layout for a 112x112 aligned face.
var arcfaceTemplate = [5][2]float32{
	{38.2946, 51.6963}, {73.5318, 51.5014}, {56.0252, 71.7366},
	{41.5493, 92.3655}, {70.7299, 92.2041},
}

// similarity returns the least-squares similarity transform (uniform scale,
// rotation, translation) mapping src onto dst:  x' = a*x - b*y + tx,
// y' = b*x + a*y + ty. For point sets without reflection this equals Umeyama.
func similarity(src, dst [5][2]float32) (a, b, tx, ty float32) {
	var mx, my, nx, ny float32
	for i := 0; i < 5; i++ {
		mx += src[i][0]
		my += src[i][1]
		nx += dst[i][0]
		ny += dst[i][1]
	}
	mx, my, nx, ny = mx/5, my/5, nx/5, ny/5
	var sa, sb, den float32
	for i := 0; i < 5; i++ {
		px, py := src[i][0]-mx, src[i][1]-my
		qx, qy := dst[i][0]-nx, dst[i][1]-ny
		sa += px*qx + py*qy
		sb += px*qy - py*qx
		den += px*px + py*py
	}
	a, b = sa/den, sb/den
	tx = nx - (a*mx - b*my)
	ty = ny - (b*mx + a*my)
	return
}

// warpYCbCr writes the aligned size x size face crop into a CHW tensor by
// inverse-mapping every output pixel into the source image.
func warpYCbCr(src *image.YCbCr, dst []float32, size int, kps [5][2]float32, p Pre) {
	a, b, tx, ty := similarity(kps, arcfaceTemplate)
	det := a*a + b*b
	ia, ib := a/det, -b/det // inverse rotation+scale

	W, H := src.Rect.Dx(), src.Rect.Dy()
	hd, vd := chromaDiv(src.SubsampleRatio)
	cw, chh := (W+hd-1)/hd, (H+vd-1)/vd
	plane := size * size
	cR, cB := 0, 2
	if p.BGR {
		cR, cB = 2, 0
	}
	inv := 1 / p.Std

	sample := func(pl []uint8, stride, pw, ph int, x, y float32) float32 {
		if x < 0 {
			x = 0
		}
		if y < 0 {
			y = 0
		}
		if x > float32(pw-1) {
			x = float32(pw - 1)
		}
		if y > float32(ph-1) {
			y = float32(ph - 1)
		}
		x0, y0 := int(x), int(y)
		x1, y1 := min(x0+1, pw-1), min(y0+1, ph-1)
		wx, wy := x-float32(x0), y-float32(y0)
		t := float32(pl[y0*stride+x0]) + (float32(pl[y0*stride+x1])-float32(pl[y0*stride+x0]))*wx
		bt := float32(pl[y1*stride+x0]) + (float32(pl[y1*stride+x1])-float32(pl[y1*stride+x0]))*wx
		return t + (bt-t)*wy
	}

	for v := 0; v < size; v++ {
		for u := 0; u < size; u++ {
			du, dv := float32(u)-tx, float32(v)-ty
			sx := ia*du - ib*dv
			sy := ib*du + ia*dv
			Y := sample(src.Y, src.YStride, W, H, sx, sy)
			cx, cy := (sx+0.5)/float32(hd)-0.5, (sy+0.5)/float32(vd)-0.5
			cb := sample(src.Cb, src.CStride, cw, chh, cx, cy) - 128
			cr := sample(src.Cr, src.CStride, cw, chh, cx, cy) - 128
			o := v*size + u
			dst[cR*plane+o] = (clamp255(Y+1.402*cr) - p.Mean) * inv
			dst[plane+o] = (clamp255(Y-0.344136*cb-0.714136*cr) - p.Mean) * inv
			dst[cB*plane+o] = (clamp255(Y+1.772*cb) - p.Mean) * inv
		}
	}
}

func l2normalize(v []float32) {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	if s == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(s))
	for i := range v {
		v[i] *= inv
	}
}

// synthJPEG builds a photo-like test frame (smooth structure + sensor noise)
// so that JPEG decode cost is comparable to a real webcam capture without
// needing anyone's photo.
func synthJPEG(w, h, quality int, noise float64) []byte {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	rng := rand.New(rand.NewSource(1))
	const tau = 2 * math.Pi
	for y := 0; y < h; y++ {
		fy := float64(y) / float64(h)
		for x := 0; x < w; x++ {
			fx := float64(x) / float64(w)
			base := 128 + 55*math.Sin(tau*(1.3*fx+0.7*fy)) + 35*math.Sin(tau*(4.1*fx-2.3*fy)) + 14*math.Sin(tau*(17*fx+11*fy))
			n := rng.NormFloat64() * noise
			o := img.PixOffset(x, y)
			img.Pix[o] = uint8(clamp255(float32(base + 18*math.Sin(tau*2*fy) + n)))
			img.Pix[o+1] = uint8(clamp255(float32(base + n)))
			img.Pix[o+2] = uint8(clamp255(float32(base - 22*math.Sin(tau*3*fx) + n)))
			img.Pix[o+3] = 255
		}
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: quality}); err != nil {
		panic(err)
	}
	return buf.Bytes()
}
