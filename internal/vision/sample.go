package vision

import (
	"math"
	"runtime"
	"sync"

	"github.com/muhriddinsalohiddin/likeness/internal/align"
)

// Norm describes the pixel normalisation and channel order a model expects:
// every channel becomes (value - Mean) / Std with values on the 0..255 scale.
type Norm struct {
	Mean, Std [3]float32 // in R, G, B order
	BGR       bool       // emit planes as B, G, R instead of R, G, B
}

// Uniform is a Norm with the same mean and deviation for every channel.
func Uniform(mean, std float32, bgr bool) Norm {
	return Norm{Mean: [3]float32{mean, mean, mean}, Std: [3]float32{std, std, std}, BGR: bgr}
}

// RawRGB leaves pixel values as they are: R, G and B planes on the 0..255
// scale. A crop sampled this way can be inspected and later handed to any
// model with Norm.Apply.
var RawRGB = Uniform(0, 1, false)

// Apply converts a RawRGB tensor of three planes of area values each into
// the layout n describes. The result is bit-for-bit what sampling with n
// directly would have produced.
func (n Norm) Apply(dst, src []float32, area int) {
	if len(src) < 3*area || len(dst) < 3*area {
		panic("vision: tensor buffer does not match its dimensions")
	}
	nz := n.prepare()
	for c := 0; c < 3; c++ {
		in := src[c*area : (c+1)*area]
		out := dst[nz.plane[c]*area : (nz.plane[c]+1)*area]
		mean, inv := nz.mean[c], nz.inv[c]
		for i, v := range in {
			out[i] = (v - mean) * inv
		}
	}
}

// Options tune sampling. The zero value is what production code wants.
type Options struct {
	// NoAntiAlias disables the box pre-filter used when shrinking by 2x or
	// more. Reference pipelines sample without it, so parity tests set this.
	NoAntiAlias bool
	// Workers caps the goroutines used for large outputs; 0 picks a default.
	Workers int
}

type normalizer struct {
	mean, inv [3]float32
	plane     [3]int // destination plane of R, G, B
}

func (n Norm) prepare() normalizer {
	z := normalizer{mean: n.Mean, plane: [3]int{0, 1, 2}}
	if n.BGR {
		z.plane = [3]int{2, 1, 0}
	}
	for i, s := range n.Std {
		if s == 0 {
			s = 1
		}
		z.inv[i] = 1 / s
	}
	return z
}

func (o Options) workers(pixels int) int {
	if pixels < 128*128 {
		return 1
	}
	w := o.Workers
	if w <= 0 {
		w = min(4, runtime.GOMAXPROCS(0))
	}
	return max(w, 1)
}

// Letterbox scales the picture to fit a tw x th tensor, anchored top-left
// with black padding, the way face detectors are fed. dst receives 3*tw*th
// values in CHW order. The returned transform maps tensor coordinates back
// to picture coordinates.
func (m *Image) Letterbox(dst []float32, tw, th int, n Norm, o Options) align.Affine {
	if tw <= 0 || th <= 0 || len(dst) < 3*tw*th {
		panic("vision: tensor buffer does not match its dimensions")
	}
	W, H := m.Size()
	s := math.Min(float64(tw)/float64(W), float64(th)/float64(H))
	cw := min(max(int(math.Round(float64(W)*s)), 1), tw)
	ch := min(max(int(math.Round(float64(H)*s)), 1), th)
	invX, invY := float64(W)/float64(cw), float64(H)/float64(ch)

	nz := n.prepare()
	fillPadding(dst, tw, th, cw, ch, nz)

	k := 1
	if !o.NoAntiAlias {
		k = int(math.Min(invX, invY))
	}
	src, k := m.level(k)
	workers := o.workers(cw * ch)

	toPicture := align.Affine{
		A: float32(invX), C: float32(0.5*invX - 0.5),
		E: float32(invY), F: float32(0.5*invY - 0.5),
	}
	if src.kind == kindYCbCr && m.orient == 1 {
		src.letterboxYCbCr(dst, tw, th, cw, ch, invX, invY, k, nz, workers)
	} else {
		src.sample(dst, tw, th, cw, ch, levelFromStored(k).Mul(m.storedFromUpright()).Mul(toPicture), true, nz, workers)
	}
	return toPicture
}

// Warp fills a size x size CHW tensor with the region that toCrop maps onto
// it, as cv2.warpAffine does: bilinear sampling, black outside the picture.
func (m *Image) Warp(dst []float32, size int, toCrop align.Affine, n Norm, o Options) error {
	if size <= 0 || len(dst) < 3*size*size {
		panic("vision: tensor buffer does not match its dimensions")
	}
	fromCrop, err := toCrop.Invert()
	if err != nil {
		return err
	}
	nz := n.prepare()
	workers := o.workers(size * size)
	toStored := m.storedFromUpright().Mul(fromCrop)

	k := 1
	if !o.NoAntiAlias {
		k = clampReduction(int(fromCrop.Scale()), m.w, m.h)
	}
	if k == 1 {
		m.sample(dst, size, size, size, size, toStored, false, nz, workers)
		return nil
	}

	// A large face is pre-filtered like a letterboxed frame, but only over the
	// pixels the crop actually looks at.
	x0, y0, x1, y1 := footprint(toStored, size)
	v, ox, oy, ok := m.view(x0-k-2, y0-k-2, x1+k+2, y1+k+2)
	if !ok {
		fillPadding(dst, size, size, 0, 0, nz) // the crop misses the picture entirely
		return nil
	}
	toLevel := levelFromStored(k).Mul(align.Affine{A: 1, C: -float32(ox), E: 1, F: -float32(oy)}).Mul(toStored)
	v.reduce(k, 1).sample(dst, size, size, size, size, toLevel, false, nz, workers)
	return nil
}

// footprint is the bounding box, in stored pixels, of a size x size crop.
func footprint(toStored align.Affine, size int) (x0, y0, x1, y1 int) {
	const far = 1 << 24
	minX, minY, maxX, maxY := math.Inf(1), math.Inf(1), math.Inf(-1), math.Inf(-1)
	for _, c := range [4]align.Point{{X: -1, Y: -1}, {X: float32(size), Y: -1}, {X: -1, Y: float32(size)}, {X: float32(size), Y: float32(size)}} {
		p := toStored.Apply(c)
		x, y := float64(p.X), float64(p.Y)
		if math.IsNaN(x) || math.IsNaN(y) {
			return 0, 0, 0, 0
		}
		minX, maxX = math.Min(minX, x), math.Max(maxX, x)
		minY, maxY = math.Min(minY, y), math.Max(maxY, y)
	}
	clamp := func(v float64) int { return int(math.Min(math.Max(v, -far), far)) }
	return clamp(math.Floor(minX)), clamp(math.Floor(minY)), clamp(math.Ceil(maxX)) + 1, clamp(math.Ceil(maxY)) + 1
}

// levelFromStored maps full-resolution stored coordinates to those of a copy
// reduced by k: pixel i of the copy covers pixels [i*k, i*k+k-1].
func levelFromStored(k int) align.Affine {
	if k <= 1 {
		return align.Identity
	}
	s := 1 / float32(k)
	return align.Affine{A: s, C: 0.5*s - 0.5, E: s, F: 0.5*s - 0.5}
}

func fillPadding(dst []float32, tw, th, cw, ch int, nz normalizer) {
	area := tw * th
	for c := 0; c < 3; c++ {
		pad := (0 - nz.mean[c]) * nz.inv[c]
		p := dst[nz.plane[c]*area : nz.plane[c]*area+area]
		for v := 0; v < th; v++ {
			row := p[v*tw : (v+1)*tw]
			from := cw
			if v >= ch {
				from = 0
			}
			for u := from; u < tw; u++ {
				row[u] = pad
			}
		}
	}
}

func splitRows(n, workers int, fn func(from, to int)) {
	if workers <= 1 || n < 2*workers {
		fn(0, n)
		return
	}
	var wg sync.WaitGroup
	step := (n + workers - 1) / workers
	for from := 0; from < n; from += step {
		wg.Add(1)
		go func(from, to int) {
			defer wg.Done()
			fn(from, to)
		}(from, min(from+step, n))
	}
	wg.Wait()
}

// raster is one 8-bit channel: a plane of a planar image or every step-th
// byte of an interleaved one.
type raster struct {
	pix       []uint8
	stride    int
	w, h      int
	off, step int
	outside   float32 // value beyond the border when not replicating
}

func (r *raster) get(x, y int) float32 {
	if x < 0 || y < 0 || x >= r.w || y >= r.h {
		return r.outside
	}
	return float32(r.pix[y*r.stride+x*r.step+r.off])
}

func ifloor(f float32) int {
	i := int(f)
	if float32(i) > f {
		i--
	}
	return i
}

func (r *raster) bilinear(x, y float32, replicate bool) float32 {
	if replicate {
		x = min(max(x, 0), float32(r.w-1))
		y = min(max(y, 0), float32(r.h-1))
	} else if !(x > -1 && y > -1 && x < float32(r.w) && y < float32(r.h)) {
		return r.outside
	}
	x0, y0 := ifloor(x), ifloor(y)
	fx, fy := x-float32(x0), y-float32(y0)
	var a, b, c, d float32
	if x0 >= 0 && y0 >= 0 && x0+1 < r.w && y0+1 < r.h {
		i := y0*r.stride + x0*r.step + r.off
		a, b = float32(r.pix[i]), float32(r.pix[i+r.step])
		c, d = float32(r.pix[i+r.stride]), float32(r.pix[i+r.stride+r.step])
	} else {
		a, b = r.get(x0, y0), r.get(x0+1, y0)
		c, d = r.get(x0, y0+1), r.get(x0+1, y0+1)
	}
	top := a + (b-a)*fx
	return top + (c+(d-c)*fx-top)*fy
}

func clamp255(v float32) float32 {
	return min(max(v, 0), 255)
}

// JFIF full-range YCbCr to RGB.
func ycbcrToRGB(y, cb, cr float32) (r, g, b float32) {
	cb -= 128
	cr -= 128
	return clamp255(y + 1.402*cr), clamp255(y - 0.344136*cb - 0.714136*cr), clamp255(y + 1.772*cb)
}

// sample fills the cw x ch top-left area of a tw x th CHW tensor. toLuma maps
// tensor coordinates to m's stored pixel grid. It handles every pixel layout
// and orientation; upright YCbCr letterboxing has a faster dedicated path.
func (m *Image) sample(dst []float32, tw, th, cw, ch int, toLuma align.Affine, replicate bool, nz normalizer, workers int) {
	area := tw * th
	pr, pg, pb := dst[nz.plane[0]*area:], dst[nz.plane[1]*area:], dst[nz.plane[2]*area:]
	put := func(i int, r, g, b float32) {
		pr[i] = (r - nz.mean[0]) * nz.inv[0]
		pg[i] = (g - nz.mean[1]) * nz.inv[1]
		pb[i] = (b - nz.mean[2]) * nz.inv[2]
	}
	coords := func(u, v int) (float32, float32) {
		return toLuma.A*float32(u) + toLuma.B*float32(v) + toLuma.C,
			toLuma.D*float32(u) + toLuma.E*float32(v) + toLuma.F
	}

	switch m.kind {
	case kindYCbCr:
		cwid, chei := m.chromaSize()
		ys := raster{pix: m.y, stride: m.yStride, w: m.w, h: m.h, step: 1}
		cbs := raster{pix: m.cb, stride: m.cStride, w: cwid, h: chei, step: 1, outside: 128}
		crs := raster{pix: m.cr, stride: m.cStride, w: cwid, h: chei, step: 1, outside: 128}
		hx, hy := 1/float32(m.hdiv), 1/float32(m.vdiv)
		splitRows(ch, workers, func(from, to int) {
			for v := from; v < to; v++ {
				for u := 0; u < cw; u++ {
					x, y := coords(u, v)
					cx, cy := (x+0.5)*hx-0.5, (y+0.5)*hy-0.5
					r, g, b := ycbcrToRGB(ys.bilinear(x, y, replicate), cbs.bilinear(cx, cy, replicate), crs.bilinear(cx, cy, replicate))
					put(v*tw+u, r, g, b)
				}
			}
		})
	case kindGray:
		ys := raster{pix: m.y, stride: m.yStride, w: m.w, h: m.h, step: 1}
		splitRows(ch, workers, func(from, to int) {
			for v := from; v < to; v++ {
				for u := 0; u < cw; u++ {
					x, y := coords(u, v)
					l := ys.bilinear(x, y, replicate)
					put(v*tw+u, l, l, l)
				}
			}
		})
	case kindRGBA:
		rs := raster{pix: m.pix, stride: m.stride, w: m.w, h: m.h, step: 4}
		gs, bs := rs, rs
		gs.off, bs.off = 1, 2
		splitRows(ch, workers, func(from, to int) {
			for v := from; v < to; v++ {
				for u := 0; u < cw; u++ {
					x, y := coords(u, v)
					put(v*tw+u, rs.bilinear(x, y, replicate), gs.bilinear(x, y, replicate), bs.bilinear(x, y, replicate))
				}
			}
		})
	}
}

// axis holds, for every output sample along one dimension, the two source
// samples it blends and the weight of the second.
type axis struct {
	i0, i1 []int32
	w      []float32
}

// newAxis maps n output samples onto a plane of length planeLen whose samples
// are div source pixels apart; inv is source pixels per output pixel.
func newAxis(n, planeLen int, inv float64, div int) axis {
	a := axis{make([]int32, n), make([]int32, n), make([]float32, n)}
	last := float64(planeLen - 1)
	for d := 0; d < n; d++ {
		p := ((float64(d)+0.5)*inv)/float64(div) - 0.5
		p = math.Min(math.Max(p, 0), last)
		i0 := int(p)
		a.i0[d], a.i1[d], a.w[d] = int32(i0), int32(min(i0+1, planeLen-1)), float32(p-float64(i0))
	}
	return a
}

// letterboxYCbCr is the hot path: an upright JPEG scaled into a detector
// tensor. The scale is axis-aligned, so source indices and weights are
// computed once per row and column instead of once per pixel.
func (m *Image) letterboxYCbCr(dst []float32, tw, th, cw, ch int, invX, invY float64, k int, nz normalizer, workers int) {
	cwid, chei := m.chromaSize()
	xl, xc := newAxis(cw, m.w, invX, k), newAxis(cw, cwid, invX, k*m.hdiv)
	yl, yc := newAxis(ch, m.h, invY, k), newAxis(ch, chei, invY, k*m.vdiv)
	area := tw * th
	pr, pg, pb := dst[nz.plane[0]*area:], dst[nz.plane[1]*area:], dst[nz.plane[2]*area:]

	splitRows(ch, workers, func(from, to int) {
		for v := from; v < to; v++ {
			ya, yb, wy := m.y[int(yl.i0[v])*m.yStride:], m.y[int(yl.i1[v])*m.yStride:], yl.w[v]
			cba, cbb := m.cb[int(yc.i0[v])*m.cStride:], m.cb[int(yc.i1[v])*m.cStride:]
			cra, crb := m.cr[int(yc.i0[v])*m.cStride:], m.cr[int(yc.i1[v])*m.cStride:]
			wc := yc.w[v]
			o := v * tw
			for u := 0; u < cw; u++ {
				x0, x1, wx := xl.i0[u], xl.i1[u], xl.w[u]
				t := float32(ya[x0]) + (float32(ya[x1])-float32(ya[x0]))*wx
				b := float32(yb[x0]) + (float32(yb[x1])-float32(yb[x0]))*wx
				lum := t + (b-t)*wy

				c0, c1, wu := xc.i0[u], xc.i1[u], xc.w[u]
				t = float32(cba[c0]) + (float32(cba[c1])-float32(cba[c0]))*wu
				b = float32(cbb[c0]) + (float32(cbb[c1])-float32(cbb[c0]))*wu
				cb := t + (b-t)*wc
				t = float32(cra[c0]) + (float32(cra[c1])-float32(cra[c0]))*wu
				b = float32(crb[c0]) + (float32(crb[c1])-float32(crb[c0]))*wu
				cr := t + (b-t)*wc

				r, g, bl := ycbcrToRGB(lum, cb, cr)
				pr[o+u] = (r - nz.mean[0]) * nz.inv[0]
				pg[o+u] = (g - nz.mean[1]) * nz.inv[1]
				pb[o+u] = (bl - nz.mean[2]) * nz.inv[2]
			}
		}
	})
}
