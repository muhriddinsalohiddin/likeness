package vision

// maxReduction bounds the box filter; beyond this the picture is far larger
// than anything a face model needs and the remaining aliasing is harmless.
const maxReduction = 32

// level returns a copy of m reduced k times by box averaging, together with
// the factor actually used. Bilinear sampling reads only four source pixels,
// so shrinking by more than 2x without this pre-filter discards most of the
// picture and aliases fine detail. Copies are built once and cached.
func (m *Image) level(k int) (*Image, int) {
	k = clampReduction(k, m.w, m.h)
	if k <= 1 {
		return m, 1
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if l, ok := m.levels[k]; ok {
		return l, k
	}
	if m.levels == nil {
		m.levels = make(map[int]*Image)
	}
	l := m.reduce(k, 0)
	m.levels[k] = l
	return l, k
}

func clampReduction(k, w, h int) int {
	return max(min(k, maxReduction, w, h), 1)
}

// view returns the part of m inside [x0,x1) x [y0,y1) without copying, and
// the origin it ended up with: the rectangle is clipped to the picture and
// its origin snapped to the chroma grid so the view is a valid picture of the
// same layout. ok is false when the rectangle misses the picture.
func (m *Image) view(x0, y0, x1, y1 int) (v *Image, ox, oy int, ok bool) {
	x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, m.w), min(y1, m.h)
	if x0 >= x1 || y0 >= y1 {
		return nil, 0, 0, false
	}
	v = &Image{kind: m.kind, orient: 1, hdiv: m.hdiv, vdiv: m.vdiv, yStride: m.yStride, cStride: m.cStride, stride: m.stride}
	switch m.kind {
	case kindYCbCr:
		x0, y0 = x0-x0%m.hdiv, y0-y0%m.vdiv
		c := (y0/m.vdiv)*m.cStride + x0/m.hdiv
		v.y, v.cb, v.cr = m.y[y0*m.yStride+x0:], m.cb[c:], m.cr[c:]
	case kindGray:
		v.y = m.y[y0*m.yStride+x0:]
	case kindRGBA:
		v.pix = m.pix[y0*m.stride+4*x0:]
	}
	v.w, v.h = x1-x0, y1-y0
	return v, x0, y0, true
}

// reduce box-averages m by k. Partial blocks at the right and bottom edges
// are kept, averaged over the pixels that exist.
func (m *Image) reduce(k, workers int) *Image {
	l := &Image{kind: m.kind, w: (m.w + k - 1) / k, h: (m.h + k - 1) / k, orient: m.orient, hdiv: m.hdiv, vdiv: m.vdiv}
	if workers <= 0 {
		workers = Options{}.workers(m.w * m.h)
	}
	switch m.kind {
	case kindYCbCr:
		cw, ch := m.chromaSize()
		lcw, lch := l.chromaSize()
		l.yStride, l.cStride = l.w, lcw
		l.y = make([]uint8, l.w*l.h)
		l.cb = make([]uint8, lcw*lch)
		l.cr = make([]uint8, lcw*lch)
		boxDown(l.y, l.w, l.h, l.yStride, 1, 0, m.y, m.w, m.h, m.yStride, 1, 0, k, workers)
		boxDown(l.cb, lcw, lch, l.cStride, 1, 0, m.cb, cw, ch, m.cStride, 1, 0, k, workers)
		boxDown(l.cr, lcw, lch, l.cStride, 1, 0, m.cr, cw, ch, m.cStride, 1, 0, k, workers)
	case kindGray:
		l.yStride = l.w
		l.y = make([]uint8, l.w*l.h)
		boxDown(l.y, l.w, l.h, l.yStride, 1, 0, m.y, m.w, m.h, m.yStride, 1, 0, k, workers)
	case kindRGBA:
		l.stride = 4 * l.w
		l.pix = make([]uint8, 4*l.w*l.h)
		for c := 0; c < 3; c++ {
			boxDown(l.pix, l.w, l.h, l.stride, 4, c, m.pix, m.w, m.h, m.stride, 4, c, k, workers)
		}
	}
	return l
}

// boxDown writes the mean of each k x k source block to dst. Blocks reaching
// past the source average the samples that exist; a block entirely past it
// (possible for chroma planes of odd-sized pictures) repeats the edge sample.
func boxDown(dst []uint8, dw, dh, dstStride, dstStep, dstOff int, src []uint8, sw, sh, srcStride, srcStep, srcOff, k, workers int) {
	splitRows(dh, workers, func(from, to int) {
		for j := from; j < to; j++ {
			y0 := min(j*k, sh-1)
			y1 := min(j*k+k, sh)
			for i := 0; i < dw; i++ {
				x0 := min(i*k, sw-1)
				x1 := min(i*k+k, sw)
				sum := 0
				for y := y0; y < y1; y++ {
					row := src[y*srcStride+srcOff:]
					for x := x0; x < x1; x++ {
						sum += int(row[x*srcStep])
					}
				}
				n := (y1 - y0) * (x1 - x0)
				dst[j*dstStride+i*dstStep+dstOff] = uint8((sum + n/2) / n)
			}
		}
	})
}
