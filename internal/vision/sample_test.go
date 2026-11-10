package vision

import (
	"image"
	"image/color"
	"math"
	"math/rand"
	"testing"

	"github.com/muhriddinsalohiddin/likeness/internal/align"
)

func TestSampleMatchesNaiveReference(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	images := map[string]*Image{
		"ycbcr420": FromImage(smoothYCbCr(rng, 97, 61, image.YCbCrSubsampleRatio420), 1),
		"ycbcr422": FromImage(smoothYCbCr(rng, 80, 45, image.YCbCrSubsampleRatio422), 1),
		"ycbcr444": FromImage(smoothYCbCr(rng, 64, 64, image.YCbCrSubsampleRatio444), 1),
		"ycbcr440": FromImage(smoothYCbCr(rng, 51, 70, image.YCbCrSubsampleRatio440), 1),
		"gray":     FromImage(smoothGray(rng, 73, 58), 1),
		"rgba":     FromImage(smoothRGBA(rng, 66, 49), 1),
	}
	norm := Norm{Mean: [3]float32{123.7, 116.3, 103.5}, Std: [3]float32{58.4, 57.1, 57.4}, BGR: true}
	const size = 40
	for name, m := range images {
		for trial := 0; trial < 25; trial++ {
			angle, scale := rng.Float64()*2*math.Pi, 0.3+rng.Float64()*3
			c, s := float32(scale*math.Cos(angle)), float32(scale*math.Sin(angle))
			// centred on the picture, so part of the crop usually falls outside
			to := align.Affine{A: c, B: -s, C: float32(m.w)/2 - 20*c + 20*s, D: s, E: c, F: float32(m.h)/2 - 20*s - 20*c}
			for _, replicate := range []bool{false, true} {
				got := make([]float32, 3*size*size)
				m.sample(got, size, size, size, size, to, replicate, norm.prepare(), 3)
				want := naiveSample(m, size, size, size, size, to, replicate, norm)
				if d, at := maxDiff(got, want); d > 2e-3 {
					t.Fatalf("%s trial %d replicate=%v: differs by %g at %d", name, trial, replicate, d, at)
				}
			}
		}
	}
}

func TestWarpIdentityReproducesPixels(t *testing.T) {
	src := uniqueRGBA(24, 24)
	m := FromImage(src, 1)
	got := make([]float32, 3*24*24)
	if err := m.Warp(got, 24, align.Identity, raw, Options{}); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < 24; y++ {
		for x := 0; x < 24; x++ {
			c := src.RGBAAt(x, y)
			i := y*24 + x
			if got[i] != float32(c.R) || got[576+i] != float32(c.G) || got[1152+i] != float32(c.B) {
				t.Fatalf("pixel (%d,%d): got %v %v %v, want %v", x, y, got[i], got[576+i], got[1152+i], c)
			}
		}
	}
}

func TestWarpTreatsOutsideAsBlack(t *testing.T) {
	white := image.NewGray(image.Rect(0, 0, 32, 32))
	for i := range white.Pix {
		white.Pix[i] = 200
	}
	m := FromImage(white, 1)
	norm := Uniform(127.5, 127.5, false)
	got := make([]float32, 3*16*16)

	// crop pixel u samples picture x = u - 8.5: u=8 lies half a pixel outside
	shifted := align.Affine{A: 1, C: 8.5, E: 1}
	if err := m.Warp(got, 16, shifted, norm, Options{}); err != nil {
		t.Fatal(err)
	}
	black, half, full := float32(-1), float32((100-127.5)/127.5), float32((200-127.5)/127.5)
	for u, want := range map[int]float32{0: black, 7: black, 8: half, 9: full, 15: full} {
		if g := got[5*16+u]; math.Abs(float64(g-want)) > 1e-5 {
			t.Errorf("column %d: got %v, want %v", u, g, want)
		}
	}
}

func TestWarpRejectsSingularTransform(t *testing.T) {
	m := FromImage(smoothGray(rand.New(rand.NewSource(1)), 32, 32), 1)
	if err := m.Warp(make([]float32, 3*8*8), 8, align.Affine{}, raw, Options{}); err == nil {
		t.Fatal("a singular transform must be rejected")
	}
}

func TestLetterboxGeometry(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 200, 100))
	for i := 3; i < len(src.Pix); i += 4 {
		src.Pix[i] = 255
	}
	// a bright block centred on picture coordinates (149.5, 49.5)
	for y := 40; y < 60; y++ {
		for x := 140; x < 160; x++ {
			src.SetRGBA(x, y, color.RGBA{255, 255, 255, 255})
		}
	}
	m := FromImage(src, 1)
	norm := Uniform(127.5, 128, false)
	const side = 64
	got := make([]float32, 3*side*side)
	for i := range got {
		got[i] = 99 // stale data from a previous request
	}
	toPicture := m.Letterbox(got, side, side, norm, Options{})

	pad := float32(-127.5 / 128)
	for v := 0; v < side; v++ {
		for u := 0; u < side; u++ {
			if v >= 32 && got[v*side+u] != pad {
				t.Fatalf("(%d,%d) below the picture holds %v, want padding %v", u, v, got[v*side+u], pad)
			}
		}
	}
	// centroid of the bright block, mapped back to the picture
	var sx, sy, n float64
	for v := 0; v < 32; v++ {
		for u := 0; u < side; u++ {
			if w := float64(got[v*side+u] - pad); w > 0.5 {
				sx, sy, n = sx+float64(u)*w, sy+float64(v)*w, n+w
			}
		}
	}
	p := toPicture.Apply(align.Point{X: float32(sx / n), Y: float32(sy / n)})
	if math.Abs(float64(p.X)-149.5) > 0.6 || math.Abs(float64(p.Y)-49.5) > 0.6 {
		t.Fatalf("block found at %v, want about (149.5, 49.5)", p)
	}
}

func TestLetterboxFastPathMatchesGeneralPath(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, ratio := range []image.YCbCrSubsampleRatio{
		image.YCbCrSubsampleRatio420, image.YCbCrSubsampleRatio422,
		image.YCbCrSubsampleRatio444, image.YCbCrSubsampleRatio440,
	} {
		for _, dim := range [][2]int{{640, 360}, {331, 517}, {90, 70}} {
			for _, noAA := range []bool{true, false} {
				m := FromImage(smoothYCbCr(rng, dim[0], dim[1], ratio), 1)
				const tw, th = 96, 80
				fast := make([]float32, 3*tw*th)
				toPicture := m.Letterbox(fast, tw, th, raw, Options{NoAntiAlias: noAA})

				k := 1
				if !noAA {
					k = int(math.Min(float64(toPicture.A), float64(toPicture.E)))
				}
				level, k := m.level(k)
				cw := int(math.Round(float64(dim[0]) / float64(toPicture.A)))
				ch := int(math.Round(float64(dim[1]) / float64(toPicture.E)))
				general := make([]float32, 3*tw*th)
				fillPadding(general, tw, th, cw, ch, raw.prepare())
				level.sample(general, tw, th, cw, ch, levelFromStored(k).Mul(toPicture), true, raw.prepare(), 1)

				if d, at := maxDiff(fast, general); d > 0.05 {
					t.Fatalf("%v %v noAA=%v: paths differ by %g at %d", ratio, dim, noAA, d, at)
				}
			}
		}
	}
}

// TestOrientationIsUndone stores a picture the way a camera would for each
// EXIF orientation and expects the upright picture back. The table is written
// from the wording of the EXIF specification ("the 0th row is the visual
// right-hand side, the 0th column is the visual top" for 6, and so on), in the
// stored-to-upright direction, the opposite of what the implementation uses.
func TestOrientationIsUndone(t *testing.T) {
	const w, h = 7, 5
	upright := uniqueRGBA(w, h)
	toUpright := map[int]func(sx, sy int) (int, int){
		1: func(sx, sy int) (int, int) { return sx, sy },
		2: func(sx, sy int) (int, int) { return w - 1 - sx, sy },
		3: func(sx, sy int) (int, int) { return w - 1 - sx, h - 1 - sy },
		4: func(sx, sy int) (int, int) { return sx, h - 1 - sy },
		5: func(sx, sy int) (int, int) { return sy, sx },
		6: func(sx, sy int) (int, int) { return w - 1 - sy, sx },
		7: func(sx, sy int) (int, int) { return w - 1 - sy, h - 1 - sx },
		8: func(sx, sy int) (int, int) { return sy, h - 1 - sx },
	}
	for o, f := range toUpright {
		sw, sh := w, h
		if o >= 5 {
			sw, sh = h, w
		}
		stored := image.NewRGBA(image.Rect(0, 0, sw, sh))
		for sy := 0; sy < sh; sy++ {
			for sx := 0; sx < sw; sx++ {
				x, y := f(sx, sy)
				stored.SetRGBA(sx, sy, upright.RGBAAt(x, y))
			}
		}
		m := FromImage(stored, o)
		if gw, gh := m.Size(); gw != w || gh != h {
			t.Fatalf("orientation %d: size %dx%d, want %dx%d", o, gw, gh, w, h)
		}
		got := make([]float32, 3*w*h)
		m.Letterbox(got, w, h, raw, Options{NoAntiAlias: true})
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				c := upright.RGBAAt(x, y)
				i := y*w + x
				if got[i] != float32(c.R) || got[w*h+i] != float32(c.G) || got[2*w*h+i] != float32(c.B) {
					t.Fatalf("orientation %d, pixel (%d,%d): got %v/%v/%v, want %v", o, x, y, got[i], got[w*h+i], got[2*w*h+i], c)
				}
			}
		}
	}
	if m := FromImage(upright, 42); m.orient != 1 {
		t.Fatalf("an invalid orientation must fall back to upright, got %d", m.orient)
	}
}

func TestAntiAliasingAveragesFineDetail(t *testing.T) {
	board := image.NewGray(image.Rect(0, 0, 256, 256))
	for y := 0; y < 256; y++ {
		for x := 0; x < 256; x++ {
			if (x+y)%2 == 0 {
				board.Pix[y*256+x] = 255
			}
		}
	}
	m := FromImage(board, 1)
	spread := func(o Options) float64 {
		got := make([]float32, 3*32*32)
		m.Letterbox(got, 32, 32, raw, o)
		var worst float64
		for _, v := range got[:32*32] {
			worst = math.Max(worst, math.Abs(float64(v)-127.5))
		}
		return worst
	}
	if s := spread(Options{}); s > 1 {
		t.Errorf("with the pre-filter a checkerboard should come out flat, deviates by %v", s)
	}
	// Guards the test itself: 4-pixel stripes shrunk 8x. Without the filter the
	// taps of every output row land on rows 3 and 4 of a stripe period, which
	// are both dark here, so the picture comes out black instead of grey.
	for i := range board.Pix {
		if (i/256+2)%8 < 4 {
			board.Pix[i] = 255
		} else {
			board.Pix[i] = 0
		}
	}
	m = FromImage(board, 1)
	if s := spread(Options{NoAntiAlias: true}); s < 60 {
		t.Errorf("expected visible aliasing without the pre-filter, deviation is only %v", s)
	}
	if s := spread(Options{}); s > 1 {
		t.Errorf("4-pixel stripes should average out with the pre-filter, deviates by %v", s)
	}
}

func TestReducedCopiesAreCachedAndKeepTheMean(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, m := range []*Image{
		FromImage(smoothYCbCr(rng, 120, 90, image.YCbCrSubsampleRatio420), 1),
		FromImage(smoothGray(rng, 120, 90), 1),
		FromImage(smoothRGBA(rng, 120, 90), 1),
	} {
		a, k := m.level(3)
		b, _ := m.level(3)
		if a != b || k != 3 {
			t.Fatalf("level(3) is not cached (k=%d)", k)
		}
		if a.w != 40 || a.h != 30 {
			t.Fatalf("reduced size %dx%d, want 40x30", a.w, a.h)
		}
		full, reduced := make([]float32, 3*120*90), make([]float32, 3*40*30)
		m.sample(full, 120, 90, 120, 90, align.Identity, true, raw.prepare(), 1)
		a.sample(reduced, 40, 30, 40, 30, align.Identity, true, raw.prepare(), 1)
		if d := math.Abs(mean(full) - mean(reduced)); d > 0.6 {
			t.Fatalf("kind %d: mean moved by %v after reduction", m.kind, d)
		}
	}
	tiny := FromImage(smoothGray(rng, 20, 10), 1)
	if _, k := tiny.level(1000); k != 10 {
		t.Fatalf("reduction must be capped by the picture size, got k=%d", k)
	}
}

func mean(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x)
	}
	return s / float64(len(v))
}

func TestNormalisationAndChannelOrder(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 16, 16))
	for i := 0; i < len(src.Pix); i += 4 {
		src.Pix[i], src.Pix[i+1], src.Pix[i+2], src.Pix[i+3] = 10, 20, 30, 255
	}
	m := FromImage(src, 1)
	n := Norm{Mean: [3]float32{1, 2, 3}, Std: [3]float32{3, 0, 9}, BGR: true} // zero deviation means "leave as is"
	got := make([]float32, 3*16*16)
	m.Letterbox(got, 16, 16, n, Options{})
	if b, g, r := got[0], got[256], got[512]; b != 3 || g != 18 || r != 3 {
		t.Fatalf("got B=%v G=%v R=%v, want 3 18 3", b, g, r)
	}
}

// A face much larger than the crop goes through the pre-filtered path, which
// samples a reduced view of part of the picture. The geometry must not move.
func TestWarpWithPreFilterKeepsGeometry(t *testing.T) {
	const sw, sh = 640, 480
	block := func(set func(x, y int)) {
		for y := 294; y < 306; y++ { // centred on stored (411.5, 299.5)
			for x := 406; x < 418; x++ {
				set(x, y)
			}
		}
	}
	ycc := image.NewYCbCr(image.Rect(0, 0, sw, sh), image.YCbCrSubsampleRatio420)
	for i := range ycc.Cb {
		ycc.Cb[i], ycc.Cr[i] = 128, 128
	}
	block(func(x, y int) { ycc.Y[ycc.YOffset(x, y)] = 255 })
	gray := image.NewGray(image.Rect(0, 0, sw, sh))
	block(func(x, y int) { gray.Pix[gray.PixOffset(x, y)] = 255 })
	rgba := image.NewRGBA(image.Rect(0, 0, sw, sh))
	block(func(x, y int) { rgba.SetRGBA(x, y, color.RGBA{255, 255, 255, 255}) })

	for name, src := range map[string]image.Image{"ycbcr": ycc, "gray": gray, "rgba": rgba} {
		for _, orientation := range []int{1, 6} {
			m := FromImage(src, orientation)
			centre := align.Point{X: 411.5, Y: 299.5}
			if orientation == 6 { // upright (x, y) is stored (y, sh-1-x)
				centre = align.Point{X: sh - 1 - 299.5, Y: 411.5}
			}
			// shrink 4.3x, turn 25 degrees, put the block at crop position (28, 30)
			c, s := float32(math.Cos(0.436)/4.3), float32(math.Sin(0.436)/4.3)
			toCrop := align.Affine{A: c, B: -s, D: s, E: c}
			p := toCrop.Apply(centre)
			toCrop.C, toCrop.F = 28-p.X, 30-p.Y

			for _, o := range []Options{{}, {NoAntiAlias: true}} {
				got := make([]float32, 3*56*56)
				if err := m.Warp(got, 56, toCrop, raw, o); err != nil {
					t.Fatal(err)
				}
				var sx, sy, n float64
				for i, v := range got[:56*56] {
					sx, sy, n = sx+float64(i%56)*float64(v), sy+float64(i/56)*float64(v), n+float64(v)
				}
				if n == 0 || math.Abs(sx/n-28) > 0.3 || math.Abs(sy/n-30) > 0.3 {
					t.Fatalf("%s orientation %d %+v: block at (%.2f, %.2f), want (28, 30)", name, orientation, o, sx/n, sy/n)
				}
			}
		}
	}
}

func TestWarpPreFilterAgreesOnSmoothContent(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	for _, m := range []*Image{
		FromImage(smoothYCbCr(rng, 801, 603, image.YCbCrSubsampleRatio420), 1),
		FromImage(smoothRGBA(rng, 801, 603), 8),
	} {
		// a crop that hangs over the top-left corner of the picture
		toCrop := align.Affine{A: 0.2, B: 0.05, C: 20, D: -0.05, E: 0.2, F: 35}
		with, without := make([]float32, 3*112*112), make([]float32, 3*112*112)
		if err := m.Warp(with, 112, toCrop, raw, Options{}); err != nil {
			t.Fatal(err)
		}
		if err := m.Warp(without, 112, toCrop, raw, Options{NoAntiAlias: true}); err != nil {
			t.Fatal(err)
		}
		var sum float64
		for i := range with {
			sum += math.Abs(float64(with[i] - without[i]))
		}
		// the test pictures carry pixel noise, which chroma-to-RGB conversion
		// amplifies and the filter removes; anything grossly wrong is far above this
		if avg := sum / float64(len(with)); avg > 4 {
			t.Fatalf("kind %d: filtered and unfiltered crops differ by %.2f levels on average", m.kind, avg)
		}
		if with[0] != 0 || without[0] != 0 {
			t.Fatalf("kind %d: the corner outside the picture must be black, got %v and %v", m.kind, with[0], without[0])
		}
	}
}

func TestWarpFarOutsideIsBlack(t *testing.T) {
	m := FromImage(smoothGray(rand.New(rand.NewSource(4)), 300, 300), 1)
	got := make([]float32, 3*32*32)
	for i := range got {
		got[i] = 7
	}
	far := align.Affine{A: 0.1, C: 5000, E: 0.1, F: -5000}
	if err := m.Warp(got, 32, far, Uniform(127.5, 127.5, false), Options{}); err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if v != -1 {
			t.Fatalf("value %d is %v, want -1 (black)", i, v)
		}
	}
}

func TestReduceNeverDividesByZero(t *testing.T) {
	for _, ratio := range []image.YCbCrSubsampleRatio{image.YCbCrSubsampleRatio420, image.YCbCrSubsampleRatio410, image.YCbCrSubsampleRatio444} {
		for w := 1; w <= 40; w++ {
			for k := 2; k <= 9; k++ {
				m := FromImage(image.NewYCbCr(image.Rect(0, 0, w, 41-w), ratio), 1)
				l := m.reduce(k, 1)
				if l.w != (w+k-1)/k || l.h != (41-w+k-1)/k {
					t.Fatalf("w=%d k=%d: reduced to %dx%d", w, k, l.w, l.h)
				}
			}
		}
	}
}

// Sampling raw and normalising afterwards must give exactly what sampling
// with the model's normalisation gives: the engine relies on it to judge a
// crop before spending an inference on it.
func TestApplyMatchesDirectSampling(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	m := FromImage(smoothYCbCr(rng, 300, 220, image.YCbCrSubsampleRatio420), 1)
	toCrop := align.Affine{A: 0.4, B: -0.1, C: -10, D: 0.1, E: 0.4, F: 3}
	const size = 64
	for _, n := range []Norm{
		Uniform(127.5, 127.5, false),
		Uniform(0, 1, true),
		{Mean: [3]float32{123.7, 116.3, 103.5}, Std: [3]float32{58.4, 57.1, 57.4}, BGR: true},
	} {
		direct, crop, applied := make([]float32, 3*size*size), make([]float32, 3*size*size), make([]float32, 3*size*size)
		if err := m.Warp(direct, size, toCrop, n, Options{}); err != nil {
			t.Fatal(err)
		}
		if err := m.Warp(crop, size, toCrop, RawRGB, Options{}); err != nil {
			t.Fatal(err)
		}
		n.Apply(applied, crop, size*size)
		for i := range direct {
			if direct[i] != applied[i] {
				t.Fatalf("norm %+v: value %d is %v directly and %v through Apply", n, i, direct[i], applied[i])
			}
		}
	}
}
