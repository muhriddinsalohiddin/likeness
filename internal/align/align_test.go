package align

import (
	"math"
	"math/rand"
	"testing"
)

func near(a, b, tol float32) bool { return float32(math.Abs(float64(a-b))) <= tol }

func affineNear(t *testing.T, got, want Affine, tol float32) {
	t.Helper()
	g := [6]float32{got.A, got.B, got.C, got.D, got.E, got.F}
	w := [6]float32{want.A, want.B, want.C, want.D, want.E, want.F}
	for i := range g {
		if !near(g[i], w[i], tol) {
			t.Fatalf("affine mismatch:\n got  %v\n want %v", got, want)
		}
	}
}

func sim(scale, angle, tx, ty float64) Affine {
	c, s := scale*math.Cos(angle), scale*math.Sin(angle)
	return Affine{float32(c), float32(-s), float32(tx), float32(s), float32(c), float32(ty)}
}

func TestSimilarityRecoversKnownTransform(t *testing.T) {
	src := []Point{{10, 20}, {200, 35}, {120, 140}, {40, 230}, {190, 220}}
	for _, want := range []Affine{
		Identity,
		sim(0.37, 0.3, 12.5, -40),
		sim(2.4, -2.9, -300, 95),
		sim(1, math.Pi, 640, 480),
	} {
		dst := applyAll(src, want)
		got, err := Similarity(src, dst)
		if err != nil {
			t.Fatal(err)
		}
		affineNear(t, got, want, 2e-3)
	}
}

func applyAll(src []Point, m Affine) []Point {
	dst := make([]Point, len(src))
	for i, p := range src {
		dst[i] = m.Apply(p)
	}
	return dst
}

// umeyama2D is an independent derivation through the cross-covariance matrix:
// the best proper rotation maximises trace(R^T * cov), the scale follows from
// the source variance.
func umeyama2D(src, dst []Point) Affine {
	n := float64(len(src))
	var mx, my, nx, ny float64
	for i := range src {
		mx += float64(src[i].X) / n
		my += float64(src[i].Y) / n
		nx += float64(dst[i].X) / n
		ny += float64(dst[i].Y) / n
	}
	var c00, c01, c10, c11, variance float64
	for i := range src {
		px, py := float64(src[i].X)-mx, float64(src[i].Y)-my
		qx, qy := float64(dst[i].X)-nx, float64(dst[i].Y)-ny
		c00 += qx * px / n
		c01 += qx * py / n
		c10 += qy * px / n
		c11 += qy * py / n
		variance += (px*px + py*py) / n
	}
	theta := math.Atan2(c10-c01, c00+c11)
	cos, sin := math.Cos(theta), math.Sin(theta)
	scale := (cos*(c00+c11) + sin*(c10-c01)) / variance
	return sim(scale, theta, nx-scale*(cos*mx-sin*my), ny-scale*(sin*mx+cos*my))
}

func TestSimilarityMatchesUmeyamaOnNoisyLandmarks(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for trial := 0; trial < 200; trial++ {
		var lm [5]Point
		truth := sim(0.2+rng.Float64()*6, rng.Float64()*2*math.Pi, rng.Float64()*2000, rng.Float64()*2000)
		inv, err := truth.Invert()
		if err != nil {
			t.Fatal(err)
		}
		for i, r := range Reference112 {
			p := inv.Apply(r)
			lm[i] = Point{p.X + float32(rng.NormFloat64()), p.Y + float32(rng.NormFloat64())}
		}
		got, err := Similarity(lm[:], Reference112[:])
		if err != nil {
			t.Fatal(err)
		}
		want := umeyama2D(lm[:], Reference112[:])
		tol := float32(1e-3) * (1 + float32(math.Abs(float64(want.C))+math.Abs(float64(want.F))))
		affineNear(t, got, want, tol)
	}
}

func TestSimilarityIsALocalOptimum(t *testing.T) {
	lm := []Point{{412.3, 288.9}, {539.8, 301.2}, {470.1, 371.4}, {418.7, 436.6}, {521.4, 447.3}}
	best, err := Similarity(lm, Reference112[:])
	if err != nil {
		t.Fatal(err)
	}
	cost := func(m Affine) float64 {
		var c float64
		for i, p := range lm {
			q := m.Apply(p)
			dx, dy := float64(q.X-Reference112[i].X), float64(q.Y-Reference112[i].Y)
			c += dx*dx + dy*dy
		}
		return c
	}
	base := cost(best)
	for _, d := range []struct{ da, db, dc, df float32 }{
		{1e-3, 0, 0, 0}, {-1e-3, 0, 0, 0}, {0, 1e-3, 0, 0}, {0, -1e-3, 0, 0},
		{0, 0, 0.2, 0}, {0, 0, -0.2, 0}, {0, 0, 0, 0.2}, {0, 0, 0, -0.2},
	} {
		m := best
		m.A, m.E = m.A+d.da, m.E+d.da
		m.B, m.D = m.B-d.db, m.D+d.db
		m.C, m.F = m.C+d.dc, m.F+d.df
		if c := cost(m); c < base-1e-6 {
			t.Fatalf("perturbation %+v lowers the cost: %g < %g", d, c, base)
		}
	}
}

func TestSimilarityRejectsDegenerateInput(t *testing.T) {
	same := []Point{{5, 5}, {5, 5}, {5, 5}}
	if _, err := Similarity(same, []Point{{0, 0}, {1, 0}, {0, 1}}); err == nil {
		t.Fatal("coincident source points must be rejected")
	}
	if _, err := Similarity([]Point{{0, 0}, {1, 0}, {0, 1}}, same); err == nil {
		t.Fatal("coincident target points must be rejected")
	}
	if _, err := Similarity([]Point{{0, 0}}, []Point{{1, 1}}); err == nil {
		t.Fatal("a single point must be rejected")
	}
	if _, err := Similarity([]Point{{0, 0}, {1, 1}}, []Point{{0, 0}}); err == nil {
		t.Fatal("mismatched lengths must be rejected")
	}
}

func TestInvertRoundTrip(t *testing.T) {
	m := sim(1.7, 0.9, -33, 210)
	inv, err := m.Invert()
	if err != nil {
		t.Fatal(err)
	}
	affineNear(t, m.Mul(inv), Identity, 1e-4)
	affineNear(t, inv.Mul(m), Identity, 1e-4)
	if _, err := (Affine{}).Invert(); err == nil {
		t.Fatal("a singular matrix must not invert")
	}
	if s := m.Scale(); !near(s, 1.7, 1e-4) {
		t.Fatalf("Scale() = %v, want 1.7", s)
	}
}

func TestMulAppliesRightOperandFirst(t *testing.T) {
	shift := Affine{A: 1, E: 1, C: 10}
	double := Affine{A: 2, E: 2}
	if got := double.Mul(shift).Apply(Point{1, 1}); got != (Point{22, 2}) {
		t.Fatalf("got %v, want {22 2}", got)
	}
}

func TestCropScalesTheReference(t *testing.T) {
	var lm [5]Point
	for i, r := range Reference112 {
		lm[i] = Point{r.X*3 + 100, r.Y*3 + 50}
	}
	m, err := Crop(lm, 224)
	if err != nil {
		t.Fatal(err)
	}
	for i, r := range Reference112 {
		got := m.Apply(lm[i])
		if !near(got.X, r.X*2, 1e-2) || !near(got.Y, r.Y*2, 1e-2) {
			t.Fatalf("landmark %d maps to %v, want %v", i, got, Point{r.X * 2, r.Y * 2})
		}
	}
	if _, err := Crop(lm, 0); err == nil {
		t.Fatal("size 0 must be rejected")
	}
}
