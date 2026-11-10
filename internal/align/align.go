// Package align computes the similarity transform that maps detected facial
// landmarks onto the canonical layout expected by face embedding models.
package align

import (
	"errors"
	"math"
)

// Point is a position in pixel coordinates. Pixel centres sit on integers,
// the convention used by OpenCV's warpAffine.
type Point struct{ X, Y float32 }

// Affine is the 2x3 matrix
//
//	| A B C |
//	| D E F |
//
// mapping (x, y) to (A*x + B*y + C, D*x + E*y + F).
type Affine struct{ A, B, C, D, E, F float32 }

// Identity leaves points unchanged.
var Identity = Affine{A: 1, E: 1}

// ErrDegenerate is returned when the input does not determine a transform,
// for example when all landmarks coincide.
var ErrDegenerate = errors.New("align: degenerate input")

// Apply transforms p.
func (m Affine) Apply(p Point) Point {
	return Point{m.A*p.X + m.B*p.Y + m.C, m.D*p.X + m.E*p.Y + m.F}
}

// Mul returns the transform that applies n first and m second.
func (m Affine) Mul(n Affine) Affine {
	return Affine{
		A: m.A*n.A + m.B*n.D, B: m.A*n.B + m.B*n.E, C: m.A*n.C + m.B*n.F + m.C,
		D: m.D*n.A + m.E*n.D, E: m.D*n.B + m.E*n.E, F: m.D*n.C + m.E*n.F + m.F,
	}
}

// Invert returns the inverse transform.
func (m Affine) Invert() (Affine, error) {
	det := float64(m.A)*float64(m.E) - float64(m.B)*float64(m.D)
	if math.Abs(det) < 1e-12 {
		return Affine{}, ErrDegenerate
	}
	a, b := float64(m.E)/det, -float64(m.B)/det
	d, e := -float64(m.D)/det, float64(m.A)/det
	return Affine{
		A: float32(a), B: float32(b), C: float32(-(a*float64(m.C) + b*float64(m.F))),
		D: float32(d), E: float32(e), F: float32(-(d*float64(m.C) + e*float64(m.F))),
	}, nil
}

// Scale is the factor by which the transform stretches lengths (the square
// root of the absolute determinant).
func (m Affine) Scale() float32 {
	det := float64(m.A)*float64(m.E) - float64(m.B)*float64(m.D)
	return float32(math.Sqrt(math.Abs(det)))
}

// Reference112 holds the landmark positions of a 112x112 aligned face: left
// eye, right eye, nose tip, left and right mouth corner, all as seen in the
// image. ArcFace-family embedders are trained on crops laid out this way.
var Reference112 = [5]Point{
	{38.2946, 51.6963}, {73.5318, 51.5014}, {56.0252, 71.7366},
	{41.5493, 92.3655}, {70.7299, 92.2041},
}

// Similarity estimates the least-squares similarity transform (rotation,
// uniform scale, translation, no reflection) that takes src onto dst. It
// agrees with Umeyama's method, which reference pipelines use through
// scikit-image.
func Similarity(src, dst []Point) (Affine, error) {
	if len(src) != len(dst) || len(src) < 2 {
		return Affine{}, ErrDegenerate
	}
	n := float64(len(src))
	var mx, my, nx, ny float64
	for i := range src {
		mx += float64(src[i].X)
		my += float64(src[i].Y)
		nx += float64(dst[i].X)
		ny += float64(dst[i].Y)
	}
	mx, my, nx, ny = mx/n, my/n, nx/n, ny/n

	// With points as complex numbers the optimum is z = sum(conj(p) q) / sum(|p|^2).
	var re, im, den float64
	for i := range src {
		px, py := float64(src[i].X)-mx, float64(src[i].Y)-my
		qx, qy := float64(dst[i].X)-nx, float64(dst[i].Y)-ny
		re += px*qx + py*qy
		im += px*qy - py*qx
		den += px*px + py*py
	}
	if den < 1e-9 {
		return Affine{}, ErrDegenerate
	}
	a, b := re/den, im/den
	if a*a+b*b < 1e-12 {
		return Affine{}, ErrDegenerate
	}
	return Affine{
		A: float32(a), B: float32(-b), C: float32(nx - (a*mx - b*my)),
		D: float32(b), E: float32(a), F: float32(ny - (b*mx + a*my)),
	}, nil
}

// Crop returns the transform from image coordinates to a size x size aligned
// face crop for the given landmarks.
func Crop(landmarks [5]Point, size int) (Affine, error) {
	if size <= 0 {
		return Affine{}, ErrDegenerate
	}
	ref := Reference112
	if size != 112 {
		k := float32(size) / 112
		for i := range ref {
			ref[i].X *= k
			ref[i].Y *= k
		}
	}
	return Similarity(landmarks[:], ref[:])
}
