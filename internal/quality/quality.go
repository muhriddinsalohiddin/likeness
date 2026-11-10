// Package quality measures how suitable a detected face is for recognition
// and checks the measurements against a policy.
//
// Geometry (size, pose, how much of the face is in the picture) comes from
// the detection alone and costs nothing. Pixel measures (brightness,
// sharpness) need the aligned crop. A face that fails either is not worth an
// inference.
package quality

import (
	"math"

	"github.com/muhriddinsalohiddin/likeness/internal/align"
)

// Measures describes one face. Zero-valued pixel fields mean they have not
// been measured.
type Measures struct {
	// Score is the detector's confidence, 0..1.
	Score float32
	// EyeDistance is the distance between the eye centres in picture pixels.
	// A face is roughly 2.3 times as wide.
	EyeDistance float32
	// Yaw, Pitch and Roll are rough head angles in degrees, estimated from
	// five landmarks and a generic head shape; expect errors of ten degrees.
	// Yaw is positive when the face turns towards the right edge of the
	// picture, Pitch when it tilts down, Roll when it leans clockwise.
	Yaw, Pitch, Roll float32
	// InFrame is the fraction of the face box that lies inside the picture.
	InFrame float32
	// Brightness is the mean luma of the face area of the aligned crop, 0..255.
	Brightness float32
	// Sharpness is the variance of the Laplacian over the same area, taken
	// relative to its brightness (and scaled by 10^4) so that a dim picture
	// does not pass for a blurred one. A sharp close-up scores a few hundred;
	// below about 40 the fine detail is gone. Blurred and upscaled faces
	// score low.
	Sharpness float32
}

// Reason says why a face, or a picture, was turned down.
type Reason string

const (
	OK            Reason = ""
	NoFace        Reason = "no_face"
	MultipleFaces Reason = "multiple_faces"
	LowScore      Reason = "low_det_score"
	OutOfFrame    Reason = "out_of_frame"
	TooSmall      Reason = "face_too_small"
	BadPose       Reason = "bad_pose"
	TooDark       Reason = "too_dark"
	TooBright     Reason = "too_bright"
	Blurry        Reason = "blurry"
)

// Policy holds the limits a face must meet. A zero field sets no limit, so
// the zero Policy accepts everything.
type Policy struct {
	MinScore       float32
	MinEyeDistance float32
	// MaxYaw, MaxPitch and MaxRoll bound the absolute angles, in degrees.
	MaxYaw, MaxPitch, MaxRoll float32
	MinInFrame                float32
	MinBrightness             float32
	MaxBrightness             float32
	MinSharpness              float32
	// MaxRival bounds how large a second face may be relative to the main
	// one (by eye distance) before the picture counts as showing several
	// people. Only consulted when exactly one face is expected.
	MaxRival float32
}

// The head shape behind the pose estimate, in units of the eye distance and
// taken from the reference layout of an aligned face: how far the nose tip
// stands out from the plane of the eyes and mouth, and where the nose sits
// between the eye line and the mouth line when the face is frontal.
const (
	noseDepth    = 0.43
	frontalSplit = -0.0101 // (eyes-to-nose - nose-to-mouth) / (eyes-to-mouth)
	pitchGain    = 0.745   // 2 * noseDepth / (eyes-to-mouth)
)

const degrees = 180 / math.Pi

// Geometry fills in everything that follows from the detection: box is
// x1, y1, x2, y2 and the landmarks are left eye, right eye, nose, left and
// right mouth corner as seen in a picture of w x h pixels.
func Geometry(box [4]float32, lm [5]align.Point, score float32, w, h int) Measures {
	m := Measures{Score: score, InFrame: inFrame(box, w, h)}

	ex, ey := float64(lm[1].X-lm[0].X), float64(lm[1].Y-lm[0].Y)
	iod := math.Hypot(ex, ey)
	m.EyeDistance = float32(iod)
	if iod < 1e-6 {
		return m
	}
	m.Roll = float32(math.Atan2(ey, ex) * degrees)

	// Work in a frame where the eye line is horizontal, its midpoint is the
	// origin and the eye distance is the unit.
	cos, sin := ex/iod, ey/iod
	cx, cy := float64(lm[0].X+lm[1].X)/2, float64(lm[0].Y+lm[1].Y)/2
	frame := func(p align.Point) (x, y float64) {
		dx, dy := float64(p.X)-cx, float64(p.Y)-cy
		return (cos*dx + sin*dy) / iod, (-sin*dx + cos*dy) / iod
	}
	nx, ny := frame(lm[2])
	lx, ly := frame(lm[3])
	rx, ry := frame(lm[4])
	mx, my := (lx+rx)/2, (ly+ry)/2
	if my < 1e-6 {
		return m // mouth on or above the eye line: not a face we can read
	}

	// Nodding moves the nose tip, which stands out from the face, between
	// the eye line and the mouth line.
	split := (ny - (my - ny)) / my
	pitch := math.Atan((split - frontalSplit) / pitchGain)
	m.Pitch = float32(pitch * degrees)

	// Turning sideways slides it off the midline that runs from between the
	// eyes to between the mouth corners. Measuring from the midline rather
	// than from the eyes keeps a nod from being read as a turn.
	offset := nx - mx*ny/my
	m.Yaw = float32(math.Atan(offset*math.Cos(pitch)/noseDepth) * degrees)
	return m
}

func inFrame(box [4]float32, w, h int) float32 {
	area := (box[2] - box[0]) * (box[3] - box[1])
	if area <= 0 {
		return 0
	}
	iw := min(box[2], float32(w)) - max(box[0], 0)
	ih := min(box[3], float32(h)) - max(box[1], 0)
	if iw <= 0 || ih <= 0 {
		return 0
	}
	return iw * ih / area
}

// The part of an aligned crop that is always face: from just outside the
// eyes to the mouth corners, from the brows to below the mouth. Hair,
// background and clothing stay out of the pixel measures.
const (
	faceLeft, faceRight = 0.20, 0.80
	faceTop, faceBottom = 0.25, 0.90
)

// Pixels measures brightness and sharpness on an aligned crop: three planes
// (R, G, B) of size x size values on the 0..255 scale.
func (m *Measures) Pixels(crop []float32, size int) {
	area := size * size
	if size < 8 || len(crop) < 3*area {
		return
	}
	x0, x1 := int(faceLeft*float32(size)), int(faceRight*float32(size))
	y0, y1 := int(faceTop*float32(size)), int(faceBottom*float32(size))
	r, g, b := crop[:area], crop[area:2*area], crop[2*area:3*area]
	luma := func(i int) float64 {
		return 0.299*float64(r[i]) + 0.587*float64(g[i]) + 0.114*float64(b[i])
	}

	var sum, lap, lap2 float64
	n := 0
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			i := y*size + x
			c := luma(i)
			sum += c
			// the window never touches the border, so all neighbours exist
			l := 4*c - luma(i-1) - luma(i+1) - luma(i-size) - luma(i+size)
			lap += l
			lap2 += l * l
			n++
		}
	}
	if n == 0 {
		return
	}
	brightness := sum / float64(n)
	m.Brightness = float32(brightness)
	if brightness < 1 {
		return // black: nothing to be sharp
	}
	mean := lap / float64(n)
	m.Sharpness = float32(1e4 * (lap2/float64(n) - mean*mean) / (brightness * brightness))
}

func exceeds(v, limit float32) bool {
	return limit > 0 && float32(math.Abs(float64(v))) > limit
}

// CheckGeometry tests the limits that Geometry's measurements answer.
func (p Policy) CheckGeometry(m Measures) Reason {
	switch {
	case m.Score < p.MinScore:
		return LowScore
	case m.InFrame < p.MinInFrame:
		return OutOfFrame
	case m.EyeDistance < p.MinEyeDistance:
		return TooSmall
	case exceeds(m.Yaw, p.MaxYaw), exceeds(m.Pitch, p.MaxPitch), exceeds(m.Roll, p.MaxRoll):
		return BadPose
	}
	return OK
}

// CheckPixels tests the limits that need Pixels to have run.
func (p Policy) CheckPixels(m Measures) Reason {
	switch {
	case m.Brightness < p.MinBrightness:
		return TooDark
	case p.MaxBrightness > 0 && m.Brightness > p.MaxBrightness:
		return TooBright
	case m.Sharpness < p.MinSharpness:
		return Blurry
	}
	return OK
}
