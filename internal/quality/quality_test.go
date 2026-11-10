package quality

import (
	"math"
	"testing"

	"github.com/muhriddinsalohiddin/likeness/internal/align"
)

// headLandmarks projects a generic head, turned by the given angles in
// degrees, onto a picture: eyes and mouth corners in one plane, the nose tip
// standing out towards the camera. x grows to the right, y downwards.
func headLandmarks(yaw, pitch, roll, eyeDistance float64, cx, cy float64) [5]align.Point {
	return turnedHead(yaw, pitch, roll, eyeDistance, cx, cy, false)
}

// turnedHead also lets the turn come before the nod; real heads do both at
// once and no single order is right.
func turnedHead(yaw, pitch, roll, eyeDistance float64, cx, cy float64, turnFirst bool) [5]align.Point {
	const (
		eyeY, mouthY = -0.5716, 0.5830 // from the 112x112 reference layout, in eye distances
		mouthHalf    = 0.4140
	)
	model := [5][3]float64{
		{-0.5, eyeY, 0}, {0.5, eyeY, 0}, {0, 0, noseDepth}, {-mouthHalf, mouthY, 0}, {mouthHalf, mouthY, 0},
	}
	y, p, r := yaw/degrees, pitch/degrees, roll/degrees
	var out [5]align.Point
	for i, v := range model {
		px, py, pz := v[0], v[1], v[2]
		// nod: the nose tip moves down for a positive pitch
		nod := func() { py, pz = py*math.Cos(p)+pz*math.Sin(p), -py*math.Sin(p)+pz*math.Cos(p) }
		// turn: the nose tip moves right for a positive yaw
		turn := func() { px, pz = px*math.Cos(y)+pz*math.Sin(y), -px*math.Sin(y)+pz*math.Cos(y) }
		if turnFirst {
			turn()
			nod()
		} else {
			nod()
			turn()
		}
		// lean: clockwise in the picture for a positive roll
		px, py = px*math.Cos(r)-py*math.Sin(r), px*math.Sin(r)+py*math.Cos(r)
		out[i] = align.Point{X: float32(cx + px*eyeDistance), Y: float32(cy + py*eyeDistance)}
	}
	return out
}

func TestReferenceLayoutIsFrontal(t *testing.T) {
	m := Geometry([4]float32{20, 10, 92, 105}, align.Reference112, 0.9, 112, 112)
	if math.Abs(float64(m.Yaw)) > 1 || math.Abs(float64(m.Pitch)) > 1 || math.Abs(float64(m.Roll)) > 1 {
		t.Fatalf("the reference layout should read as frontal, got yaw %.1f pitch %.1f roll %.1f", m.Yaw, m.Pitch, m.Roll)
	}
	if math.Abs(float64(m.EyeDistance)-35.24) > 0.01 {
		t.Fatalf("eye distance %.2f, want 35.24", m.EyeDistance)
	}
	if m.InFrame != 1 || m.Score != 0.9 {
		t.Fatalf("in frame %.2f, score %.2f", m.InFrame, m.Score)
	}
}

func TestPoseRecoversSingleAngles(t *testing.T) {
	for _, yaw := range []float64{-50, -30, -10, 0, 15, 35, 55} {
		m := Geometry([4]float32{}, headLandmarks(yaw, 0, 0, 80, 300, 200), 1, 640, 480)
		if math.Abs(float64(m.Yaw)-yaw) > 0.5 || math.Abs(float64(m.Pitch)) > 0.5 || math.Abs(float64(m.Roll)) > 0.5 {
			t.Errorf("yaw %v: got yaw %.1f pitch %.1f roll %.1f", yaw, m.Yaw, m.Pitch, m.Roll)
		}
	}
	for _, pitch := range []float64{-35, -15, 0, 10, 30} {
		m := Geometry([4]float32{}, headLandmarks(0, pitch, 0, 80, 300, 200), 1, 640, 480)
		if math.Abs(float64(m.Pitch)-pitch) > 1 || math.Abs(float64(m.Yaw)) > 0.5 {
			t.Errorf("pitch %v: got pitch %.1f yaw %.1f", pitch, m.Pitch, m.Yaw)
		}
	}
	for _, roll := range []float64{-170, -90, -30, 0, 20, 90, 135} {
		m := Geometry([4]float32{}, headLandmarks(0, 0, roll, 80, 300, 200), 1, 640, 480)
		if math.Abs(float64(m.Roll)-roll) > 0.5 || math.Abs(float64(m.Yaw)) > 0.5 || math.Abs(float64(m.Pitch)) > 1 {
			t.Errorf("roll %v: got roll %.1f yaw %.1f pitch %.1f", roll, m.Roll, m.Yaw, m.Pitch)
		}
	}
}

func TestPoseWithCombinedAnglesStaysClose(t *testing.T) {
	for _, turnFirst := range []bool{false, true} {
		var worstYaw, worstPitch float64
		for yaw := -40.0; yaw <= 40; yaw += 10 {
			for pitch := -20.0; pitch <= 20; pitch += 10 {
				for _, roll := range []float64{-25, 0, 25} {
					for _, size := range []float64{20, 150} {
						m := Geometry([4]float32{}, turnedHead(yaw, pitch, roll, size, 500, 400, turnFirst), 1, 1000, 800)
						worstYaw = math.Max(worstYaw, math.Abs(float64(m.Yaw)-yaw))
						worstPitch = math.Max(worstPitch, math.Abs(float64(m.Pitch)-pitch))
					}
				}
			}
		}
		t.Logf("turn first=%v, combined turns up to 40/20/25 degrees: yaw off by at most %.1f, pitch by %.1f", turnFirst, worstYaw, worstPitch)
		if worstYaw > 6 || worstPitch > 6 {
			t.Errorf("the pose estimate drifts too far when angles combine (turn first=%v)", turnFirst)
		}
	}
}

func TestGeometryDegenerateInput(t *testing.T) {
	var lm [5]align.Point // all landmarks coincide
	m := Geometry([4]float32{0, 0, 10, 10}, lm, 0.5, 100, 100)
	for name, v := range map[string]float32{"yaw": m.Yaw, "pitch": m.Pitch, "roll": m.Roll, "eyes": m.EyeDistance} {
		if v != 0 || math.IsNaN(float64(v)) {
			t.Errorf("%s = %v for coincident landmarks", name, v)
		}
	}
	// eyes apart, mouth on the eye line: pitch has nothing to stand on
	lm = [5]align.Point{{X: 10, Y: 10}, {X: 30, Y: 10}, {X: 20, Y: 10}, {X: 12, Y: 10}, {X: 28, Y: 10}}
	if m := Geometry([4]float32{}, lm, 1, 100, 100); math.IsNaN(float64(m.Pitch)) || math.IsInf(float64(m.Pitch), 0) {
		t.Fatalf("pitch = %v", m.Pitch)
	}
}

func TestInFrame(t *testing.T) {
	for _, c := range []struct {
		box  [4]float32
		want float32
	}{
		{[4]float32{10, 10, 50, 50}, 1},
		{[4]float32{-20, 0, 20, 40}, 0.5},
		{[4]float32{80, 80, 120, 120}, 0.25},
		{[4]float32{200, 200, 240, 240}, 0},
		{[4]float32{-50, -50, -10, -10}, 0},
		{[4]float32{30, 30, 30, 60}, 0}, // empty box
	} {
		if got := inFrame(c.box, 100, 100); math.Abs(float64(got-c.want)) > 1e-6 {
			t.Errorf("box %v: %v, want %v", c.box, got, c.want)
		}
	}
}

const size = 112

func crop(f func(x, y int) (r, g, b float32)) []float32 {
	out := make([]float32, 3*size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			r, g, b := f(x, y)
			i := y*size + x
			out[i], out[size*size+i], out[2*size*size+i] = r, g, b
		}
	}
	return out
}

func TestPixels(t *testing.T) {
	var m Measures
	m.Pixels(crop(func(x, y int) (float32, float32, float32) { return 90, 90, 90 }), size)
	if math.Abs(float64(m.Brightness)-90) > 1e-3 || m.Sharpness > 1e-6 {
		t.Fatalf("flat grey: brightness %v sharpness %v", m.Brightness, m.Sharpness)
	}

	// luma weights: a pure green crop is brighter than a pure blue one
	var green, blue Measures
	green.Pixels(crop(func(x, y int) (float32, float32, float32) { return 0, 200, 0 }), size)
	blue.Pixels(crop(func(x, y int) (float32, float32, float32) { return 0, 0, 200 }), size)
	if math.Abs(float64(green.Brightness)-117.4) > 0.1 || math.Abs(float64(blue.Brightness)-22.8) > 0.1 {
		t.Fatalf("green %v, blue %v", green.Brightness, blue.Brightness)
	}

	// a linear ramp has no second derivative; a checkerboard has plenty
	var ramp, board Measures
	ramp.Pixels(crop(func(x, y int) (float32, float32, float32) { v := float32(x + y); return v, v, v }), size)
	board.Pixels(crop(func(x, y int) (float32, float32, float32) {
		v := float32(50 + 100*((x+y)%2))
		return v, v, v
	}), size)
	if ramp.Sharpness > 1e-3 {
		t.Fatalf("ramp sharpness %v, want 0", ramp.Sharpness)
	}
	// Laplacian of +-400 around a mean of 100: 1e4 * 160000 / 100^2
	if math.Abs(float64(board.Sharpness)-160000) > 1 {
		t.Fatalf("checkerboard sharpness %v, want 160000", board.Sharpness)
	}

	// half the light, the same sharpness
	var dim Measures
	dim.Pixels(crop(func(x, y int) (float32, float32, float32) {
		v := float32(25 + 50*((x+y)%2))
		return v, v, v
	}), size)
	if math.Abs(float64(dim.Sharpness-board.Sharpness)) > 1 || math.Abs(float64(dim.Brightness)-50) > 1e-3 {
		t.Fatalf("dimmed checkerboard: sharpness %v brightness %v", dim.Sharpness, dim.Brightness)
	}
	var black Measures
	black.Pixels(crop(func(x, y int) (float32, float32, float32) { return 0, 0, 0 }), size)
	if black.Sharpness != 0 || black.Brightness != 0 {
		t.Fatalf("black crop: %+v", black)
	}

	// what lies outside the face area does not count
	var framed Measures
	framed.Pixels(crop(func(x, y int) (float32, float32, float32) {
		if x < 15 || x > 96 || y < 20 || y > 104 {
			return float32(255 * ((x + y) % 2)), 255, 0
		}
		return 90, 90, 90
	}), size)
	if math.Abs(float64(framed.Brightness)-90) > 1e-3 || framed.Sharpness > 1e-6 {
		t.Fatalf("border leaked into the measures: brightness %v sharpness %v", framed.Brightness, framed.Sharpness)
	}

	// buffers that cannot hold a crop are ignored
	var none Measures
	none.Pixels(make([]float32, 10), size)
	none.Pixels(nil, 4)
	if none != (Measures{}) {
		t.Fatalf("got %+v from an unusable buffer", none)
	}
}

func TestPolicy(t *testing.T) {
	good := Measures{Score: 0.9, EyeDistance: 60, Yaw: 5, Pitch: -4, Roll: 3, InFrame: 1, Brightness: 120, Sharpness: 300}
	p := Policy{
		MinScore: 0.6, MinEyeDistance: 40, MaxYaw: 25, MaxPitch: 25, MaxRoll: 30,
		MinInFrame: 0.9, MinBrightness: 50, MaxBrightness: 210, MinSharpness: 50,
	}
	if r := p.CheckGeometry(good); r != OK {
		t.Fatalf("geometry of a good face: %q", r)
	}
	if r := p.CheckPixels(good); r != OK {
		t.Fatalf("pixels of a good face: %q", r)
	}
	geometry := map[Reason]func(*Measures){
		LowScore:   func(m *Measures) { m.Score = 0.55 },
		OutOfFrame: func(m *Measures) { m.InFrame = 0.7 },
		TooSmall:   func(m *Measures) { m.EyeDistance = 30 },
		BadPose:    func(m *Measures) { m.Yaw = -40 },
	}
	for want, spoil := range geometry {
		m := good
		spoil(&m)
		if got := p.CheckGeometry(m); got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	}
	for _, spoil := range []func(*Measures){
		func(m *Measures) { m.Pitch = 30 },
		func(m *Measures) { m.Roll = -45 },
	} {
		m := good
		spoil(&m)
		if got := p.CheckGeometry(m); got != BadPose {
			t.Errorf("got %q, want %q", got, BadPose)
		}
	}
	pixels := map[Reason]func(*Measures){
		TooDark:   func(m *Measures) { m.Brightness = 20 },
		TooBright: func(m *Measures) { m.Brightness = 240 },
		Blurry:    func(m *Measures) { m.Sharpness = 10 },
	}
	for want, spoil := range pixels {
		m := good
		spoil(&m)
		if got := p.CheckPixels(m); got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	}

	// the first failing limit wins, in a fixed order
	bad := Measures{Score: 0.1, EyeDistance: 5, Yaw: 80, InFrame: 0.1}
	if got := p.CheckGeometry(bad); got != LowScore {
		t.Errorf("got %q, want %q first", got, LowScore)
	}

	// no limits, no objections
	var none Policy
	if none.CheckGeometry(Measures{}) != OK || none.CheckPixels(Measures{}) != OK {
		t.Error("the zero policy must accept everything")
	}
}
