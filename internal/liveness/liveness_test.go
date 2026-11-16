package liveness

import (
	"bytes"
	"encoding/json"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/muhriddinsalohiddin/likeness/internal/fixtures"
	"github.com/muhriddinsalohiddin/likeness/internal/models"
	"github.com/muhriddinsalohiddin/likeness/internal/ortx"
	"github.com/muhriddinsalohiddin/likeness/internal/testenv"
	"github.com/muhriddinsalohiddin/likeness/internal/vision"
)

// reference is what the models' own code made of a picture: testdata was
// written by testdata/export.py, which runs the original implementation.
type reference struct {
	Picture string
	Box     [4]int // x, y, width, height
	Logits  map[string][]float32
	Real    float64
}

func references(t *testing.T) []reference {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "liveness.json"))
	if err != nil {
		t.Fatal(err)
	}
	var refs []reference
	if err := json.Unmarshal(raw, &refs); err != nil {
		t.Fatal(err)
	}
	return refs
}

func checker(t *testing.T) *Checker {
	t.Helper()
	rt := testenv.Runtime(t)
	var dir string
	for _, m := range models.Liveness {
		_, dir = testenv.Model(t, m.Name)
	}
	c, err := New(rt, models.Liveness, dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(c.Close)
	return c
}

// The models, fed the very crops the original code cut, give its scores.
func TestModelsAgreeWithTheOriginal(t *testing.T) {
	c := checker(t)
	for _, ref := range references(t) {
		for _, n := range c.nets {
			raw, err := os.ReadFile(filepath.Join("testdata", ref.Picture+"_"+n.m.Name+".png"))
			if err != nil {
				t.Fatal(err)
			}
			crop, err := png.Decode(bytes.NewReader(raw))
			if err != nil {
				t.Fatal(err)
			}
			in, err := ortx.NewFloat32(1, 3, int64(n.size), int64(n.size))
			if err != nil {
				t.Fatal(err)
			}
			defer in.Close()
			plane := n.size * n.size
			for y := 0; y < n.size; y++ {
				for x := 0; x < n.size; x++ {
					r, g, b, _ := crop.At(x, y).RGBA()
					i := y*n.size + x
					in.Float32()[i] = float32(b >> 8) // blue first, as OpenCV has it
					in.Float32()[plane+i] = float32(g >> 8)
					in.Float32()[2*plane+i] = float32(r >> 8)
				}
			}
			var out ortx.Outputs
			if err := n.sess.Run([]*ortx.Tensor{in}, &out); err != nil {
				t.Fatal(err)
			}
			for i, want := range ref.Logits[n.m.Name] {
				if got := out.Data[0][i]; math.Abs(float64(got-want)) > 1e-3 {
					t.Errorf("%s, %s: score %d is %v, the original gives %v", ref.Picture, n.m.Name, i, got, want)
				}
			}
		}
	}
}

// From the picture and the box to the answer, our own cropping included.
func TestRealAgreesWithTheOriginal(t *testing.T) {
	c := checker(t)
	if len(c.Names()) != 2 {
		t.Fatalf("models %v", c.Names())
	}
	for _, ref := range references(t) {
		raw, err := fixtures.Closeup(ref.Picture)
		if err != nil {
			t.Fatal(err)
		}
		img, err := vision.Decode(raw, vision.DefaultLimits)
		if err != nil {
			t.Fatal(err)
		}
		b := ref.Box
		got, err := c.Real(img, [4]float32{float32(b[0]), float32(b[1]), float32(b[0] + b[2]), float32(b[1] + b[3])})
		if err != nil {
			t.Fatal(err)
		}
		// Our JPEG decoder and OpenCV's differ by a level or two in places.
		if math.Abs(float64(got)-ref.Real) > 0.02 {
			t.Errorf("%s: %.4f, the original gives %.4f", ref.Picture, got, ref.Real)
		}
		t.Logf("%s: %.4f (original %.4f)", ref.Picture, got, ref.Real)
	}
	if _, err := c.Real(nil, [4]float32{}); err == nil {
		t.Error("no picture, and an answer all the same")
	}
}

func TestNewRefusesOtherModels(t *testing.T) {
	rt := testenv.Runtime(t)
	m, dir := testenv.Model(t, "yunet_2023mar")
	if _, err := New(rt, []*models.Manifest{m}, dir); err == nil {
		t.Error("a detector was taken for an anti-spoofing model")
	}
	if _, err := New(rt, nil, dir); err == nil {
		t.Error("a checker without models")
	}
}

// Worked out by hand from the reference implementation's rules.
func TestSurroundings(t *testing.T) {
	for _, c := range []struct {
		name          string
		box           [4]float32
		width, height int
		scale         float64
		want          [4]int
	}{
		// 100x100 about (200,200), grown 2.7 times: 270 wide about the same centre
		{"in the middle", [4]float32{150, 150, 250, 250}, 1280, 720, 2.7, [4]int{65, 65, 335, 335}},
		// sticks out on the left and the top: moved in, the size kept
		{"in the corner", [4]float32{10, 20, 110, 120}, 1280, 720, 2.7, [4]int{0, 0, 270, 270}},
		// sticks out on the right and the bottom
		{"in the far corner", [4]float32{1170, 610, 1270, 710}, 1280, 720, 2.7, [4]int{1009, 449, 1279, 719}},
		// 300 high in a picture 448 high cannot grow 4 times: (448-1)/300 at most
		{"a face that fills the picture", [4]float32{100, 74, 300, 374}, 448, 448, 4, [4]int{51, 0, 349, 447}},
		// fractions of a pixel are dropped before anything else
		{"fractions", [4]float32{150.9, 150.9, 250.95, 250.95}, 1280, 720, 2.7, [4]int{65, 65, 335, 335}},
		{"no box at all", [4]float32{5, 5, 5, 5}, 640, 480, 2.7, [4]int{0, 0, 639, 479}},
	} {
		x1, y1, x2, y2 := surroundings(c.box, c.width, c.height, c.scale)
		if got := [4]int{x1, y1, x2, y2}; got != c.want {
			t.Errorf("%s: %v, want %v", c.name, got, c.want)
		}
	}
}

func TestResizeMapsCentresOntoCentres(t *testing.T) {
	// pixels 100..259 (160 of them) onto 80: two source pixels a target one
	m := resize(100, 40, 259, 199, 80)
	for _, c := range []struct{ x, y, u, v float32 }{
		{100.5, 40.5, 0, 0},    // between the first two source pixels: the centre of the first target pixel
		{258.5, 198.5, 79, 79}, // and the last
		{180, 120, 39.75, 39.75},
	} {
		u, v := m.A*c.x+m.B*c.y+m.C, m.D*c.x+m.E*c.y+m.F
		if math.Abs(float64(u-c.u)) > 1e-4 || math.Abs(float64(v-c.v)) > 1e-4 {
			t.Errorf("(%v, %v) lands on (%v, %v), want (%v, %v)", c.x, c.y, u, v, c.u, c.v)
		}
	}
}

func TestSoftmax(t *testing.T) {
	p := softmax([]float32{0, 0, 0})
	for _, v := range p {
		if math.Abs(v-1.0/3) > 1e-12 {
			t.Errorf("equal scores: %v", p)
		}
	}
	p = softmax([]float32{-2.3442819, 3.5587418, -1.2169279})
	if math.Abs(p[live]-0.98896) > 1e-5 || math.Abs(p[0]+p[1]+p[2]-1) > 1e-12 {
		t.Errorf("the reference scores of a live face: %v", p)
	}
	// large scores must not overflow
	if p = softmax([]float32{1000, 999, -1000}); math.IsNaN(p[0]) || math.Abs(p[0]-0.7310586) > 1e-6 {
		t.Errorf("large scores: %v", p)
	}
}
