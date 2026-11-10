package detect

import (
	"math"
	"testing"

	"github.com/muhriddinsalohiddin/likeness/internal/align"
	"github.com/muhriddinsalohiddin/likeness/internal/ortx"
)

func near(a, b, tol float32) bool { return float32(math.Abs(float64(a-b))) <= tol }

func faceNear(t *testing.T, got, want Face, tol float32) {
	t.Helper()
	for i := range want.Box {
		if !near(got.Box[i], want.Box[i], tol) {
			t.Fatalf("box %v, want %v", got.Box, want.Box)
		}
	}
	if !near(got.Score, want.Score, 1e-4) {
		t.Fatalf("score %v, want %v", got.Score, want.Score)
	}
	for i := range want.Landmarks {
		if !near(got.Landmarks[i].X, want.Landmarks[i].X, tol) || !near(got.Landmarks[i].Y, want.Landmarks[i].Y, tol) {
			t.Fatalf("landmark %d is %v, want %v", i, got.Landmarks[i], want.Landmarks[i])
		}
	}
}

// scrfdOutputs builds the nine SCRFD outputs for a 64x64 input in a
// scrambled order, with one detection at stride 16, cell (row 1, col 2),
// second anchor.
func scrfdOutputs() *ortx.Outputs {
	const size = 64
	var out ortx.Outputs
	add := func(n, width int64, fill func([]float32)) {
		data := make([]float32, n*width)
		if fill != nil {
			fill(data)
		}
		out.Data = append(out.Data, data)
		out.Dims = append(out.Dims, []int64{n, width})
	}
	// stride 16: 4x4 cells x 2 anchors = 32 entries; index (1*4+2)*2+1 = 13
	const i = 13
	add(8, 10, nil) // stride 32 kps
	add(32, 1, func(d []float32) { d[i] = 0.9 })
	add(128, 4, nil) // stride 8 boxes
	add(32, 4, func(d []float32) { d[i*4], d[i*4+1], d[i*4+2], d[i*4+3] = 1, 0.5, 1.5, 2 })
	add(8, 1, nil)
	add(128, 10, nil)
	add(32, 10, func(d []float32) {
		for k := 0; k < 5; k++ {
			d[i*10+2*k], d[i*10+2*k+1] = float32(k)*0.25-0.5, 0.75
		}
	})
	add(128, 1, func(d []float32) { d[5] = 0.2 }) // below threshold
	add(8, 4, nil)
	_ = size
	return &out
}

func TestDecodeSCRFD(t *testing.T) {
	faces := decodeSCRFD(scrfdOutputs(), 64, 0.5, nil)
	if len(faces) != 1 {
		t.Fatalf("got %d faces, want 1", len(faces))
	}
	want := Face{Box: [4]float32{32 - 16, 16 - 8, 32 + 24, 16 + 32}, Score: 0.9}
	for k := range want.Landmarks {
		want.Landmarks[k] = align.Point{X: 32 + (float32(k)*0.25-0.5)*16, Y: 16 + 0.75*16}
	}
	faceNear(t, faces[0], want, 1e-4)

	if got := decodeSCRFD(scrfdOutputs(), 64, 0.1, nil); len(got) != 2 {
		t.Fatalf("with a low threshold got %d faces, want 2", len(got))
	}
	// an output set that is not SCRFD-shaped decodes to nothing rather than panicking
	broken := scrfdOutputs()
	broken.Data, broken.Dims = broken.Data[:5], broken.Dims[:5]
	if got := decodeSCRFD(broken, 64, 0.1, nil); len(got) != 0 {
		t.Fatalf("truncated outputs decoded %d faces", len(got))
	}
}

func yunetOutputsAndIndex() (*ortx.Outputs, map[string]int) {
	var out ortx.Outputs
	index := map[string]int{}
	add := func(name string, n, width int, fill func([]float32)) {
		data := make([]float32, n*width)
		if fill != nil {
			fill(data)
		}
		index[name] = len(out.Data)
		out.Data = append(out.Data, data)
		out.Dims = append(out.Dims, []int64{1, int64(n), int64(width)})
	}
	// stride 8 on a 64 input: 8x8 cells; face at row 3, col 5 -> index 29
	const i = 29
	add("kps_32", 4, 10, nil)
	add("cls_8", 64, 1, func(d []float32) { d[i] = 0.81; d[2] = 1.7 })
	add("obj_8", 64, 1, func(d []float32) { d[i] = 1.0; d[2] = 2.5 })
	add("bbox_8", 64, 4, func(d []float32) {
		d[i*4], d[i*4+1], d[i*4+2], d[i*4+3] = 0.5, 0.25, float32(math.Log(2)), float32(math.Log(4))
	})
	add("kps_8", 64, 10, func(d []float32) {
		for k := 0; k < 5; k++ {
			d[i*10+2*k], d[i*10+2*k+1] = 0.1*float32(k+1), 0.2
		}
	})
	add("cls_16", 16, 1, nil)
	add("obj_16", 16, 1, nil)
	add("bbox_16", 16, 4, nil)
	add("kps_16", 16, 10, nil)
	add("cls_32", 4, 1, nil)
	add("obj_32", 4, 1, nil)
	add("bbox_32", 4, 4, nil)
	return &out, index
}

func TestDecodeYuNet(t *testing.T) {
	out, index := yunetOutputsAndIndex()
	faces := decodeYuNet(out, index, 64, 0.5, nil)
	if len(faces) != 2 {
		t.Fatalf("got %d faces, want 2", len(faces))
	}
	// cell 2 has scores above 1 on both heads: clamped to exactly 1
	if faces[0].Score != 1 {
		t.Fatalf("clamped score %v, want 1", faces[0].Score)
	}
	want := Face{Box: [4]float32{44 - 8, 26 - 16, 44 + 8, 26 + 16}, Score: 0.9}
	for k := range want.Landmarks {
		want.Landmarks[k] = align.Point{X: (0.1*float32(k+1) + 5) * 8, Y: (0.2 + 3) * 8}
	}
	faceNear(t, faces[1], want, 1e-4)
}

func TestNMS(t *testing.T) {
	faces := []Face{
		{Box: [4]float32{0, 0, 10, 10}, Score: 0.6},
		{Box: [4]float32{1, 1, 11, 11}, Score: 0.9},   // same face, better score
		{Box: [4]float32{50, 50, 60, 60}, Score: 0.7}, // elsewhere
		{Box: [4]float32{8, 8, 18, 18}, Score: 0.8},   // touches the first two only slightly
	}
	got := nms(faces, 0.4)
	if len(got) != 3 || got[0].Score != 0.9 || got[1].Score != 0.8 || got[2].Score != 0.7 {
		t.Fatalf("got %+v", got)
	}
	if got := nms(nil, 0.4); len(got) != 0 {
		t.Fatal("nms of nothing must be nothing")
	}
	if o := overlap([4]float32{0, 0, 10, 10}, [4]float32{5, 0, 15, 10}); !near(o, 1.0/3, 1e-6) {
		t.Fatalf("overlap %v, want 1/3", o)
	}
	if o := overlap([4]float32{0, 0, 10, 10}, [4]float32{10, 10, 20, 20}); o != 0 {
		t.Fatalf("touching boxes overlap %v, want 0", o)
	}
	if o := overlap([4]float32{5, 5, 5, 5}, [4]float32{5, 5, 5, 5}); o != 0 {
		t.Fatalf("empty boxes must not divide by zero, got %v", o)
	}
}

func TestToPictureCoords(t *testing.T) {
	f := Face{Box: [4]float32{10, 20, 30, 40}, Score: 1}
	f.Landmarks[2] = align.Point{X: 20, Y: 30}
	m := align.Affine{A: 2, C: 0.5, E: 2, F: 0.5}
	got := toPictureCoords(f, m)
	if got.Box != [4]float32{20.5, 40.5, 60.5, 80.5} || got.Landmarks[2] != (align.Point{X: 40.5, Y: 60.5}) {
		t.Fatalf("got %+v", got)
	}
	if got.Width() != 40 || got.Height() != 40 {
		t.Fatalf("size %v x %v", got.Width(), got.Height())
	}
}
