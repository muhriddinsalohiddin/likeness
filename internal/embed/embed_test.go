package embed

import (
	"math"
	"sync"
	"testing"

	"github.com/muhriddinsalohiddin/likeness/internal/align"
	"github.com/muhriddinsalohiddin/likeness/internal/fixtures"
	"github.com/muhriddinsalohiddin/likeness/internal/models"
	"github.com/muhriddinsalohiddin/likeness/internal/testenv"
	"github.com/muhriddinsalohiddin/likeness/internal/vision"
)

func picture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := fixtures.Picture(name)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func decode(t *testing.T, b []byte) *vision.Image {
	t.Helper()
	img, err := vision.Decode(b, vision.Limits{})
	if err != nil {
		t.Fatal(err)
	}
	return img
}

// loader opens each model once per top-level test; a subtest that asks for
// an uninstalled model is skipped on its own.
type loader struct {
	parent *testing.T
	cache  map[string]*Embedder
}

func newLoader(t *testing.T) *loader { return &loader{parent: t, cache: map[string]*Embedder{}} }

func (l *loader) embedder(t *testing.T, name string) *Embedder {
	t.Helper()
	if e, ok := l.cache[name]; ok {
		return e
	}
	m, dir := testenv.Model(t, name)
	e, err := New(testenv.Runtime(t), m, dir)
	if err != nil {
		t.Fatal(err)
	}
	l.parent.Cleanup(func() { e.Close() })
	l.cache[name] = e
	return e
}

func points(lm [5][2]float32) [5]align.Point {
	var p [5]align.Point
	for i := range lm {
		p[i] = align.Point{X: lm[i][0], Y: lm[i][1]}
	}
	return p
}

func cosine(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

// parityCosine is the least similarity accepted between our vector and the
// reference implementation's for the same landmarks: the crops differ only by
// interpolation rounding. Measured on the fixtures: 0.99985.
const parityCosine = 0.998

func TestParityWithReferences(t *testing.T) {
	refs, err := fixtures.References()
	if err != nil {
		t.Fatal(err)
	}
	l := newLoader(t)
	worst := 1.0
	checked := 0
	for _, ref := range refs {
		img := decode(t, picture(t, ref.Picture))
		for name, faces := range ref.Embeddings {
			t.Run(ref.Picture+"/"+name, func(t *testing.T) {
				e := l.embedder(t, name)
				for _, f := range faces {
					if len(f.Vector) != e.Dim() {
						t.Fatalf("reference vector has %d values, model emits %d", len(f.Vector), e.Dim())
					}
					got := make([]float32, e.Dim())
					if err := e.Embed(img, points(f.Landmarks), got, Options{NoAntiAlias: true}); err != nil {
						t.Fatal(err)
					}
					c := cosine(got, f.Vector)
					worst = math.Min(worst, c)
					checked++
					if c < parityCosine {
						t.Errorf("cosine to the reference is %.4f", c)
					}
				}
			})
		}
	}
	t.Logf("%d vectors compared; lowest cosine to the reference %.5f", checked, worst)
}

func TestVectorsAreUnitLengthAndDiscriminative(t *testing.T) {
	refs, err := fixtures.References()
	if err != nil {
		t.Fatal(err)
	}
	l := newLoader(t)
	for _, name := range []string{"sface_2021dec", "w600k_mbf"} {
		e := l.embedder(t, name)
		var vectors [][]float32
		for _, ref := range refs {
			faces := ref.Embeddings[name]
			if len(faces) == 0 {
				continue
			}
			img := decode(t, picture(t, ref.Picture))
			for _, f := range faces {
				v := make([]float32, e.Dim())
				if err := e.Embed(img, points(f.Landmarks), v, Options{}); err != nil {
					t.Fatal(err)
				}
				if n := cosine(v, v); math.Abs(n-1) > 1e-4 {
					t.Fatalf("%s: vector norm %.5f", name, n)
				}
				vectors = append(vectors, v)
			}
		}
		if len(vectors) < 4 {
			t.Fatalf("%s: only %d faces", name, len(vectors))
		}
		// seven different people: no pair should look alike
		for i := range vectors {
			for j := i + 1; j < len(vectors); j++ {
				if c := cosine(vectors[i], vectors[j]); c > 0.5 {
					t.Errorf("%s: faces %d and %d have cosine %.3f", name, i, j, c)
				}
			}
		}
	}
}

// TestJPEGParity runs the production decode path (JPEG, planar YCbCr with
// bilinear chroma) against OpenCV's decode of the very same bytes. The
// fixtures also show that SFace itself is sensitive to compression on small
// faces: OpenCV's own PNG-versus-JPEG cosines range from 0.93 (60 px faces)
// to 0.99 (200 px faces), which is a reason to insist on face size and
// picture quality at enrolment rather than a decoding problem.
func TestJPEGParity(t *testing.T) {
	refs, err := fixtures.References()
	if err != nil {
		t.Fatal(err)
	}
	l := newLoader(t)
	worst := 1.0
	checked := 0
	for _, ref := range refs {
		for name, faces := range ref.JPEGEmbeddings {
			t.Run(ref.Picture+"/"+name, func(t *testing.T) {
				e := l.embedder(t, name)
				jpg, err := fixtures.JPEG(ref.Picture)
				if err != nil {
					t.Fatal(err)
				}
				img := decode(t, jpg)
				for _, f := range faces {
					got := make([]float32, e.Dim())
					if err := e.Embed(img, points(f.Landmarks), got, Options{NoAntiAlias: true}); err != nil {
						t.Fatal(err)
					}
					c := cosine(got, f.Vector)
					worst = math.Min(worst, c)
					checked++
					if c < parityCosine {
						t.Errorf("cosine to the reference is %.4f", c)
					}
				}
			})
		}
	}
	if checked == 0 {
		t.Skip("no JPEG references")
	}
	t.Logf("%d JPEG vectors compared; lowest cosine to the reference %.5f", checked, worst)
}

func TestEmbedConcurrently(t *testing.T) {
	refs, err := fixtures.References()
	if err != nil {
		t.Fatal(err)
	}
	e := newLoader(t).embedder(t, "sface_2021dec")
	var ref fixtures.Reference
	for _, r := range refs {
		if len(r.Embeddings["sface_2021dec"]) == 1 {
			ref = r
			break
		}
	}
	img := decode(t, picture(t, ref.Picture))
	lm := points(ref.Embeddings["sface_2021dec"][0].Landmarks)
	want := make([]float32, e.Dim())
	if err := e.Embed(img, lm, want, Options{}); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got := make([]float32, e.Dim())
			for i := 0; i < 10; i++ {
				if err := e.Embed(img, lm, got, Options{}); err != nil {
					errs <- err
					return
				}
				if cosine(got, want) < 0.9999 {
					errs <- err
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	if err := <-errs; err != nil {
		t.Fatal(err)
	}
}

func TestEmbedErrors(t *testing.T) {
	e := newLoader(t).embedder(t, "sface_2021dec")
	img := decode(t, picture(t, "hathaway"))
	var lm [5]align.Point
	if err := e.Embed(img, lm, make([]float32, e.Dim()), Options{}); err == nil {
		t.Fatal("coincident landmarks must be rejected")
	}
	lm = points([5][2]float32{{100, 100}, {160, 100}, {130, 140}, {105, 170}, {155, 170}})
	if err := e.Embed(img, lm, make([]float32, 3), Options{}); err == nil {
		t.Fatal("a destination of the wrong length must be rejected")
	}
	if err := e.Embed(nil, lm, make([]float32, e.Dim()), Options{}); err == nil {
		t.Fatal("a nil image must be rejected")
	}
	if e.Fingerprint() != e.Manifest().Fingerprint() || e.Dim() != e.Manifest().Embedding {
		t.Fatal("fingerprint or dimension do not match the manifest")
	}
	m, dir := testenv.Model(t, "yunet_2023mar")
	if _, err := New(testenv.Runtime(t), m, dir); err == nil {
		t.Fatal("a detector must not load as an embedder")
	}
	if _, err := New(testenv.Runtime(t), &models.Manifest{Name: "x", Kind: models.SFace, File: "missing.onnx", Input: 112, Embedding: 128}, dir); err == nil {
		t.Fatal("a missing file must be an error")
	}
}

func TestNormalize(t *testing.T) {
	dst := make([]float32, 3)
	Normalize(dst, []float32{3, 0, 4})
	if dst[0] != 0.6 || dst[1] != 0 || dst[2] != 0.8 {
		t.Fatalf("got %v", dst)
	}
	Normalize(dst, []float32{0, 0, 0})
	if dst[0] != 0 || dst[2] != 0 {
		t.Fatalf("zero vector became %v", dst)
	}
}
