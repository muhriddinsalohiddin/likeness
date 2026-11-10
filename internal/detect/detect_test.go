package detect

import (
	"math"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/muhriddinsalohiddin/likeness/internal/fixtures"
	"github.com/muhriddinsalohiddin/likeness/internal/models"
	"github.com/muhriddinsalohiddin/likeness/internal/testenv"
	"github.com/muhriddinsalohiddin/likeness/internal/vision"
)

func picture(t *testing.T, name string) *vision.Image {
	t.Helper()
	b, err := fixtures.Picture(name)
	if err != nil {
		t.Fatal(err)
	}
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
	cache  map[string]*Detector
}

func newLoader(t *testing.T) *loader { return &loader{parent: t, cache: map[string]*Detector{}} }

func (l *loader) detector(t *testing.T, name string) *Detector {
	t.Helper()
	if d, ok := l.cache[name]; ok {
		return d
	}
	m, dir := testenv.Model(t, name)
	d, err := New(testenv.Runtime(t), m, dir)
	if err != nil {
		t.Fatal(err)
	}
	l.parent.Cleanup(func() { d.Close() })
	l.cache[name] = d
	return d
}

// The tolerances allow for the differences between OpenCV's fixed-point
// resize and our float sampling: a pixel value can differ by a level or two,
// which moves a regressed coordinate by a fraction of a pixel. Measured on
// the fixtures: 0.35 px, 0.32 px and 0.001.
const (
	boxTolerance      = 1.0 // pixels, in picture coordinates
	landmarkTolerance = 1.0
	scoreTolerance    = 0.02
)

func TestParityWithReferences(t *testing.T) {
	refs, err := fixtures.References()
	if err != nil {
		t.Fatal(err)
	}
	if len(refs) == 0 {
		t.Fatal("no reference results")
	}
	models := newLoader(t)
	var worstBox, worstLm, worstScore float64
	checked := 0
	for _, ref := range refs {
		img := picture(t, ref.Picture)
		for key, want := range ref.Detections {
			name, size := key, want.InputSize
			if base, s, ok := strings.Cut(key, "@"); ok {
				name = base
				if size, err = strconv.Atoi(s); err != nil {
					t.Fatal(err)
				}
			}
			if _, ok := modelsLookup(name); !ok {
				t.Fatalf("%s: reference names unknown model %s", ref.Picture, name)
			}
			t.Run(ref.Picture+"/"+key, func(t *testing.T) {
				d := models.detector(t, name)
				got, err := d.Detect(img, Options{Size: size, NoAntiAlias: true})
				if err != nil {
					t.Fatal(err)
				}
				if len(got) != len(want.Faces) {
					t.Fatalf("found %d faces, reference has %d", len(got), len(want.Faces))
				}
				for _, w := range want.Faces {
					g, iou := closest(got, w.Box)
					if iou < 0.8 {
						t.Fatalf("no match for reference face %v (best overlap %.2f)", w.Box, iou)
					}
					for i := range w.Box {
						worstBox = math.Max(worstBox, math.Abs(float64(g.Box[i]-w.Box[i])))
					}
					for i := range w.Landmarks {
						worstLm = math.Max(worstLm, math.Abs(float64(g.Landmarks[i].X-w.Landmarks[i][0])))
						worstLm = math.Max(worstLm, math.Abs(float64(g.Landmarks[i].Y-w.Landmarks[i][1])))
					}
					worstScore = math.Max(worstScore, math.Abs(float64(g.Score-w.Score)))
					checked++
				}
			})
		}
	}
	t.Logf("%d faces compared; largest deviation: box %.2f px, landmark %.2f px, score %.3f", checked, worstBox, worstLm, worstScore)
	if checked > 0 && (worstBox > boxTolerance || worstLm > landmarkTolerance || worstScore > scoreTolerance) {
		t.Fatal("deviation from the reference implementation exceeds the tolerance")
	}
}

func modelsLookup(name string) (*models.Manifest, bool) { return models.Lookup(name) }

func closest(faces []Face, box [4]float32) (Face, float32) {
	var best Face
	var bestIoU float32
	for _, f := range faces {
		if o := overlap(f.Box, box); o > bestIoU {
			best, bestIoU = f, o
		}
	}
	return best, bestIoU
}

func TestAntiAliasingKeepsDetections(t *testing.T) {
	l := newLoader(t)
	for _, name := range []string{"yunet_2023mar", "det_500m"} {
		d := l.detector(t, name)
		for _, pic := range fixtures.Names() {
			img := picture(t, pic)
			plain, err := d.Detect(img, Options{NoAntiAlias: true})
			if err != nil {
				t.Fatal(err)
			}
			filtered, err := d.Detect(img, Options{})
			if err != nil {
				t.Fatal(err)
			}
			if len(plain) != len(filtered) {
				t.Fatalf("%s/%s: %d faces without the pre-filter, %d with", name, pic, len(plain), len(filtered))
			}
			for _, f := range plain {
				if _, iou := closest(filtered, f.Box); iou < 0.9 {
					t.Fatalf("%s/%s: face %v moved with the pre-filter (overlap %.2f)", name, pic, f.Box, iou)
				}
			}
		}
	}
}

func TestSmallInputIsAllowedForDynamicModels(t *testing.T) {
	l := newLoader(t)
	d := l.detector(t, "det_500m")
	if got := d.InputSize(Options{Size: 333}); got != 320 {
		t.Fatalf("input size %d, want 320", got)
	}
	if got := d.InputSize(Options{}); got != 640 {
		t.Fatalf("default input size %d, want 640", got)
	}
	y := l.detector(t, "yunet_2023mar")
	if got := y.InputSize(Options{Size: 320}); got != 640 {
		t.Fatalf("a fixed-size model must keep its size, got %d", got)
	}
	img := picture(t, "hathaway")
	faces, err := d.Detect(img, Options{Size: 160})
	if err != nil {
		t.Fatal(err)
	}
	if len(faces) != 1 {
		t.Fatalf("at 160 px got %d faces, want 1", len(faces))
	}
}

func TestDetectConcurrently(t *testing.T) {
	d := newLoader(t).detector(t, "yunet_2023mar")
	img := picture(t, "crew11")
	want, err := d.Detect(img, Options{})
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 5; i++ {
				got, err := d.Detect(img, Options{})
				if err != nil {
					errs <- err
					return
				}
				if len(got) != len(want) || got[0] != want[0] {
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

func TestNewRejectsEmbedders(t *testing.T) {
	m, dir := testenv.Model(t, "sface_2021dec")
	if _, err := New(testenv.Runtime(t), m, dir); err == nil {
		t.Fatal("an embedder must not load as a detector")
	}
	if _, err := New(testenv.Runtime(t), &models.Manifest{Name: "x", Kind: models.YuNet, File: "missing.onnx"}, dir); err == nil {
		t.Fatal("a missing file must be an error")
	}
	if _, err := newLoader(t).detector(t, "yunet_2023mar").Detect(nil, Options{}); err == nil {
		t.Fatal("a nil image must be an error")
	}
}
