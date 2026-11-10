package likeness

import (
	"context"
	"errors"
	"image"
	"math"
	"sync"
	"testing"

	"github.com/muhriddinsalohiddin/likeness/internal/fixtures"
)

// anyReason marks an outcome the two detectors disagree on.
const anyReason Reason = "*"

func TestCloseupsAreAccepted(t *testing.T) {
	detectSize := map[string]int{"default": 640, "buffalo_s": 320}
	for _, preset := range presets {
		t.Run(preset, func(t *testing.T) {
			e := open(t, preset)
			for _, name := range fixtures.Closeups() {
				pic, err := fixtures.Closeup(name)
				if err != nil {
					t.Fatal(err)
				}
				res := analyze(t, e, pic, ForEnrolment())
				f, reason := res.Best()
				if reason != Accepted {
					t.Fatalf("%s: turned down as %q (%+v)", name, reason, f)
				}
				if res.Width != 448 || res.Height != 448 || res.Found != 1 || len(res.Faces) != 1 || res.DetectSize != detectSize[preset] {
					t.Errorf("%s: %dx%d, found %d, %d faces, detector input %d", name, res.Width, res.Height, res.Found, len(res.Faces), res.DetectSize)
				}
				if len(f.Embedding) != e.Info().Dim {
					t.Fatalf("%s: vector of %d values, engine says %d", name, len(f.Embedding), e.Info().Dim)
				}
				if n := cosine(f.Embedding, f.Embedding); math.Abs(n-1) > 1e-4 {
					t.Errorf("%s: vector norm %.5f", name, n)
				}
				q := f.Quality
				if q.EyeDistance < 55 || q.EyeDistance > 85 || q.InFrame != 1 || q.Brightness < 100 || q.Brightness > 180 || q.Sharpness < 150 || q.Score < 0.8 {
					t.Errorf("%s: implausible quality %+v", name, q)
				}
				if math.Abs(float64(q.Yaw)) > 20 || math.Abs(float64(q.Roll)) > 5 {
					t.Errorf("%s: a frontal portrait reads as yaw %.1f roll %.1f", name, q.Yaw, q.Roll)
				}
				// the box holds the landmarks
				for i, p := range f.Landmarks {
					if p.X < f.Box.Left || p.X > f.Box.Right || p.Y < f.Box.Top || p.Y > f.Box.Bottom {
						t.Errorf("%s: landmark %d at %v lies outside the box %v", name, i, p, f.Box)
					}
				}
				tm := res.Timing
				if tm.Decode <= 0 || tm.Detect <= 0 || tm.Align <= 0 || tm.Embed <= 0 || tm.Total < tm.Decode+tm.Detect+tm.Align+tm.Embed {
					t.Errorf("%s: timings do not add up: %+v", name, tm)
				}
			}
		})
	}
}

func TestPeopleAreToldApart(t *testing.T) {
	for _, preset := range presets {
		t.Run(preset, func(t *testing.T) {
			e := open(t, preset)
			vectors := map[string][]float32{}
			for _, name := range fixtures.Closeups() {
				pic, _ := fixtures.Closeup(name)
				f, reason := analyze(t, e, pic, ForSearch()).Best()
				if reason != Accepted {
					t.Fatalf("%s: %q", name, reason)
				}
				vectors[name] = f.Embedding
			}
			if c := cosine(vectors["hathaway"], vectors["cassada"]); c > 0.3 {
				t.Errorf("two different people have a similarity of %.3f", c)
			}
			// The same photograph at a third of the resolution, in a wider
			// frame, through another codec: still the same person.
			for name, vec := range vectors {
				wide, _ := fixtures.Picture(name)
				f, reason := analyze(t, e, wide, ForSearch()).Best()
				if reason != Accepted {
					t.Fatalf("%s wide: %q", name, reason)
				}
				c := cosine(vec, f.Embedding)
				t.Logf("%s: close-up against the wide shot %.3f", name, c)
				if c < 0.85 {
					t.Errorf("%s: close-up and wide shot have a similarity of only %.3f", name, c)
				}
			}
		})
	}
}

func TestQualityGates(t *testing.T) {
	type picture func(src *image.RGBA) image.Image
	cut := func(percent int) picture {
		return func(src *image.RGBA) image.Image {
			b := src.Bounds()
			return src.SubImage(image.Rect(0, 0, b.Dx()*percent/100, b.Dy()))
		}
	}
	cases := []struct {
		name          string
		make          picture
		enrol, search Reason
		// broken: the picture is deliberately unusable; if a policy lets it
		// through anyway, nothing is promised about its vector
		broken bool
	}{
		{"clean", func(s *image.RGBA) image.Image { return s }, Accepted, Accepted, false},
		{"slight blur", func(s *image.RGBA) image.Image { return blur(s, 1) }, Accepted, Accepted, false},
		{"soft focus", func(s *image.RGBA) image.Image { return blur(s, 2) }, Blurry, Accepted, false},
		{"blur", func(s *image.RGBA) image.Image { return blur(s, 3) }, Blurry, Accepted, false},
		{"heavy blur", func(s *image.RGBA) image.Image { return blur(s, 6) }, Blurry, Blurry, false},
		{"dim room", func(s *image.RGBA) image.Image { return expose(s, 0.25, 0) }, TooDark, Accepted, false},
		{"nearly black", func(s *image.RGBA) image.Image { return expose(s, 0.07, 0) }, TooDark, TooDark, false},
		{"bright", func(s *image.RGBA) image.Image { return expose(s, 1.5, 30) }, TooBright, Accepted, false},
		{"washed out", func(s *image.RGBA) image.Image { return expose(s, 2.2, 60) }, TooBright, TooBright, false},
		{"half the size", func(s *image.RGBA) image.Image { return scale(s, 0.5) }, FaceTooSmall, Accepted, false},
		{"a fifth of the size", func(s *image.RGBA) image.Image { return scale(s, 0.2) }, FaceTooSmall, FaceTooSmall, false},
		// A face cut in half: enrolment refuses it. Search may not notice,
		// because a detector boxes only the half it sees and the box then
		// lies almost entirely inside the picture.
		{"cut by the edge", cut(52), OutOfFrame, anyReason, true},
	}
	for _, preset := range presets {
		t.Run(preset, func(t *testing.T) {
			e := open(t, preset)
			src := portrait(t)
			clean, _ := analyze(t, e, encodePNG(t, src), Options{}).Best()
			for _, c := range cases {
				pic := encodePNG(t, c.make(src))
				for _, run := range []struct {
					policy     string
					options    Options
					want       Reason
					similarity float64 // least similarity to the clean vector when accepted
				}{
					{"enrolment", ForEnrolment(), c.enrol, 0.97},
					{"search", ForSearch(), c.search, 0.85},
				} {
					f, got := analyze(t, e, pic, run.options).Best()
					if f == nil {
						t.Errorf("%s, %s: no face at all (%q)", c.name, run.policy, got)
						continue
					}
					if run.want != anyReason && got != run.want {
						t.Errorf("%s, %s: got %q, want %q (%+v)", c.name, run.policy, got, run.want, f.Quality)
					}
					if (got == Accepted) != (f.Embedding != nil) {
						t.Errorf("%s, %s: reason %q but embedding present=%v", c.name, run.policy, got, f.Embedding != nil)
					}
					if got == Accepted && !c.broken {
						if sim := cosine(clean.Embedding, f.Embedding); sim < run.similarity {
							t.Errorf("%s, %s: accepted, yet only %.3f similar to the clean picture (want %.2f)", c.name, run.policy, sim, run.similarity)
						}
					}
				}
			}
		})
	}
}

func TestProfileIsBadPose(t *testing.T) {
	profile, err := fixtures.Picture("adenot") // a head seen from the side
	if err != nil {
		t.Fatal(err)
	}
	for _, preset := range presets {
		t.Run(preset, func(t *testing.T) {
			e := open(t, preset)
			f, reason := analyze(t, e, profile, Options{Policy: Policy{MaxYaw: 55}}).Best()
			if f == nil || reason != BadPose || f.Quality.Yaw > -55 || f.Embedding != nil {
				t.Fatalf("got %q, face %+v", reason, f)
			}
			// with no policy the same face is embedded, for what it is worth
			if f, reason := analyze(t, e, profile, Options{}).Best(); reason != Accepted || f.Embedding == nil {
				t.Fatalf("the zero policy turned a face down: %q", reason)
			}
		})
	}
}

func TestLeaningHeadIsBadPoseAtEnrolment(t *testing.T) {
	// Only detectors whose landmarks follow the lean can tell. YuNet's do
	// not beyond some twenty degrees, so the default preset is not tested.
	e := open(t, "buffalo_s")
	src := portrait(t)
	f, reason := analyze(t, e, encodePNG(t, rotate(src, 40)), ForEnrolment()).Best()
	if f == nil || reason != BadPose || f.Quality.Roll < 30 || f.Quality.Roll > 50 {
		t.Fatalf("got %q, face %+v", reason, f)
	}
	// alignment undoes the lean, so recognition still works
	upright, _ := analyze(t, e, encodePNG(t, src), ForSearch()).Best()
	leaning, reason := analyze(t, e, encodePNG(t, rotate(src, 40)), ForSearch()).Best()
	if reason != Accepted {
		t.Fatalf("search turned the leaning head down: %q", reason)
	}
	if sim := cosine(upright.Embedding, leaning.Embedding); sim < 0.85 {
		t.Fatalf("leaning by 40 degrees leaves a similarity of %.3f", sim)
	}
}

func TestSelection(t *testing.T) {
	group, err := fixtures.Picture("crew11") // four people
	if err != nil {
		t.Fatal(err)
	}
	for _, preset := range presets {
		t.Run(preset, func(t *testing.T) {
			e := open(t, preset)

			all := analyze(t, e, group, Options{Faces: All})
			if all.Found != 4 || len(all.Faces) != 4 || all.Reject != Accepted {
				t.Fatalf("found %d, analysed %d, reject %q", all.Found, len(all.Faces), all.Reject)
			}
			for i, f := range all.Faces {
				if f.Reject != Accepted || f.Embedding == nil {
					t.Errorf("face %d: %q", i, f.Reject)
				}
				if i > 0 && f.Box.Width()*f.Box.Height() > all.Faces[i-1].Box.Width()*all.Faces[i-1].Box.Height()*1.0001 {
					t.Errorf("face %d is larger than face %d", i, i-1)
				}
				for j := 0; j < i; j++ {
					if c := cosine(f.Embedding, all.Faces[j].Embedding); c > 0.5 {
						t.Errorf("faces %d and %d of different people are %.2f similar", i, j, c)
					}
				}
			}

			if two := analyze(t, e, group, Options{Faces: All, MaxFaces: 2}); len(two.Faces) != 2 || two.Found != 4 {
				t.Errorf("MaxFaces 2: %d faces, found %d", len(two.Faces), two.Found)
			}

			largest := analyze(t, e, group, Options{})
			if len(largest.Faces) != 1 || largest.Found != 4 || largest.Faces[0].Box != all.Faces[0].Box {
				t.Errorf("Largest: %d faces, found %d", len(largest.Faces), largest.Found)
			}

			single := analyze(t, e, group, Options{Faces: Single, Policy: Policy{MaxRival: 0.5}})
			f, reason := single.Best()
			if single.Reject != MultipleFaces || reason != MultipleFaces || f == nil || f.Embedding != nil {
				t.Errorf("Single: reject %q, best %q", single.Reject, reason)
			}

			// faces this small are of no use to either policy
			for _, o := range []Options{ForSearch(), ForEnrolment()} {
				o.Faces = All
				res := analyze(t, e, group, o)
				for i, f := range res.Faces {
					if f.Reject != FaceTooSmall || f.Embedding != nil || f.Quality.Sharpness != 0 {
						t.Errorf("face %d: %q, quality %+v", i, f.Reject, f.Quality)
					}
				}
			}
		})
	}
}

func TestOnePersonIsNotRivalled(t *testing.T) {
	for _, preset := range presets {
		t.Run(preset, func(t *testing.T) {
			e := open(t, preset)
			for _, name := range fixtures.Names() {
				if name == "crew11" {
					continue
				}
				pic, _ := fixtures.Picture(name)
				// the wide shots show helmets and patches around one person
				if res := analyze(t, e, pic, Options{Faces: Single, Policy: Policy{MaxRival: 0.5}}); res.Reject != Accepted {
					t.Errorf("%s: %q with %d detections", name, res.Reject, res.Found)
				}
			}
		})
	}
}

func TestSmallFaceInALargeFrame(t *testing.T) {
	e := open(t, "buffalo_s") // its detector accepts any input size
	src := portrait(t)

	near := analyze(t, e, encodePNG(t, src), ForSearch())
	if near.DetectSize != 320 {
		t.Fatalf("a close-up should be found at the small input, used %d", near.DetectSize)
	}
	far := analyze(t, e, encodePNG(t, onCanvas(src, 0.5, 1600, 1200)), ForSearch())
	if far.DetectSize != 640 {
		t.Fatalf("a small face in a large frame should fall back to the large input, used %d", far.DetectSize)
	}
	a, _ := near.Best()
	b, reason := far.Best()
	if reason != Accepted {
		t.Fatalf("far: %q", reason)
	}
	if b.Quality.EyeDistance < 30 || b.Quality.EyeDistance > 40 {
		t.Errorf("eye distance %.1f, want about 35", b.Quality.EyeDistance)
	}
	if sim := cosine(a.Embedding, b.Embedding); sim < 0.9 {
		t.Errorf("near and far have a similarity of %.3f", sim)
	}
}

func TestNoFace(t *testing.T) {
	for _, preset := range presets {
		t.Run(preset, func(t *testing.T) {
			e := open(t, preset)
			res := analyze(t, e, encodePNG(t, image.NewRGBA(image.Rect(0, 0, 320, 240))), ForSearch())
			f, reason := res.Best()
			if res.Reject != NoFace || reason != NoFace || f != nil || len(res.Faces) != 0 || res.Found != 0 {
				t.Fatalf("reject %q, best %q, %d faces", res.Reject, reason, len(res.Faces))
			}
			if res.Width != 320 || res.Height != 240 || res.Timing.Total <= 0 || res.Timing.Embed != 0 {
				t.Fatalf("%+v", res)
			}
		})
	}
}

func TestEngineAgreesWithReferences(t *testing.T) {
	refs, err := fixtures.References()
	if err != nil {
		t.Fatal(err)
	}
	for _, preset := range presets {
		t.Run(preset, func(t *testing.T) {
			e := open(t, preset)
			worst := 1.0
			for _, ref := range refs {
				pic, _ := fixtures.Picture(ref.Picture)
				res := analyze(t, e, pic, Options{Faces: All})
				want := ref.Embeddings[e.Info().Embedder]
				if len(want) != len(res.Faces) {
					t.Fatalf("%s: %d faces, reference has %d", ref.Picture, len(res.Faces), len(want))
				}
				for _, w := range want {
					// pair by the position of the left eye
					best, dist := -1, math.Inf(1)
					for i, f := range res.Faces {
						d := math.Hypot(float64(f.Landmarks[0].X-w.Landmarks[0][0]), float64(f.Landmarks[0].Y-w.Landmarks[0][1]))
						if d < dist {
							best, dist = i, d
						}
					}
					// the engine may run the detector at another input size
					// than the reference did, which moves landmarks a little
					if dist > 6 {
						t.Fatalf("%s: no face near the reference's (nearest %.1f px away)", ref.Picture, dist)
					}
					// A profile is located too loosely for two detector passes
					// to agree on it; no policy would embed it either.
					if math.Abs(float64(res.Faces[best].Quality.Yaw)) > 45 {
						continue
					}
					worst = math.Min(worst, cosine(res.Faces[best].Embedding, w.Vector))
				}
			}
			t.Logf("lowest similarity between the engine's vectors and the reference implementation's: %.4f", worst)
			// 0.9999 when the engine detects at the reference's input size,
			// about 0.98 when it finds a small face at half that size
			if worst < 0.97 {
				t.Error("the engine drifts from the reference implementation")
			}
		})
	}
}

func TestBadPictures(t *testing.T) {
	e := open(t, "default")
	pic, _ := fixtures.Closeup("hathaway")
	for name, c := range map[string]struct {
		data []byte
		want error
	}{
		"empty":     {nil, ErrUnsupportedImage},
		"text":      {[]byte("not a picture at all"), ErrUnsupportedImage},
		"truncated": {pic[:len(pic)/3], ErrCorruptImage},
		"tiny":      {encodePNG(t, image.NewRGBA(image.Rect(0, 0, 8, 8))), ErrImageTooSmall},
	} {
		res, err := e.Analyze(context.Background(), c.data, ForSearch())
		if !errors.Is(err, c.want) || res != nil {
			t.Errorf("%s: got %v, want %v", name, err, c.want)
		}
	}

	// a second engine with a tighter limit shares the first one's runtime
	strict, err := Open(Config{Threads: testThreads, MaxPixels: 100_000})
	if err != nil {
		t.Fatal(err)
	}
	defer strict.Close()
	if _, err := strict.Analyze(context.Background(), pic, ForSearch()); !errors.Is(err, ErrImageTooLarge) {
		t.Errorf("448x448 against a 100 000 pixel limit: %v", err)
	}
	if _, err := e.Analyze(context.Background(), pic, ForSearch()); err != nil {
		t.Errorf("the first engine is unaffected by the second one's limit: %v", err)
	}
}

func TestCancelledCall(t *testing.T) {
	e := open(t, "default")
	pic, _ := fixtures.Closeup("hathaway")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if res, err := e.Analyze(ctx, pic, ForSearch()); !errors.Is(err, context.Canceled) || res != nil {
		t.Fatalf("got %v", err)
	}
}

func TestConcurrentCalls(t *testing.T) {
	for _, preset := range presets {
		t.Run(preset, func(t *testing.T) {
			e := open(t, preset)
			pics := map[string][]byte{}
			want := map[string][]float32{}
			for _, name := range fixtures.Closeups() {
				pics[name], _ = fixtures.Closeup(name)
				f, _ := analyze(t, e, pics[name], ForSearch()).Best()
				want[name] = f.Embedding
			}
			var wg sync.WaitGroup
			errs := make(chan string, 16)
			for w := 0; w < 8; w++ {
				wg.Add(1)
				go func(w int) {
					defer wg.Done()
					name := fixtures.Closeups()[w%2]
					for i := 0; i < 6; i++ {
						res, err := e.Analyze(context.Background(), pics[name], ForSearch())
						if err != nil {
							errs <- err.Error()
							return
						}
						f, reason := res.Best()
						if reason != Accepted || cosine(f.Embedding, want[name]) < 0.9999 {
							errs <- "a concurrent call gave a different answer for " + name
							return
						}
					}
				}(w)
			}
			wg.Wait()
			close(errs)
			for msg := range errs {
				t.Error(msg)
			}
		})
	}
}
