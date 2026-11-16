package likeness

import (
	"context"
	"errors"
	"os"
	"testing"

	"github.com/muhriddinsalohiddin/likeness/internal/fixtures"
	"github.com/muhriddinsalohiddin/likeness/internal/home"
	"github.com/muhriddinsalohiddin/likeness/internal/models"
	"github.com/muhriddinsalohiddin/likeness/internal/testenv"
)

// openLive is open with the liveness models loaded as well.
func openLive(t testing.TB, preset string) *Engine {
	t.Helper()
	if _, err := os.Stat(home.LibraryPath()); err != nil {
		t.Skipf("onnxruntime is not installed: %v", err)
	}
	for _, name := range presetModels[preset] {
		testenv.Model(t, name)
	}
	for _, m := range models.Liveness {
		testenv.Model(t, m.Name)
	}
	e, err := Open(Config{Preset: preset, Threads: testThreads, Liveness: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(e.Close)
	return e
}

func TestLiveness(t *testing.T) {
	picture, err := fixtures.Closeup("hathaway")
	if err != nil {
		t.Fatal(err)
	}
	for _, preset := range presets {
		t.Run(preset, func(t *testing.T) {
			e := openLive(t, preset)
			if got := e.Info().Liveness; len(got) != 2 || got[0] != "minifasnet_v2" {
				t.Errorf("liveness models %v", got)
			}

			// a portrait photographed for print is, to these models, a face
			res := analyze(t, e, picture, ForSearch())
			face, reason := res.Best()
			if reason != Accepted || face.Liveness < 0.8 || face.Embedding == nil || res.Timing.Live <= 0 {
				t.Fatalf("liveness %v, %s, judged in %v", face.Liveness, reason, res.Timing.Live)
			}

			// a floor it cannot reach turns it down before it is embedded
			o := ForSearch()
			o.MinLiveness = 0.9999
			res = analyze(t, e, picture, o)
			if face, reason := res.Best(); reason != Spoof || face.Embedding != nil || face.Liveness <= 0 || res.Timing.Embed != 0 {
				t.Errorf("under an impossible floor: %s, liveness %v", reason, face.Liveness)
			}
			o.MinLiveness = 0.5
			if _, reason := analyze(t, e, picture, o).Best(); reason != Accepted {
				t.Errorf("under a floor of 0.5: %s", reason)
			}
		})
	}
}

func TestLivenessHasToBeAskedFor(t *testing.T) {
	e := open(t, "default")
	if got := e.Info().Liveness; len(got) != 0 {
		t.Errorf("an engine opened without liveness lists %v", got)
	}
	picture, err := fixtures.Closeup("hathaway")
	if err != nil {
		t.Fatal(err)
	}
	if face, _ := analyze(t, e, picture, ForSearch()).Best(); face.Liveness != 0 {
		t.Errorf("liveness %v from an engine that does not judge it", face.Liveness)
	}
	o := ForSearch()
	o.MinLiveness = 0.5
	if _, err := e.Analyze(context.Background(), picture, o); !errors.Is(err, ErrNoLiveness) {
		t.Errorf("a floor with nothing to judge by: %v", err)
	}
}

func TestOpenNeedsTheLivenessModelsInstalled(t *testing.T) {
	if _, err := os.Stat(home.LibraryPath()); err != nil {
		t.Skipf("onnxruntime is not installed: %v", err)
	}
	// a home with the library and nothing else: the detector is missed
	// first either way, so what is checked is that asking for liveness
	// does not get past a missing model
	dir := t.TempDir()
	if _, err := Open(Config{Home: dir, Library: home.LibraryPath(), Liveness: true}); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("an empty home: %v", err)
	}
}
