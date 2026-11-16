package eval

import (
	"math"
	"testing"
)

// histogram counts the given scores.
func histogram(scores ...float32) *Histogram {
	h := &Histogram{}
	for _, s := range scores {
		h.Add(s)
	}
	return h
}

// near reports whether two figures agree to within rounding.
func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestHistogramCounts(t *testing.T) {
	h := histogram(0.1, 0.3, 0.45, 0.55, 0.8, 0.9)
	if h.Count() != 6 || h.Skipped() != 0 {
		t.Fatalf("six scores counted as %d, %d skipped", h.Count(), h.Skipped())
	}
	for _, c := range []struct {
		threshold float64
		atOrAbove uint64
	}{
		{-1, 6}, {0, 6}, {0.1, 6}, {0.101, 5}, {0.2, 5}, {0.3, 5}, {0.301, 4},
		{0.5, 3}, {0.55, 3}, {0.551, 2}, {0.85, 1}, {0.9, 1}, {0.901, 0}, {1, 0},
		// beyond what a score can be, and no threshold at all
		{-3, 6}, {1.2, 0}, {math.Inf(1), 0}, {math.Inf(-1), 6}, {math.NaN(), 0},
		// a threshold is taken to the nearest thousandth
		{0.5504, 3}, {0.5506, 2}, {0.0996, 6},
	} {
		if got := h.AtOrAbove(c.threshold); got != c.atOrAbove {
			t.Errorf("at or above %v: %d scores, want %d", c.threshold, got, c.atOrAbove)
		}
		if got := h.Below(c.threshold); got != 6-c.atOrAbove {
			t.Errorf("below %v: %d scores, want %d", c.threshold, got, 6-c.atOrAbove)
		}
	}
}

func TestHistogramEdges(t *testing.T) {
	// A float32 cannot hold 0.7, 0.363 or 0.999 exactly and falls a hair
	// short of some of them; the score still counts as reaching a
	// threshold of the same name, as it does in the server.
	for _, edge := range []float32{-0.999, -0.5, 0, 0.001, 0.363, 0.6, 0.7, 0.999} {
		h := histogram(edge)
		if h.AtOrAbove(float64(edge)) != 1 {
			t.Errorf("a score of %v is not at or above the threshold %v", edge, edge)
		}
		if h.AtOrAbove(float64(edge)+0.001) != 0 {
			t.Errorf("a score of %v is at or above the next threshold up", edge)
		}
	}
	// just under an edge is under it
	if h := histogram(0.6999, 0.69999); h.AtOrAbove(0.7) != 0 || h.AtOrAbove(0.699) != 2 {
		t.Errorf("scores just under 0.7: %d at or above 0.7, %d at or above 0.699", h.AtOrAbove(0.7), h.AtOrAbove(0.699))
	}
}

func TestHistogramClampsAndSkips(t *testing.T) {
	inf := float32(math.Inf(1))
	h := histogram(-1, 1, -1.0000001, 1.0000001, -7, 7, -inf, inf)
	if h.Count() != 8 {
		t.Fatalf("counted %d of 8 scores", h.Count())
	}
	if got := h.AtOrAbove(1); got != 4 {
		t.Errorf("%d scores at 1, want 4", got)
	}
	if got := h.Below(-0.999); got != 4 {
		t.Errorf("%d scores in the first bin, want 4", got)
	}
	if h.counts[0] != 4 || h.counts[bins-1] != 4 {
		t.Errorf("the first and last bins hold %d and %d, want 4 and 4", h.counts[0], h.counts[bins-1])
	}
	if !near(h.Mean(), 0) || !near(h.StdDev(), 1) {
		t.Errorf("mean %v and deviation %v, want 0 and 1: what is out of range counts as -1 or 1", h.Mean(), h.StdDev())
	}

	nan := float32(math.NaN())
	h = histogram(0.5, nan, nan, 0.7, nan)
	if h.Count() != 2 || h.Skipped() != 3 {
		t.Errorf("two scores and three NaNs: counted %d, skipped %d", h.Count(), h.Skipped())
	}
	if !near(h.Mean(), (float64(float32(0.5))+float64(float32(0.7)))/2) {
		t.Errorf("mean %v: a NaN got into it", h.Mean())
	}
}

func TestHistogramMoments(t *testing.T) {
	var empty Histogram
	if empty.Count() != 0 || empty.Mean() != 0 || empty.StdDev() != 0 || empty.AtOrAbove(-1) != 0 || empty.Below(1) != 0 {
		t.Errorf("an empty histogram: %d scores, mean %v, deviation %v", empty.Count(), empty.Mean(), empty.StdDev())
	}

	h := histogram(0.25, 0.5, 0.75)
	if !near(h.Mean(), 0.5) {
		t.Errorf("mean %v, want 0.5", h.Mean())
	}
	if want := math.Sqrt(2 * 0.25 * 0.25 / 3); !near(h.StdDev(), want) {
		t.Errorf("deviation %v, want %v", h.StdDev(), want)
	}

	// all scores the same: no spread, and rounding must not make it negative
	h = histogram(0.3, 0.3, 0.3, 0.3)
	if !near(h.Mean(), float64(float32(0.3))) || h.StdDev() > 1e-6 || math.IsNaN(h.StdDev()) {
		t.Errorf("four equal scores: mean %v, deviation %v", h.Mean(), h.StdDev())
	}
}

func TestHistogramMerge(t *testing.T) {
	nan := float32(math.NaN())
	a, b := histogram(0.25, 0.5, nan), histogram(0.5, 0.75, 1, nan, nan)
	a.Merge(b)
	if *a != *histogram(0.25, 0.5, 0.5, 0.75, 1, nan, nan, nan) {
		t.Errorf("merged: %d scores, %d skipped, mean %v", a.Count(), a.Skipped(), a.Mean())
	}
	before := *a
	a.Merge(&Histogram{})
	if *a != before {
		t.Error("merging an empty histogram changed something")
	}
}

func TestEdges(t *testing.T) {
	for _, c := range []struct {
		threshold float64
		edge      int
	}{
		{-1, 0}, {-0.999, 1}, {0, 1000}, {0.363, 1363}, {0.7, 1700}, {0.999, 1999}, {1, 2000},
		{1.0004, 2000}, {1.0006, bins}, {2, bins}, {-1.5, 0},
	} {
		if got := edgeOf(c.threshold); got != c.edge {
			t.Errorf("the threshold %v starts at bin %d, want %d", c.threshold, got, c.edge)
		}
	}
	for edge := 0; edge < bins; edge++ {
		if back := edgeOf(thresholdOf(edge)); back != edge {
			t.Fatalf("bin %d starts at %v, which is taken for bin %d", edge, thresholdOf(edge), back)
		}
	}
}
