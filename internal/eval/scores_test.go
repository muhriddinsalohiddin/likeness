package eval

import (
	"math"
	"testing"
)

// scored makes the two distributions out of the scores given.
func scored(genuine, impostor []float32) *Scores {
	s := &Scores{}
	for _, v := range genuine {
		s.Genuine.Add(v)
	}
	for _, v := range impostor {
		s.Impostor.Add(v)
	}
	return s
}

// repeat returns n copies of a score.
func repeat(score float32, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = score
	}
	return out
}

func TestErrorRates(t *testing.T) {
	// Six comparisons: three of the same person, three of strangers, one
	// of each on the wrong side of 0.5.
	s := scored([]float32{0.45, 0.8, 0.9}, []float32{0.1, 0.3, 0.55})
	third := 1.0 / 3
	for _, c := range []struct {
		threshold float64
		fmr, fnmr float64
	}{
		{-1, 1, 0},
		{0.1, 1, 0},
		{0.2, 2 * third, 0},
		{0.3, 2 * third, 0},
		{0.4, third, 0},
		{0.45, third, 0},
		{0.5, third, third},
		{0.55, third, third},
		{0.6, 0, third},
		{0.8, 0, third},
		{0.85, 0, 2 * third},
		{0.95, 0, 1},
		{1, 0, 1},
	} {
		if got := s.FMR(c.threshold); !near(got, c.fmr) {
			t.Errorf("FMR at %v: %v, want %v", c.threshold, got, c.fmr)
		}
		if got := s.FNMR(c.threshold); !near(got, c.fnmr) {
			t.Errorf("FNMR at %v: %v, want %v", c.threshold, got, c.fnmr)
		}
	}

	var empty Scores
	if empty.FMR(0.5) != 0 || empty.FNMR(0.5) != 0 {
		t.Errorf("no pairs at all: FMR %v, FNMR %v, want 0 and 0", empty.FMR(0.5), empty.FNMR(0.5))
	}
}

func TestEER(t *testing.T) {
	for name, c := range map[string]struct {
		genuine, impostor []float32
		rate, threshold   float64
	}{
		// FMR and FNMR are both a third from 0.451 to 0.55; the middle of
		// the stretch is reported.
		"equal over a stretch": {[]float32{0.45, 0.8, 0.9}, []float32{0.1, 0.3, 0.55}, 1.0 / 3, 0.5005},
		// At 0.500 FMR is 1/4 and FNMR 0; at 0.501 FMR is 0 and FNMR 1/4.
		// The lines cross half-way, at 1/8.
		"between two edges": {[]float32{0.5005, 0.9, 0.9, 0.9}, []float32{0.1, 0.1, 0.1, 0.5005}, 0.125, 0.5005},
		// At 0.500 FMR is 1/2 and FNMR 0; at 0.501 FMR is 0 and FNMR 1/5.
		// The gap of 1/2 closes and opens to 1/5 the other way: the
		// crossing is 5/7 of the way along, where both are 1/7.
		"between two edges, unevenly": {[]float32{0.5005, 0.9, 0.9, 0.9, 0.9}, []float32{0.1, 0.1, 0.5005, 0.5005}, 1.0 / 7, 0.5 + 5.0/7000},
		// Nothing overlaps: no error from 0.101 to 0.9.
		"apart": {[]float32{0.9, 0.95}, []float32{0.1, 0.05}, 0, 0.5005},
		// Next to each other: no error at 0.501 and nowhere else.
		"side by side": {[]float32{0.5015, 0.9}, []float32{0.1, 0.5005}, 0, 0.501},
		// Every pair scores the same: any threshold fails one side
		// entirely, and the lines cross at a half within the one bin.
		"all scores equal": {repeat(0.5, 7), repeat(0.5, 3), 0.5, 0.5005},
		// The wrong way round: strangers score higher than the same person.
		"upside down": {[]float32{0.1}, []float32{0.9}, 1, 0.5005},
		// Identical vectors throughout: the crossing is not put above 1.
		"all scores 1":  {repeat(1, 4), repeat(1, 4), 0.5, 1},
		"all scores -1": {repeat(-1, 4), repeat(-1, 4), 0.5, -0.9995},
	} {
		rate, threshold, ok := scored(c.genuine, c.impostor).EER()
		if !ok || !near(rate, c.rate) || !near(threshold, c.threshold) {
			t.Errorf("%s: an equal error rate of %v at %v (%v), want %v at %v", name, rate, threshold, ok, c.rate, c.threshold)
		}
	}

	for name, s := range map[string]*Scores{
		"no pairs":          {},
		"no genuine pairs":  scored(nil, []float32{0.1}),
		"no impostor pairs": scored([]float32{0.9}, nil),
	} {
		if rate, threshold, ok := s.EER(); ok || rate != 0 || threshold != 0 {
			t.Errorf("%s: an equal error rate of %v at %v", name, rate, threshold)
		}
	}
}

func TestAtFMR(t *testing.T) {
	// A thousand impostor scores, one in each bin from 0.000 to 0.999, so
	// that k of them are at or above 1 - k/1000; and four genuine ones.
	impostor := make([]float32, 1000)
	for i := range impostor {
		impostor[i] = float32(i) / 1000
	}
	s := scored([]float32{0.5, 0.985, 0.995, 1}, impostor)

	for _, c := range []struct {
		target    float64
		want      Point
		whyNoMore string
	}{
		{1e-2, Point{Threshold: 0.99, FMR: 0.01, FNMR: 0.5, Supported: true}, "ten of a thousand may pass"},
		{0.0105, Point{Threshold: 0.99, FMR: 0.01, FNMR: 0.5, Supported: true}, "ten and a half is ten"},
		{0.25, Point{Threshold: 0.75, FMR: 0.25, FNMR: 0.25, Supported: true}, "a quarter may pass"},
		{1, Point{Threshold: -1, FMR: 1, FNMR: 0, Supported: true}, "everything may pass"},
		{1e-3, Point{Threshold: 0.999, FMR: 0.001, FNMR: 0.75}, "one may pass, and a thousand pairs cannot show that"},
		{0, Point{Threshold: 1, FMR: 0, FNMR: 0.75}, "nothing may pass"},
	} {
		if got := s.AtFMR(c.target); !near(got.Threshold, c.want.Threshold) || !near(got.FMR, c.want.FMR) ||
			!near(got.FNMR, c.want.FNMR) || got.Supported != c.want.Supported {
			t.Errorf("for an FMR of %v (%s): %+v, want %+v", c.target, c.whyNoMore, got, c.want)
		}
	}

	// identical vectors under two names: no threshold keeps them apart
	twins := scored([]float32{1}, append(repeat(0.2, 8), 1, 1))
	if got := twins.AtFMR(0.1); got.Threshold != 1 || !near(got.FMR, 0.2) || got.FNMR != 0 {
		t.Errorf("a fifth of the impostors score 1: %+v, want the threshold 1 and the FMR left at 0.2", got)
	}

	var empty Scores
	if got := empty.AtFMR(1e-2); got.Supported || got.Threshold != -1 || got.FMR != 0 || got.FNMR != 0 {
		t.Errorf("no pairs: %+v", got)
	}
}

func TestPairsNeeded(t *testing.T) {
	for target, want := range map[float64]uint64{
		1e-1: 100, 1e-2: 1_000, 1e-3: 10_000, 1e-4: 100_000, 1e-5: 1_000_000, 1e-6: 10_000_000,
		0.3: 34, 1: 10, 2.5e-9: 4_000_000_000,
		// nothing measures these
		0: math.MaxUint64, -0.1: math.MaxUint64, 1e-300: math.MaxUint64,
	} {
		if got := PairsNeeded(target); got != want {
			t.Errorf("an FMR of %v takes %d pairs, want %d", target, got, want)
		}
	}
	if got := PairsNeeded(math.NaN()); got != math.MaxUint64 {
		t.Errorf("no target at all takes %d pairs", got)
	}

	// the flag of a point follows the same count
	for pairs, supported := range map[int]bool{999: false, 1000: true, 1001: true} {
		if got := scored(nil, repeat(0.1, pairs)).AtFMR(1e-2).Supported; got != supported {
			t.Errorf("%d impostor pairs support an FMR of 1%%: %v, want %v", pairs, got, supported)
		}
	}
}

func TestSearchFMR(t *testing.T) {
	for _, c := range []struct {
		fpir    float64
		gallery int
	}{
		{0.01, 1000}, {0.01, 100000}, {0.001, 10000}, {0.1, 50}, {0.5, 1000},
	} {
		fmr := SearchFMR(c.fpir, c.gallery)
		// what a search of that many strangers then comes to
		fpir := 1 - math.Pow(1-fmr, float64(c.gallery))
		if math.Abs(fpir-c.fpir) > c.fpir/float64(c.gallery) {
			t.Errorf("an FMR of %v in a gallery of %d gives an FPIR of %v, want %v", fmr, c.gallery, fpir, c.fpir)
		}
	}
	if got, want := SearchFMR(0.01, 1000), 1.0050335853501442e-05; math.Abs(got-want) > 1e-18 {
		t.Errorf("for an FPIR of 1%% among 1000: %v, want %v", got, want)
	}
	if got := SearchFMR(0.01, 1); !near(got, -math.Log(0.99)) {
		t.Errorf("a gallery of one: %v", got)
	}
}
