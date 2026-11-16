// Package eval measures how well face vectors tell people apart: how the
// similarities of pictures of the same person and of different people are
// distributed, which error rates a threshold gives, and how a search fares
// in a gallery of a given size.
//
// It works on vectors and scores alone. Reading pictures and running the
// models is the caller's business.
package eval

import "math"

const (
	// A bin is a thousandth wide, which is as finely as a threshold is set.
	binsPerUnit = 1000
	// Bin i holds the scores from -1 + i/binsPerUnit up to the next bin;
	// the last one holds the scores of exactly 1.
	bins = 2*binsPerUnit + 1
	// A float32 that ought to sit on a bin edge is off it by up to 3e-8:
	// 0.7 is stored as 0.69999999. This much of a bin is added so that
	// such a score lands on the edge, where the server's own float32
	// comparison with the threshold puts it.
	edgeSlack = 1e-4
)

// Histogram is the distribution of a set of similarity scores, counted in
// bins 0.001 wide over [-1, 1]. Counting the scores instead of keeping them
// is what lets a hundred million comparisons fit in a few kilobytes. The
// zero value is empty and ready to use.
type Histogram struct {
	counts  [bins]uint64
	n       uint64
	skipped uint64
	sum     float64
	squares float64
}

// Add counts one score. A score outside [-1, 1], which rounding can
// produce, counts as -1 or 1. NaN is not a score: it is left out and
// tallied in Skipped.
func (h *Histogram) Add(score float32) {
	s := float64(score)
	if math.IsNaN(s) {
		h.skipped++
		return
	}
	s = min(max(s, -1), 1)
	h.counts[min(int((s+1)*binsPerUnit+edgeSlack), bins-1)]++
	h.n++
	h.sum += s
	h.squares += s * s
}

// Merge adds the scores counted in other.
func (h *Histogram) Merge(other *Histogram) {
	for i, c := range other.counts {
		h.counts[i] += c
	}
	h.n += other.n
	h.skipped += other.skipped
	h.sum += other.sum
	h.squares += other.squares
}

// Count is the number of scores counted.
func (h *Histogram) Count() uint64 { return h.n }

// Skipped is the number of NaNs that were offered and left out.
func (h *Histogram) Skipped() uint64 { return h.skipped }

// Mean is the average score, 0 when there are none.
func (h *Histogram) Mean() float64 {
	if h.n == 0 {
		return 0
	}
	return h.sum / float64(h.n)
}

// StdDev is the standard deviation of the scores, 0 when there are none.
func (h *Histogram) StdDev() float64 {
	if h.n == 0 {
		return 0
	}
	mean := h.Mean()
	return math.Sqrt(max(h.squares/float64(h.n)-mean*mean, 0))
}

// AtOrAbove is the number of scores at or above the threshold t. The
// threshold is taken to the nearest thousandth.
func (h *Histogram) AtOrAbove(t float64) uint64 {
	var n uint64
	for _, c := range h.counts[edgeOf(t):] {
		n += c
	}
	return n
}

// Below is the number of scores below the threshold t.
func (h *Histogram) Below(t float64) uint64 { return h.n - h.AtOrAbove(t) }

// edgeOf returns the first bin whose scores are at or above the threshold
// t; bins if no score can be.
func edgeOf(t float64) int {
	switch {
	case math.IsNaN(t), t > 1+0.5/binsPerUnit:
		return bins
	case t < -1:
		return 0
	}
	return int(math.Round((t + 1) * binsPerUnit))
}

// thresholdOf is the score at which a bin starts.
func thresholdOf(edge int) float64 { return float64(edge-binsPerUnit) / binsPerUnit }

// share is part/whole, 0 when there is nothing to take a share of.
func share(part, whole uint64) float64 {
	if whole == 0 {
		return 0
	}
	return float64(part) / float64(whole)
}
