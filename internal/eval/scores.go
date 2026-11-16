package eval

import "math"

// Scores are the two distributions a one-to-one comparison is judged by:
// the scores of pairs of pictures of the same person, and those of pairs
// of pictures of different people.
type Scores struct {
	Genuine, Impostor Histogram
}

// FMR, the false match rate, is the share of impostor pairs that score at
// or above the threshold t: strangers taken for the same person. It is 0
// when there are no impostor pairs.
func (s *Scores) FMR(t float64) float64 {
	return share(s.Impostor.AtOrAbove(t), s.Impostor.n)
}

// FNMR, the false non-match rate, is the share of genuine pairs that score
// below the threshold t: a person not recognised as themselves. It is 0
// when there are no genuine pairs.
func (s *Scores) FNMR(t float64) float64 {
	return share(s.Genuine.Below(t), s.Genuine.n)
}

// EER is the equal error rate, the rate at the threshold where FMR and
// FNMR are the same, and that threshold. As the threshold rises FMR falls
// and FNMR climbs. Where they pass each other between two bin edges, the
// crossing is taken on the straight lines between the edges; where they
// are equal over a stretch of thresholds, as they are at zero when the two
// kinds of pair do not overlap, the middle of the stretch is given. ok is
// false when either distribution is empty.
func (s *Scores) EER() (rate, threshold float64, ok bool) {
	genuine, impostor := s.Genuine.n, s.Impostor.n
	if genuine == 0 || impostor == 0 {
		return 0, 0, false
	}
	above, below := impostor, uint64(0)
	fmr, fnmr := 1.0, 0.0
	for edge := 1; edge <= bins; edge++ {
		above -= s.Impostor.counts[edge-1]
		below += s.Genuine.counts[edge-1]
		nextFMR, nextFNMR := share(above, impostor), share(below, genuine)
		switch {
		case nextFNMR < nextFMR:
			fmr, fnmr = nextFMR, nextFNMR
			continue
		case nextFNMR == nextFMR:
			// Neither moves until the next bin that holds a score.
			last := edge
			for last < bins-1 && s.Impostor.counts[last] == 0 && s.Genuine.counts[last] == 0 {
				last++
			}
			return nextFMR, (thresholdOf(edge) + thresholdOf(last)) / 2, true
		}
		// how far along the way from the last edge to this one
		along := (fmr - fnmr) / ((fmr - fnmr) + (nextFNMR - nextFMR))
		return fmr + along*(nextFMR-fmr), min(thresholdOf(edge-1)+along/binsPerUnit, 1), true
	}
	// Past the last bin nothing is at or above and everything is below,
	// so the loop has returned.
	return 0, 0, false
}

// Point is a threshold and the error rates it gives.
type Point struct {
	Threshold float64
	FMR, FNMR float64
	// Supported is false when there are too few impostor pairs to measure
	// a false match rate as low as the one asked for (see PairsNeeded).
	// The threshold of such a point is a guess from a handful of pairs
	// and should not be shown as a measurement.
	Supported bool
}

// AtFMR finds the lowest threshold at which the false match rate is no
// higher than target. If even a threshold of 1 lets more through, which
// takes identical vectors under different names, the point is at 1 with
// the rate that remains.
func (s *Scores) AtFMR(target float64) Point {
	impostor := s.Impostor.n
	above, edge := impostor, 0
	for edge < bins-1 && share(above, impostor) > target {
		above -= s.Impostor.counts[edge]
		edge++
	}
	t := thresholdOf(edge)
	return Point{
		Threshold: t,
		FMR:       share(above, impostor),
		FNMR:      s.FNMR(t),
		Supported: impostor >= PairsNeeded(target),
	}
}

// PairsNeeded is the number of impostor pairs it takes to measure a false
// match rate as low as target: enough for ten false matches to be expected
// at that rate. With fewer, the rate seen is mostly chance.
func PairsNeeded(target float64) uint64 {
	if !(target > 0) {
		return math.MaxUint64
	}
	// just under the quotient, so that 10/0.01 is a thousand and not one
	// more for the sake of the last bit
	n := math.Ceil(10 / target * (1 - 1e-12))
	if n >= math.MaxUint64 {
		return math.MaxUint64
	}
	return uint64(n)
}

// SearchFMR is the false match rate a single comparison may have when a
// search among gallery people is to name somebody for a stranger no more
// often than fpir, the false positive identification rate. The search goes
// wrong when any one of its comparisons does, with probability
// 1-(1-FMR)^gallery; solved for FMR that is -ln(1-fpir)/gallery, to within
// a hair for any gallery worth the name.
func SearchFMR(fpir float64, gallery int) float64 {
	return -math.Log1p(-fpir) / float64(gallery)
}
