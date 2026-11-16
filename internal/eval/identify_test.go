package eval

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/muhriddinsalohiddin/likeness/index"
)

// six people in five dimensions, laid out so that every search can be
// followed by hand. With every third person left out:
//
//	anna    enrolled along axis 0; her probe scores 0.8 with herself
//	boris   enrolled along axis 1; his probe scores 0.8 with himself, 0.6 with anna
//	clara   left out; one picture resembles nobody (0), one scores 0.6 with anna
//	dmitri  enrolled along axis 3; his probe scores 0.9 with anna and 0.44 with himself
//	elena   enrolled along axis 4, with no second picture
//	fyodor  left out; his picture scores 0.95 with elena
func sixPeople() []Person {
	return []Person{
		{axes(5, 0, 1), axes(5, 0, 0.8, 1, 0.6)},
		{axes(5, 1, 1), axes(5, 0, 0.6, 1, 0.8)},
		{axes(5, 2, 1), axes(5, 0, 0.6, 2, 0.8)},
		{axes(5, 3, 1), axes(5, 0, 0.9, 3, math.Sqrt(1-0.81))},
		{axes(5, 4, 1)},
		{axes(5, 0, math.Sqrt(1-0.9025), 4, 0.95)},
	}
}

func TestIdentifyByHand(t *testing.T) {
	id, err := Identify(ctx, sixPeople(), 3, 0)
	if err != nil {
		t.Fatal(err)
	}
	if id.Gallery != 4 || id.LeftOut != 2 || id.Mated != 3 || id.NonMated != 3 {
		t.Fatalf("a gallery of %d with %d left out, %d mated and %d non-mated probes; want 4, 2, 3 and 3",
			id.Gallery, id.LeftOut, id.Mated, id.NonMated)
	}
	third := 1.0 / 3
	if !near(id.Rank1(), 2*third) {
		t.Errorf("rank-1 hit rate %v, want two of three: dmitri's probe looks more like anna", id.Rank1())
	}
	for _, c := range []struct {
		threshold  float64
		fnir, fpir float64
	}{
		{-1, third, 1},
		{0.1, third, 2 * third},
		{0.5, third, 2 * third},
		{0.7, third, third},
		{0.85, 1, third},
		{0.99, 1, 0},
	} {
		if got := id.FNIR(c.threshold); !near(got, c.fnir) {
			t.Errorf("FNIR at %v: %v, want %v", c.threshold, got, c.fnir)
		}
		if got := id.FPIR(c.threshold); !near(got, c.fpir) {
			t.Errorf("FPIR at %v: %v, want %v", c.threshold, got, c.fpir)
		}
	}
}

func TestIdentifyHoldout(t *testing.T) {
	// nobody left out: clara's and fyodor's first pictures are enrolled too
	id, err := Identify(ctx, sixPeople(), 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if id.Gallery != 6 || id.LeftOut != 0 || id.Mated != 4 || id.NonMated != 0 {
		t.Fatalf("with nobody left out: a gallery of %d with %d left out, %d mated and %d non-mated probes; want 6, 0, 4 and 0",
			id.Gallery, id.LeftOut, id.Mated, id.NonMated)
	}
	if !near(id.Rank1(), 0.75) || !near(id.FNIR(0.7), 0.25) || !near(id.FNIR(0.9), 1) {
		t.Errorf("rank-1 hit rate %v, FNIR %v at 0.7 and %v at 0.9; want 0.75, 0.25 and 1", id.Rank1(), id.FNIR(0.7), id.FNIR(0.9))
	}
	if id.FPIR(-1) != 0 {
		t.Errorf("an FPIR of %v with no non-mated probes", id.FPIR(-1))
	}

	// people without usable pictures do not shift who is left out
	gaps := sixPeople()
	gaps = append([]Person{{}}, append(gaps[:3:3], append([]Person{nil, {}}, gaps[3:]...)...)...)
	a, err := Identify(ctx, sixPeople(), 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Identify(ctx, gaps, 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	if *a != *b {
		t.Errorf("with empty people in between: %d enrolled, %d left out, %d and %d probes; without: %d, %d, %d and %d",
			b.Gallery, b.LeftOut, b.Mated, b.NonMated, a.Gallery, a.LeftOut, a.Mated, a.NonMated)
	}

	for holdout, want := range map[int]Identification{
		1: {Gallery: 0, LeftOut: 6, Mated: 0, NonMated: 10},
		2: {Gallery: 3, LeftOut: 3, Mated: 2, NonMated: 5},
		5: {Gallery: 5, LeftOut: 1, Mated: 4, NonMated: 1},
		6: {Gallery: 5, LeftOut: 1, Mated: 4, NonMated: 1},
		7: {Gallery: 6, LeftOut: 0, Mated: 4, NonMated: 0},
	} {
		id, err := Identify(ctx, sixPeople(), holdout, 0)
		if err != nil {
			t.Fatal(err)
		}
		if id.Gallery != want.Gallery || id.LeftOut != want.LeftOut || id.Mated != want.Mated || id.NonMated != want.NonMated {
			t.Errorf("with a holdout of %d: a gallery of %d with %d left out, %d mated and %d non-mated probes; want %d, %d, %d and %d",
				holdout, id.Gallery, id.LeftOut, id.Mated, id.NonMated, want.Gallery, want.LeftOut, want.Mated, want.NonMated)
		}
		if holdout == 1 && (id.FPIR(-1) != 0 || id.FNIR(-1) != 0 || id.Rank1() != 0) {
			t.Errorf("an empty gallery named somebody: FPIR %v", id.FPIR(-1))
		}
	}
}

func TestIdentifyCrowd(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	pictures := make([]int, 60)
	for i := range pictures {
		pictures[i] = 1 + i%4
	}
	people := crowd(rng, 64, pictures...)
	id, err := Identify(ctx, people, 5, 0)
	if err != nil {
		t.Fatal(err)
	}
	// 12 left out, with 1+i%4 pictures each; the other 48 enrol one
	// picture and probe with the rest
	if id.Gallery != 48 || id.LeftOut != 12 || id.NonMated != 30 || id.Mated != 150-30-48 {
		t.Fatalf("a gallery of %d with %d left out, %d mated and %d non-mated probes", id.Gallery, id.LeftOut, id.Mated, id.NonMated)
	}
	if id.Rank1() != 1 || id.FNIR(0.5) != 0 || id.FPIR(0.6) != 0 {
		t.Errorf("people far apart: rank-1 hit rate %v, FNIR %v at 0.5, FPIR %v at 0.6", id.Rank1(), id.FNIR(0.5), id.FPIR(0.6))
	}
	if id.FNIR(0.99) != 1 || id.FPIR(-1) != 1 {
		t.Errorf("at the extremes: FNIR %v at 0.99, FPIR %v at -1", id.FNIR(0.99), id.FPIR(-1))
	}
	// the rates only ever move one way as the threshold rises
	for edge := 1; edge <= bins; edge++ {
		low, high := thresholdOf(edge-1), thresholdOf(edge)
		if id.FNIR(high) < id.FNIR(low) || id.FPIR(high) > id.FPIR(low) {
			t.Fatalf("between %v and %v FNIR went from %v to %v and FPIR from %v to %v",
				low, high, id.FNIR(low), id.FNIR(high), id.FPIR(low), id.FPIR(high))
		}
	}
}

func TestIdentifyNobody(t *testing.T) {
	for name, people := range map[string][]Person{
		"nobody":              nil,
		"people with nothing": {{}, nil},
	} {
		id, err := Identify(ctx, people, 5, 0)
		if err != nil || *id != (Identification{}) {
			t.Errorf("%s: %+v, %v", name, id, err)
		}
		if id.Rank1() != 0 || id.FNIR(0.5) != 0 || id.FPIR(0.5) != 0 {
			t.Errorf("%s: rates out of nothing", name)
		}
	}
}

func TestIdentifyRefuses(t *testing.T) {
	zero := make([]float32, 5)
	for name, c := range map[string]struct {
		people []Person
		is     error
	}{
		"a zero vector to enrol":   {[]Person{{axes(5, 0, 1)}, {zero, axes(5, 1, 1)}}, index.ErrVector},
		"a zero vector as a probe": {[]Person{{axes(5, 0, 1), zero}, {axes(5, 1, 1)}}, index.ErrVector},
		"a short vector to enrol":  {[]Person{{axes(5, 0, 1)}, {axes(4, 1, 1)}}, index.ErrDimension},
		"a short probe":            {[]Person{{axes(5, 0, 1), axes(4, 1, 1), axes(5, 1, 1)}}, index.ErrDimension},
	} {
		if id, err := Identify(ctx, c.people, 0, 2); !errors.Is(err, c.is) || id != nil {
			t.Errorf("%s: %+v, %v", name, id, err)
		}
	}
}

func TestIdentifyStopsWhenCancelled(t *testing.T) {
	stopped, cancel := context.WithCancel(ctx)
	cancel()
	if id, err := Identify(stopped, sixPeople(), 3, 2); !errors.Is(err, context.Canceled) || id != nil {
		t.Errorf("after the context was cancelled: %+v, %v", id, err)
	}
}
