package eval

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"strings"
	"testing"
)

var ctx = context.Background()

// axes returns a unit vector of dim values that points along the given
// axes with the given weights: axes(5, 0, 0.6, 1, 0.8) is 0.6 along the
// first axis and 0.8 along the second.
func axes(dim int, pairs ...float64) []float32 {
	v := make([]float32, dim)
	for i := 0; i < len(pairs); i += 2 {
		v[int(pairs[i])] = float32(pairs[i+1])
	}
	return v
}

func unit(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	var sum float64
	for i := range v {
		v[i] = float32(rng.NormFloat64())
		sum += float64(v[i]) * float64(v[i])
	}
	for i := range v {
		v[i] /= float32(math.Sqrt(sum))
	}
	return v
}

// like returns a unit vector at about the given cosine from v.
func like(rng *rand.Rand, v []float32, cos float64) []float32 {
	noise := unit(rng, len(v))
	out := make([]float32, len(v))
	var sum float64
	for i := range v {
		out[i] = float32(cos)*v[i] + float32(math.Sqrt(1-cos*cos))*noise[i]
		sum += float64(out[i]) * float64(out[i])
	}
	for i := range out {
		out[i] /= float32(math.Sqrt(sum))
	}
	return out
}

// crowd makes people with the given numbers of pictures: pictures of one
// person are alike, different people are not.
func crowd(rng *rand.Rand, dim int, pictures ...int) []Person {
	people := make([]Person, len(pictures))
	for i, n := range pictures {
		face := unit(rng, dim)
		for range n {
			people[i] = append(people[i], like(rng, face, 0.9))
		}
	}
	return people
}

func TestPairsByHand(t *testing.T) {
	half := math.Sqrt(0.5)
	people := []Person{
		{axes(4, 0, 1), axes(4, 0, 1), axes(4, 0, 0.6, 1, 0.8)}, // similarities 1, 0.6 and 0.6
		{axes(4, 1, 1)}, // 0, 0 and 0.8 with the first person's pictures
		{axes(4, 0, half, 1, half), axes(4, 2, 1)}, // 0.707, 0.707, 0.990 and 0.707; 0 four times; 0 with each other
	}
	s, err := Pairs(ctx, people, 0)
	if err != nil {
		t.Fatal(err)
	}
	if s.Genuine.Count() != 4 || s.Impostor.Count() != 11 {
		t.Fatalf("%d genuine and %d impostor pairs, want 4 and 11", s.Genuine.Count(), s.Impostor.Count())
	}
	for _, c := range []struct {
		threshold float64
		genuine   uint64
		impostor  uint64
	}{
		{-0.5, 4, 11}, {0, 4, 11}, {0.001, 3, 5}, {0.6, 3, 5}, {0.65, 1, 5}, {0.75, 1, 2}, {0.85, 1, 1}, {0.995, 1, 0},
	} {
		if g, i := s.Genuine.AtOrAbove(c.threshold), s.Impostor.AtOrAbove(c.threshold); g != c.genuine || i != c.impostor {
			t.Errorf("at or above %v: %d genuine and %d impostor pairs, want %d and %d", c.threshold, g, i, c.genuine, c.impostor)
		}
	}
	if want := (1 + 0.6 + 0.6 + 0) / 4; math.Abs(s.Genuine.Mean()-want) > 1e-6 {
		t.Errorf("genuine pairs average %v, want %v", s.Genuine.Mean(), want)
	}
	if want := (0.8 + 3*half + 0.7*half*2) / 11; math.Abs(s.Impostor.Mean()-want) > 1e-6 {
		t.Errorf("impostor pairs average %v, want %v", s.Impostor.Mean(), want)
	}
}

func TestPairsCounts(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	// 21 values, so that the dot product has a tail to deal with
	people := crowd(rng, 21, 4, 1, 0, 7, 2, 3, 1, 5)
	one, err := Pairs(ctx, people, 1)
	if err != nil {
		t.Fatal(err)
	}
	// 23 pictures make 253 pairs; 6+21+1+3+10 of them within a person
	if one.Genuine.Count() != 41 || one.Impostor.Count() != 212 {
		t.Fatalf("%d genuine and %d impostor pairs, want 41 and 212", one.Genuine.Count(), one.Impostor.Count())
	}
	if g, i := one.Genuine.Mean(), one.Impostor.Mean(); g < 0.7 || g > 0.9 || math.Abs(i) > 0.1 {
		t.Errorf("genuine pairs average %v and impostor pairs %v: the two kinds are not told apart", g, i)
	}

	// however the work is divided, the same pairs are counted
	for _, workers := range []int{0, 2, 3, 64} {
		many, err := Pairs(ctx, people, workers)
		if err != nil {
			t.Fatal(err)
		}
		if many.Genuine.counts != one.Genuine.counts || many.Impostor.counts != one.Impostor.counts {
			t.Errorf("%d workers counted other scores than one did", workers)
		}
		if math.Abs(many.Impostor.Mean()-one.Impostor.Mean()) > 1e-12 || math.Abs(many.Genuine.StdDev()-one.Genuine.StdDev()) > 1e-12 {
			t.Errorf("%d workers: mean %v and deviation %v, one worker %v and %v", workers,
				many.Impostor.Mean(), many.Genuine.StdDev(), one.Impostor.Mean(), one.Genuine.StdDev())
		}
	}

	// every pair agrees with a dot product done the slow way
	var slow Scores
	var flat [][]float32
	var owner []int
	for i, p := range people {
		for _, v := range p {
			flat, owner = append(flat, v), append(owner, i)
		}
	}
	for a := range flat {
		for b := a + 1; b < len(flat); b++ {
			var sum float64
			for k := range flat[a] {
				sum += float64(flat[a][k]) * float64(flat[b][k])
			}
			if owner[a] == owner[b] {
				slow.Genuine.Add(float32(sum))
			} else {
				slow.Impostor.Add(float32(sum))
			}
		}
	}
	if math.Abs(slow.Genuine.Mean()-one.Genuine.Mean()) > 1e-6 || math.Abs(slow.Impostor.Mean()-one.Impostor.Mean()) > 1e-6 {
		t.Errorf("means %v and %v, the slow way %v and %v", one.Genuine.Mean(), one.Impostor.Mean(), slow.Genuine.Mean(), slow.Impostor.Mean())
	}
}

func TestPairsOfFew(t *testing.T) {
	for name, people := range map[string][]Person{
		"nobody":               nil,
		"people with nothing":  {{}, {}, nil},
		"one picture":          {{axes(3, 0, 1)}},
		"one picture and gaps": {{}, {axes(3, 0, 1)}, {}},
	} {
		s, err := Pairs(ctx, people, 0)
		if err != nil || s.Genuine.Count() != 0 || s.Impostor.Count() != 0 {
			t.Errorf("%s: %d genuine and %d impostor pairs, %v", name, s.Genuine.Count(), s.Impostor.Count(), err)
		}
	}

	// one person only: genuine pairs and nothing to tell them from
	s, err := Pairs(ctx, []Person{{axes(3, 0, 1), axes(3, 0, 1), axes(3, 1, 1)}}, 0)
	if err != nil || s.Genuine.Count() != 3 || s.Impostor.Count() != 0 {
		t.Errorf("three pictures of one person: %d genuine and %d impostor pairs, %v", s.Genuine.Count(), s.Impostor.Count(), err)
	}
	// two people, one picture each: the reverse
	s, err = Pairs(ctx, []Person{{axes(3, 0, 1)}, {axes(3, 1, 1)}}, 0)
	if err != nil || s.Genuine.Count() != 0 || s.Impostor.Count() != 1 {
		t.Errorf("two people: %d genuine and %d impostor pairs, %v", s.Genuine.Count(), s.Impostor.Count(), err)
	}
}

func TestPairsSkipsNaN(t *testing.T) {
	broken := axes(3, 0, 1)
	broken[2] = float32(math.NaN())
	s, err := Pairs(ctx, []Person{{axes(3, 0, 1), broken}, {axes(3, 1, 1)}}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if s.Genuine.Count() != 0 || s.Genuine.Skipped() != 1 || s.Impostor.Count() != 1 || s.Impostor.Skipped() != 1 {
		t.Errorf("a vector with a NaN in it: %d+%d genuine and %d+%d impostor pairs counted+skipped, want 0+1 and 1+1",
			s.Genuine.Count(), s.Genuine.Skipped(), s.Impostor.Count(), s.Impostor.Skipped())
	}
}

func TestPairsRefuses(t *testing.T) {
	for name, c := range map[string]struct {
		people []Person
		says   string
	}{
		"two lengths":      {[]Person{{axes(4, 0, 1)}, {axes(4, 1, 1), axes(3, 1, 1)}}, "picture 2 of person 2 has a vector of 3 values, the others have 4"},
		"an empty vector":  {[]Person{{{}}}, "no values"},
		"empty then whole": {[]Person{{{}}, {axes(4, 1, 1)}}, "picture 1 of person 1 has a vector of 0 values"},
	} {
		if _, err := Pairs(ctx, c.people, 0); err == nil || !strings.Contains(err.Error(), c.says) {
			t.Errorf("%s: %v, want an error that says %q", name, err, c.says)
		}
	}
}

func TestPairsStopsWhenCancelled(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	people := crowd(rng, 16, 5, 5, 5, 5)
	stopped, cancel := context.WithCancel(ctx)
	cancel()
	if s, err := Pairs(stopped, people, 2); !errors.Is(err, context.Canceled) || s != nil {
		t.Errorf("after the context was cancelled: %v, %v", s, err)
	}
}

func TestDot(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, dim := range []int{1, 2, 3, 4, 5, 7, 8, 64, 127, 128, 512} {
		a, b := unit(rng, dim), unit(rng, dim)
		var want float64
		for i := range a {
			want += float64(a[i]) * float64(b[i])
		}
		if got := dot(a, b); math.Abs(float64(got)-want) > 1e-6 {
			t.Errorf("%d values: %v, want %v", dim, got, want)
		}
		if self := dot(a, a); math.Abs(float64(self)-1) > 1e-5 {
			t.Errorf("%d values: a unit vector with itself gives %v", dim, self)
		}
	}
}
