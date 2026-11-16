package bench

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/muhriddinsalohiddin/likeness/index"
	"github.com/muhriddinsalohiddin/likeness/internal/enginetest"
)

const ms = time.Millisecond

func TestPercentiles(t *testing.T) {
	upTo := func(n int) []time.Duration {
		samples := make([]time.Duration, n)
		for i := range samples {
			samples[i] = time.Duration(i+1) * ms
		}
		return samples
	}
	for name, c := range map[string]struct {
		samples []time.Duration
		ps      []float64
		want    []time.Duration
	}{
		// rank = ceil(p/100 * 5): 1, 2, 2, 3, 5
		"five samples": {[]time.Duration{15 * ms, 20 * ms, 35 * ms, 40 * ms, 50 * ms}, []float64{5, 30, 40, 50, 100}, []time.Duration{15 * ms, 20 * ms, 20 * ms, 35 * ms, 50 * ms}},
		"out of order": {[]time.Duration{40 * ms, 15 * ms, 50 * ms, 20 * ms, 35 * ms}, []float64{30, 50, 100}, []time.Duration{20 * ms, 35 * ms, 50 * ms}},
		"one sample":   {[]time.Duration{7 * ms}, []float64{0, 1, 50, 99, 100}, []time.Duration{7 * ms, 7 * ms, 7 * ms, 7 * ms, 7 * ms}},
		"two samples":  {[]time.Duration{3 * ms, 9 * ms}, []float64{50, 51, 99}, []time.Duration{3 * ms, 9 * ms, 9 * ms}},
		// ranks 5, 9, ceil(9.9) = 10, 10; nothing is below the first sample
		"ten samples": {upTo(10), []float64{0, 50, 90, 99, 100}, []time.Duration{1 * ms, 5 * ms, 9 * ms, 10 * ms, 10 * ms}},
		// whole ranks stay whole: the 7th of a hundred, not the 8th
		"a hundred samples": {upTo(100), []float64{7, 50, 90, 99, 99.5, 100}, []time.Duration{7 * ms, 50 * ms, 90 * ms, 99 * ms, 100 * ms, 100 * ms}},
		"two hundred":       {upTo(200), []float64{50, 99}, []time.Duration{100 * ms, 198 * ms}},
		"equal samples":     {[]time.Duration{4 * ms, 4 * ms, 4 * ms}, []float64{50, 99}, []time.Duration{4 * ms, 4 * ms}},
		"no samples":        {nil, []float64{50, 99}, []time.Duration{0, 0}},
	} {
		before := slices.Clone(c.samples)
		if got := percentiles(c.samples, c.ps...); !slices.Equal(got, c.want) {
			t.Errorf("%s: percentiles %v are %v, want %v", name, c.ps, got, c.want)
		}
		if !slices.Equal(c.samples, before) {
			t.Errorf("%s: the samples were reordered", name)
		}
	}
}

func TestSummarise(t *testing.T) {
	samples := make([]time.Duration, 0, 100)
	for i := 100; i >= 1; i-- {
		samples = append(samples, time.Duration(i)*ms)
	}
	if got, want := summarise(samples), (spread{P50: millis(50 * ms), P90: millis(90 * ms), P99: millis(99 * ms), Max: millis(100 * ms)}); got != want {
		t.Errorf("1 to 100 ms: %+v, want %+v", got, want)
	}
}

func TestRecommend(t *testing.T) {
	level := func(inflight int, p99 time.Duration) levelTimes {
		return levelTimes{Inflight: inflight, PerSecond: float64(10 * inflight), P99: millis(p99)}
	}
	for name, c := range map[string]struct {
		levels []levelTimes
		want   int
	}{
		"the highest that keeps the target": {[]levelTimes{level(1, 40*ms), level(2, 60*ms), level(4, 95*ms), level(8, 180*ms)}, 4},
		"every level keeps it":              {[]levelTimes{level(1, 40*ms), level(2, 60*ms)}, 2},
		"exactly on the target":             {[]levelTimes{level(1, 40*ms), level(2, 100*ms), level(4, 100*ms+1)}, 2},
		"only the first":                    {[]levelTimes{level(1, 99*ms), level(2, 101*ms)}, 1},
		"none qualifies":                    {[]levelTimes{level(1, 130*ms), level(2, 150*ms), level(4, 300*ms)}, 0},
		"a lower level strayed":             {[]levelTimes{level(1, 40*ms), level(2, 120*ms), level(4, 90*ms)}, 4},
		"levels out of order":               {[]levelTimes{level(4, 90*ms), level(8, 200*ms), level(1, 30*ms), level(2, 50*ms)}, 4},
		"nothing measured":                  {nil, 0},
	} {
		got := recommend(c.levels, 100*ms)
		if got.MaxInflight != c.want || got.Target != millis(100*ms) {
			t.Errorf("%s: %+v, want level %d", name, got, c.want)
			continue
		}
		if c.want == 0 {
			if got.PerSecond != 0 || got.P99 != 0 {
				t.Errorf("%s: no level, and yet %+v", name, got)
			}
			continue
		}
		// the numbers are those of the level picked
		for _, l := range c.levels {
			if l.Inflight == c.want && (got.PerSecond != l.PerSecond || got.P99 != l.P99) {
				t.Errorf("%s: %+v does not repeat the level %+v", name, got, l)
			}
		}
	}
}

func anna(text string) picture { return picture{name: "anna.jpg", data: []byte(text)} }

func TestTimeStages(t *testing.T) {
	fake := &enginetest.Fake{}
	// however short the time, one picture is analysed
	got, err := timeStages(context.Background(), fake, anna("face:anna"), time.Nanosecond)
	if err != nil || got.Pictures < 1 || got.Pictures != fake.Calls() {
		t.Fatalf("%+v after %d calls: %v", got, fake.Calls(), err)
	}
	if got.Decode.P50 != millis(2*ms) || got.Detect.P90 != millis(3*ms) || got.Align.P99 != millis(ms) || got.Embed.Max != millis(4*ms) {
		t.Errorf("the stages: %+v", got)
	}

	// the total is the time the caller waited, not the engine's own sum
	slow := &enginetest.Fake{Delay: 15 * ms}
	got, err = timeStages(context.Background(), slow, anna("slow:anna"), 40*ms)
	if err != nil || got.Total.P50 < millis(15*ms) || got.Embed.P50 != millis(4*ms) {
		t.Errorf("a slow engine: %+v %v", got, err)
	}
}

func TestFill(t *testing.T) {
	ctx := context.Background()
	const subjects = 1500
	ix, err := fill(ctx, enginetest.Dim, subjects)
	if err != nil {
		t.Fatal(err)
	}
	defer ix.Close()
	if st := ix.Stats(); st.Subjects != subjects || st.Templates != subjects || st.Dim != enginetest.Dim {
		t.Errorf("the gallery: %+v", st)
	}

	// the faces are random: nobody in the gallery looks like anna, and the
	// same nobody on every run
	query := enginetest.Vector("anna")
	found, err := ix.Search(query, 1)
	if err != nil || len(found) != 1 || found[0].Score > 0.9 {
		t.Fatalf("searching: %+v %v", found, err)
	}
	again, err := fill(ctx, enginetest.Dim, subjects)
	if err != nil {
		t.Fatal(err)
	}
	defer again.Close()
	if same, err := again.Search(query, 1); err != nil || len(same) != 1 || same[0] != found[0] {
		t.Errorf("a second gallery answers %+v, the first %+v (%v)", same, found, err)
	}

	if _, err := fill(ctx, 0, 10); err == nil {
		t.Error("a gallery of vectors with no values was built")
	}
}

func TestTimeSearches(t *testing.T) {
	ctx := context.Background()
	ix, err := fill(ctx, enginetest.Dim, 40)
	if err != nil {
		t.Fatal(err)
	}
	defer ix.Close()
	query := enginetest.Vector("anna")

	// the time is not used up when enough searches have been timed
	got, err := timeSearches(ctx, ix, query, time.Hour)
	if err != nil || got.Gallery != 40 || got.Searches != maxSearches || got.Kernel != index.Kernel() || got.P50 <= 0 || got.P50 > got.P99 {
		t.Errorf("an hour for a small gallery: %+v %v", got, err)
	}
	// and one search is timed however short it is
	got, err = timeSearches(ctx, ix, query, time.Nanosecond)
	if err != nil || got.Searches < 1 {
		t.Errorf("a nanosecond: %+v %v", got, err)
	}

	if _, err := timeSearches(ctx, ix, query[:8], time.Second); !errors.Is(err, index.ErrDimension) {
		t.Errorf("a query of the wrong length: %v", err)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := timeSearches(cancelled, ix, query, time.Second); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled: %v", err)
	}
}

func TestTimeAtOnce(t *testing.T) {
	ctx := context.Background()
	ix, err := fill(ctx, enginetest.Dim, 40)
	if err != nil {
		t.Fatal(err)
	}
	defer ix.Close()

	fake := &enginetest.Fake{Delay: 5 * ms}
	got, err := timeAtOnce(ctx, fake, anna("slow:anna"), ix, 3, 40*ms)
	if err != nil {
		t.Fatal(err)
	}
	if got.Inflight != 3 || got.Gallery != 40 || got.Pictures != fake.Calls() || got.Pictures < 3 {
		t.Errorf("three at once: %+v after %d calls", got, fake.Calls())
	}
	if fake.Peak() != 3 {
		t.Errorf("%d pictures were analysed at once, want 3", fake.Peak())
	}
	// three pictures every 5 ms at the very best
	if got.P50 < millis(5*ms) || got.P50 > got.P99 || got.PerSecond <= 0 || got.PerSecond > 600 {
		t.Errorf("three at once: %+v", got)
	}

	// the failure that is reported is the one that stopped the others
	engine := tiringAfter(4, &enginetest.Fake{Delay: 2 * ms})
	start := time.Now()
	if _, err := timeAtOnce(ctx, engine, anna("slow:anna"), ix, 3, time.Minute); !errors.Is(err, errWornOut) {
		t.Errorf("an engine that fails halfway: %v", err)
	}
	if time.Since(start) > 10*time.Second {
		t.Error("the others carried on after the failure")
	}

	// a face that cannot be searched for
	if _, err := timeAtOnce(ctx, &enginetest.Fake{}, anna("face:"), ix, 2, time.Minute); !errors.Is(err, index.ErrDimension) {
		t.Errorf("a face without a vector: %v", err)
	}
}

func TestMeasureStopsWhenCancelled(t *testing.T) {
	onMachine(t, 4)
	p := plan{duration: 10 * ms, galleries: []int{50, 3000}, inflight: []int{1, 2}, target: time.Second}
	// called off as each part of the run begins
	for _, at := range []string{"warming up", "one picture at a time", "searching a gallery of %d", "at once"} {
		ctx, cancel := context.WithCancel(context.Background())
		reached := false
		note := func(format string, _ ...any) {
			if strings.Contains(format, at) {
				reached = true
				cancel()
			}
		}
		rep, err := measure(ctx, &enginetest.Fake{}, anna("face:anna"), p, note)
		if !reached {
			t.Errorf("no note says %q", at)
		}
		if rep != nil || !errors.Is(err, context.Canceled) {
			t.Errorf("called off at %q: %v, with a report: %t", at, err, rep != nil)
		}
		cancel()
	}
}
