package bench

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/muhriddinsalohiddin/likeness"
	"github.com/muhriddinsalohiddin/likeness/index"
	"github.com/muhriddinsalohiddin/likeness/internal/cli"
)

// plan says what to measure.
type plan struct {
	// duration is how long each measurement runs.
	duration time.Duration
	// galleries are the gallery sizes to search, in subjects, ascending.
	galleries []int
	// inflight are the numbers of pictures to analyse at once, ascending.
	inflight []int
	// target is what the 99th percentile of a picture has to stay within
	// for a level to be recommended.
	target time.Duration
}

const (
	// The engine is run for a fifth of the measuring time before anything
	// is measured, a second at most: the first calls find caches cold and
	// the processor at a low clock rate.
	warmUpShare = 5
	maxWarmUp   = time.Second

	// maxSearches bounds the searches timed in one gallery. A search takes
	// microseconds in a small gallery, and a few thousand of them settle
	// the 99th percentile.
	maxSearches = 2000
	// warmUpSearches run before the timed ones.
	warmUpSearches = 20
)

// measure runs the three measurements in turn and returns the report.
// note is told what is about to happen.
func measure(ctx context.Context, eng cli.Engine, pic picture, p plan, note func(format string, args ...any)) (*report, error) {
	info := eng.Info()
	rep := &report{
		Version: likeness.Version,
		Engine: engineInfo{
			Preset: info.Preset, Detector: info.Detector, Embedder: info.Embedder,
			Dim: info.Dim, Runtime: info.Runtime, Threads: info.Threads,
		},
		Machine:  machineInfo{OS: runtime.GOOS, Arch: runtime.GOARCH, CPUs: numCPU(), Go: runtime.Version()},
		Duration: millis(p.duration),
		Search:   make([]searchTimes, 0, len(p.galleries)),
		Inflight: make([]levelTimes, 0, len(p.inflight)),
	}

	note("warming up")
	first, query, err := analyse(ctx, eng, pic)
	if err != nil {
		return nil, err
	}
	rep.Picture = pictureInfo{Name: pic.name, Width: first.Width, Height: first.Height, Bytes: len(pic.data)}
	if _, err := timeStages(ctx, eng, pic, min(p.duration/warmUpShare, maxWarmUp)); err != nil {
		return nil, err
	}

	note("one picture at a time for %s", p.duration)
	if rep.Stages, err = timeStages(ctx, eng, pic, p.duration); err != nil {
		return nil, err
	}

	// One gallery is open at a time. The last one, the largest, stays for
	// the pictures analysed at once.
	var faces *index.Index
	defer func() {
		if faces != nil {
			faces.Close()
		}
	}()
	for _, subjects := range p.galleries {
		if faces != nil {
			faces.Close()
		}
		note("searching a gallery of %d", subjects)
		if faces, err = fill(ctx, info.Dim, subjects); err != nil {
			return nil, err
		}
		times, err := timeSearches(ctx, faces, query, p.duration)
		if err != nil {
			return nil, err
		}
		rep.Search = append(rep.Search, times)
	}

	for _, level := range p.inflight {
		note("%d at once for %s", level, p.duration)
		times, err := timeAtOnce(ctx, eng, pic, faces, level, p.duration)
		if err != nil {
			return nil, err
		}
		rep.Inflight = append(rep.Inflight, times)
	}
	rep.Recommendation = recommend(rep.Inflight, p.target)
	return rep, nil
}

// analyse runs the engine on the picture as a search request does and
// returns the vector of the face to search for. A picture without a usable
// face is an error: timing it would measure a request that stops halfway.
func analyse(ctx context.Context, eng cli.Engine, pic picture) (*likeness.Result, []float32, error) {
	res, err := eng.Analyze(ctx, pic.data, likeness.ForSearch())
	if err != nil {
		if ctx.Err() != nil {
			return nil, nil, ctx.Err()
		}
		return nil, nil, fmt.Errorf("%s: %w", pic.name, err)
	}
	face, reason := res.Best()
	if reason != likeness.Accepted {
		return nil, nil, fmt.Errorf("%s: no usable face in the picture (%s); "+
			"pass one with a single clear face as --image", pic.name, reason)
	}
	return res, face.Embedding, nil
}

// timeStages analyses the picture again and again for d, once at least.
func timeStages(ctx context.Context, eng cli.Engine, pic picture, d time.Duration) (stageTimes, error) {
	var decode, detect, align, embed, total []time.Duration
	start := time.Now()
	for {
		mark := time.Now()
		res, _, err := analyse(ctx, eng, pic)
		if err != nil {
			return stageTimes{}, err
		}
		total = append(total, time.Since(mark))
		decode = append(decode, res.Timing.Decode)
		detect = append(detect, res.Timing.Detect)
		align = append(align, res.Timing.Align)
		embed = append(embed, res.Timing.Embed)
		if time.Since(start) >= d {
			break
		}
	}
	return stageTimes{
		Pictures: len(total),
		Decode:   summarise(decode), Detect: summarise(detect),
		Align: summarise(align), Embed: summarise(embed),
		Total: summarise(total),
	}, nil
}

// fill builds an index of the given number of subjects, each with one
// random face. The faces are the same on every run.
func fill(ctx context.Context, dim, subjects int) (*index.Index, error) {
	ix, err := index.New(dim, index.Options{Capacity: subjects})
	if err != nil {
		return nil, err
	}
	rng := rand.New(rand.NewSource(1))
	vector := make([]float32, dim)
	for i := range subjects {
		if i%1024 == 0 && ctx.Err() != nil {
			ix.Close()
			return nil, ctx.Err()
		}
		for j := range vector {
			vector[j] = float32(rng.NormFloat64())
		}
		// Add copies the vector and scales the copy to unit length.
		if err := ix.Add("subject-"+strconv.Itoa(i), 1, vector); err != nil {
			ix.Close()
			return nil, err
		}
	}
	return ix, nil
}

// timeSearches looks the query up in the index, one search at a time, for
// d or maxSearches times, whichever is over first.
func timeSearches(ctx context.Context, ix *index.Index, query []float32, d time.Duration) (searchTimes, error) {
	search := func() (time.Duration, error) {
		mark := time.Now()
		_, err := ix.Search(query, 1)
		return time.Since(mark), err
	}
	for range warmUpSearches {
		if _, err := search(); err != nil {
			return searchTimes{}, err
		}
	}
	times := make([]time.Duration, 0, maxSearches)
	start := time.Now()
	for len(times) < maxSearches {
		if err := ctx.Err(); err != nil {
			return searchTimes{}, err
		}
		took, err := search()
		if err != nil {
			return searchTimes{}, err
		}
		times = append(times, took)
		if time.Since(start) >= d {
			break
		}
	}
	stats, all := ix.Stats(), summarise(times)
	return searchTimes{Gallery: stats.Subjects, Searches: len(times), Kernel: stats.Kernel, P50: all.P50, P99: all.P99}, nil
}

// timeAtOnce keeps level pictures in the works for d: each of level
// goroutines analyses the picture and searches for its face, again and
// again, as the server does with that many requests in flight.
func timeAtOnce(ctx context.Context, eng cli.Engine, pic picture, ix *index.Index, level int, d time.Duration) (levelTimes, error) {
	ctx, stop := context.WithCancel(ctx)
	defer stop()
	var (
		wg     sync.WaitGroup
		once   sync.Once
		failed error
		times  = make([][]time.Duration, level)
	)
	start := time.Now()
	for w := range level {
		wg.Go(func() {
			for {
				mark := time.Now()
				_, vector, err := analyse(ctx, eng, pic)
				if err == nil {
					_, err = ix.Search(vector, 1)
				}
				if err != nil {
					// The first failure stops the others; what they
					// report after that is only the echo of it.
					once.Do(func() {
						failed = err
						stop()
					})
					return
				}
				times[w] = append(times[w], time.Since(mark))
				if time.Since(start) >= d {
					return
				}
			}
		})
	}
	wg.Wait()
	elapsed := time.Since(start)
	if failed != nil {
		return levelTimes{}, failed
	}

	all := summarise(slices.Concat(times...))
	pictures := 0
	for _, t := range times {
		pictures += len(t)
	}
	return levelTimes{
		Inflight: level, Gallery: ix.Stats().Subjects, Pictures: pictures,
		PerSecond: float64(pictures) / elapsed.Seconds(),
		P50:       all.P50, P99: all.P99,
	}, nil
}

// recommend picks the highest level whose 99th percentile stays within the
// target. MaxInflight is zero in its answer when no level does.
func recommend(levels []levelTimes, target time.Duration) recommendation {
	rec := recommendation{Target: millis(target)}
	for _, l := range levels {
		if l.P99 <= rec.Target && l.Inflight > rec.MaxInflight {
			rec.MaxInflight, rec.PerSecond, rec.P99 = l.Inflight, l.PerSecond, l.P99
		}
	}
	return rec
}

// summarise reduces the durations of one thing done many times to the few
// numbers the report shows.
func summarise(samples []time.Duration) spread {
	q := percentiles(samples, 50, 90, 99, 100)
	return spread{P50: millis(q[0]), P90: millis(q[1]), P99: millis(q[2]), Max: millis(q[3])}
}

// percentiles returns, for each p from 0 to 100, the p-th percentile of
// the samples by the nearest-rank method: the smallest sample that at
// least p in a hundred of them do not exceed. It works on a sorted copy
// and gives zeros when there are no samples.
func percentiles(samples []time.Duration, ps ...float64) []time.Duration {
	out := make([]time.Duration, len(ps))
	if len(samples) == 0 {
		return out
	}
	sorted := slices.Clone(samples)
	slices.Sort(sorted)
	for i, p := range ps {
		// Multiplying first keeps whole ranks whole: 0.07*100 is a little
		// more than 7, 7*100/100 is not.
		rank := int(math.Ceil(p * float64(len(sorted)) / 100))
		out[i] = sorted[min(max(rank, 1), len(sorted))-1]
	}
	return out
}
