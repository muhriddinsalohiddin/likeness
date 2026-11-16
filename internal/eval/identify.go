package eval

import (
	"context"
	"fmt"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/muhriddinsalohiddin/likeness/index"
)

// Identification is the outcome of searching a gallery with pictures of
// people who are in it (mated probes) and of people who are not (non-mated
// probes).
type Identification struct {
	// Gallery is the number of people enrolled, LeftOut the number kept
	// out of the gallery to act as strangers.
	Gallery, LeftOut int
	// Mated and NonMated are the numbers of probes of each kind.
	Mated, NonMated int

	// found holds the top scores of the mated probes whose own person
	// came out on top; raised, those of the non-mated probes, for whom
	// any name is a false one.
	found, raised Histogram
}

// Rank1 is the share of mated probes whose own person was the most similar
// in the gallery, whatever the score.
func (id *Identification) Rank1() float64 {
	return share(id.found.n, uint64(id.Mated))
}

// FNIR, the false negative identification rate, is the share of mated
// probes that a search with threshold t does not answer with their own
// person: somebody else came out on top, or the score fell short.
func (id *Identification) FNIR(t float64) float64 {
	return share(uint64(id.Mated)-id.found.AtOrAbove(t), uint64(id.Mated))
}

// FPIR, the false positive identification rate, is the share of non-mated
// probes for which a search with threshold t names somebody.
func (id *Identification) FPIR(t float64) float64 {
	return share(id.raised.AtOrAbove(t), uint64(id.NonMated))
}

type probe struct {
	vector []float32
	// subject is the probe's own id in the gallery, empty for a stranger.
	subject string
}

// Identify runs the searches a server would. The first picture of every
// person is enrolled and their other pictures are mated probes; every
// holdout-th person is left out of the gallery instead, and all of their
// pictures are non-mated probes. A holdout of 0 leaves nobody out. People
// without vectors take no part. Each probe is one search for the most
// similar person; the searches are spread over workers goroutines, one per
// CPU if workers is 0.
func Identify(ctx context.Context, people []Person, holdout, workers int) (*Identification, error) {
	id := &Identification{}
	dim := 0
	for _, p := range people {
		if len(p) > 0 {
			dim = len(p[0])
			break
		}
	}
	if dim == 0 {
		return id, nil
	}

	var enrolled []Person
	var probes []probe
	taking := 0
	for _, p := range people {
		if len(p) == 0 {
			continue
		}
		taking++
		if holdout > 0 && taking%holdout == 0 {
			id.LeftOut++
			for _, v := range p {
				probes = append(probes, probe{vector: v})
			}
			continue
		}
		subject := strconv.Itoa(len(enrolled))
		enrolled = append(enrolled, p)
		for _, v := range p[1:] {
			probes = append(probes, probe{vector: v, subject: subject})
		}
	}
	id.Gallery = len(enrolled)

	ix, err := index.New(dim, index.Options{Capacity: max(len(enrolled), 1)})
	if err != nil {
		return nil, fmt.Errorf("eval: %w", err)
	}
	defer ix.Close()
	for i, p := range enrolled {
		if err := ix.Add(strconv.Itoa(i), 1, p[0]); err != nil {
			return nil, fmt.Errorf("eval: enrolling person %d of the gallery: %w", i+1, err)
		}
	}

	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	var (
		mu     sync.Mutex
		failed error
		next   atomic.Int64
		wg     sync.WaitGroup
	)
	for range max(min(workers, len(probes)), 1) {
		wg.Go(func() {
			own := &Identification{}
			var err error
			for ctx.Err() == nil {
				i := int(next.Add(1)) - 1
				if i >= len(probes) {
					break
				}
				if err = own.search(ix, probes[i]); err != nil {
					next.Store(int64(len(probes))) // the others need not go on
					break
				}
			}
			mu.Lock()
			defer mu.Unlock()
			id.Mated += own.Mated
			id.NonMated += own.NonMated
			id.found.Merge(&own.found)
			id.raised.Merge(&own.raised)
			if failed == nil {
				failed = err
			}
		})
	}
	wg.Wait()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if failed != nil {
		return nil, fmt.Errorf("eval: %w", failed)
	}
	return id, nil
}

// search looks one probe up and counts how it went.
func (id *Identification) search(ix *index.Index, p probe) error {
	top, err := ix.Search(p.vector, 1)
	if err != nil {
		return err
	}
	if p.subject == "" {
		id.NonMated++
		if len(top) > 0 {
			id.raised.Add(top[0].Score)
		}
		return nil
	}
	id.Mated++
	if len(top) > 0 && top[0].Subject == p.subject {
		id.found.Add(top[0].Score)
	}
	return nil
}
