package index

import (
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
)

// chunkRows is how many rows are scored at a time. The scores of one chunk
// fit comfortably in the first-level cache.
const chunkRows = 1024

// One worker is worth starting for about this many values to scan: a
// fraction of a millisecond of work.
const valuesPerWorker = 1 << 20

type hit struct {
	slot  uint32
	row   int32
	score float32
}

// best keeps the k highest-scoring subjects seen so far, one entry each.
type best struct {
	hits  []hit
	k     int
	full  bool
	min   float32 // the lowest score held, once full
	minAt int
}

func (b *best) offer(h hit) {
	for i := range b.hits {
		if b.hits[i].slot != h.slot {
			continue
		}
		// another template of a subject already listed
		if h.score > b.hits[i].score {
			b.hits[i] = h
			if b.full && i == b.minAt {
				b.findMin()
			}
		}
		return
	}
	if !b.full {
		b.hits = append(b.hits, h)
		if len(b.hits) == b.k {
			b.full = true
			b.findMin()
		}
		return
	}
	if h.score > b.min {
		b.hits[b.minAt] = h
		b.findMin()
	}
}

func (b *best) findMin() {
	b.minAt, b.min = 0, b.hits[0].score
	for i, h := range b.hits[1:] {
		if h.score < b.min {
			b.minAt, b.min = i+1, h.score
		}
	}
}

// scan scores rows lo..hi of the snapshot against q.
func (ix *Index) scan(s *snapshot, q []float32, lo, hi, k int) []hit {
	buf := ix.scores.Get().(*[]float32)
	defer ix.scores.Put(buf)
	top := best{k: k, hits: make([]hit, 0, min(k, 16))}
	for lo < hi {
		n := min(chunkRows, hi-lo)
		scores := (*buf)[:n]
		dotRows(scores, q, s.b.vec[lo*ix.dim:(lo+n)*ix.dim])
		for i, score := range scores {
			// Once k subjects are held, a score that does not beat the
			// lowest of them changes nothing, whoever it belongs to.
			if top.full && score <= top.min {
				continue
			}
			owner := atomic.LoadUint32(&s.b.owner[lo+i])
			if owner == dead {
				continue
			}
			top.offer(hit{owner, int32(lo + i), score})
		}
		lo += n
	}
	return top.hits
}

// Search returns up to limit subjects, most similar first. Ties go to the
// template enrolled earlier.
func (ix *Index) Search(query []float32, limit int) ([]Match, error) {
	if len(query) != ix.dim {
		return nil, fmt.Errorf("%w: %d values, want %d", ErrDimension, len(query), ix.dim)
	}
	scale, err := unitScale(query)
	if err != nil {
		return nil, err
	}
	qbuf := ix.queries.Get().(*[]float32)
	defer ix.queries.Put(qbuf)
	q := *qbuf
	for i, v := range query {
		q[i] = v * scale
	}

	ix.life.RLock()
	defer ix.life.RUnlock()
	s := ix.snap.Load()
	if s == nil {
		return nil, ErrClosed
	}
	if limit <= 0 || s.rows == 0 {
		return nil, nil
	}

	workers := min(max(s.rows*ix.dim/valuesPerWorker, 1), ix.workers)
	var hits []hit
	if workers == 1 {
		hits = ix.scan(s, q, 0, s.rows, limit)
	} else {
		parts := make([][]hit, workers)
		step := (s.rows + workers - 1) / workers
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			lo, hi := w*step, min((w+1)*step, s.rows)
			if lo >= hi {
				break
			}
			wg.Add(1)
			go func() {
				defer wg.Done()
				parts[w] = ix.scan(s, q, lo, hi, limit)
			}()
		}
		wg.Wait()
		// a subject's templates may have landed in different parts
		merged := best{k: limit}
		for _, part := range parts {
			for _, h := range part {
				merged.offer(h)
			}
		}
		hits = merged.hits
	}

	sort.Slice(hits, func(a, b int) bool {
		if hits[a].score != hits[b].score {
			return hits[a].score > hits[b].score
		}
		return hits[a].row < hits[b].row
	})
	matches := make([]Match, len(hits))
	for i, h := range hits {
		matches[i] = Match{Subject: s.names[h.slot], Score: h.score, Template: s.b.tmpl[h.row]}
	}
	return matches, nil
}

// Compare scores the query against one subject's templates and returns the
// best. The boolean is false when the subject is not enrolled.
func (ix *Index) Compare(subject string, query []float32) (Match, bool, error) {
	if len(query) != ix.dim {
		return Match{}, false, fmt.Errorf("%w: %d values, want %d", ErrDimension, len(query), ix.dim)
	}
	scale, err := unitScale(query)
	if err != nil {
		return Match{}, false, err
	}

	ix.life.RLock()
	defer ix.life.RUnlock()
	s := ix.snap.Load()
	if s == nil {
		return Match{}, false, ErrClosed
	}
	ix.smu.RLock()
	defer ix.smu.RUnlock()
	si, ok := ix.byID[subject]
	if !ok {
		return Match{}, false, nil
	}
	m := Match{Subject: subject, Score: -2}
	for _, r := range ix.slots[si].rows {
		if score := dot(query, s.b.vec[int(r)*ix.dim:(int(r)+1)*ix.dim]) * scale; score > m.Score {
			m.Score, m.Template = score, s.b.tmpl[r]
		}
	}
	return m, true, nil
}
