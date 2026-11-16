package eval

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
)

// Person holds the vectors of one person's pictures, all of one length and
// of unit length, so that the dot product of two of them is their
// similarity.
type Person [][]float32

// table is every vector of a set of people in one block, one row each,
// with the person each row belongs to.
type table struct {
	dim    int
	values []float32
	owner  []int32
}

func (t *table) rows() int { return len(t.owner) }

func (t *table) row(i int) []float32 { return t.values[i*t.dim : (i+1)*t.dim] }

// tabulate lays the vectors out row by row, person after person.
func tabulate(people []Person) (*table, error) {
	t := &table{}
	rows := 0
	for _, p := range people {
		rows += len(p)
		if t.dim == 0 && len(p) > 0 {
			t.dim = len(p[0])
		}
	}
	if rows > 0 && t.dim == 0 {
		return nil, errors.New("eval: a vector with no values")
	}
	t.values = make([]float32, 0, rows*t.dim)
	t.owner = make([]int32, 0, rows)
	for i, p := range people {
		for k, v := range p {
			if len(v) != t.dim {
				return nil, fmt.Errorf("eval: picture %d of person %d has a vector of %d values, the others have %d", k+1, i+1, len(v), t.dim)
			}
			t.values = append(t.values, v...)
			t.owner = append(t.owner, int32(i))
		}
	}
	return t, nil
}

// Pairs compares every picture with every other one and counts the score
// under Genuine when the two show the same person and under Impostor when
// they do not. The work is spread over workers goroutines, one per CPU if
// workers is 0. It stops early, with the context's error, when ctx is done.
func Pairs(ctx context.Context, people []Person, workers int) (*Scores, error) {
	t, err := tabulate(people)
	if err != nil {
		return nil, err
	}
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	rows := t.rows()
	workers = max(min(workers, rows-1), 1)

	var (
		total = &Scores{}
		mu    sync.Mutex
		next  atomic.Int64
		wg    sync.WaitGroup
	)
	for range workers {
		wg.Go(func() {
			// Each worker counts on its own and hands over at the end;
			// a row is the unit of work, and the early rows are the
			// long ones.
			own := &Scores{}
			for ctx.Err() == nil {
				i := int(next.Add(1)) - 1
				if i >= rows-1 {
					break
				}
				a, person := t.row(i), t.owner[i]
				for j := i + 1; j < rows; j++ {
					score := dot(a, t.row(j))
					if t.owner[j] == person {
						own.Genuine.Add(score)
					} else {
						own.Impostor.Add(score)
					}
				}
			}
			mu.Lock()
			total.Genuine.Merge(&own.Genuine)
			total.Impostor.Merge(&own.Impostor)
			mu.Unlock()
		})
	}
	wg.Wait()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return total, nil
}

// dot is the dot product of two vectors of one length. Four separate sums
// let the processor overlap the multiply-adds.
func dot(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	b = b[:len(a)]
	n := len(a) &^ 3
	for i := 0; i < n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for i := n; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}
