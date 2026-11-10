package main

import (
	"math/rand"
	"sync"
)

// Index is an exact (brute-force) cosine index: one contiguous row-major
// matrix of L2-normalised vectors, so similarity is a plain dot product.
type Index struct {
	n, d int
	data []float32
}

type Hit struct {
	ID    int
	Score float32
}

func randomIndex(n, d int, seed int64) *Index {
	rng := rand.New(rand.NewSource(seed))
	ix := &Index{n: n, d: d, data: make([]float32, n*d)}
	for i := 0; i < n; i++ {
		row := ix.data[i*d : (i+1)*d]
		for j := range row {
			row[j] = float32(rng.NormFloat64())
		}
		l2normalize(row)
	}
	return ix
}

// dotGo is the portable baseline: 4 independent accumulators let the CPU run
// the multiply-adds out of order, but the Go compiler emits no SIMD here.
func dotGo(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	n := len(a) &^ 3
	b = b[:len(a)]
	for i := 0; i < n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for i := n; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return s0 + s1 + s2 + s3
}

func (ix *Index) scan(q []float32, lo, hi, k int, dot func(a, b []float32) float32) []Hit {
	top := make([]Hit, 0, k+1)
	for i := lo; i < hi; i++ {
		s := dot(q, ix.data[i*ix.d:(i+1)*ix.d])
		if len(top) == k && s <= top[k-1].Score {
			continue
		}
		j := len(top)
		top = append(top, Hit{})
		for j > 0 && top[j-1].Score < s {
			top[j] = top[j-1]
			j--
		}
		top[j] = Hit{i, s}
		if len(top) > k {
			top = top[:k]
		}
	}
	return top
}

// Search returns the k best matches, splitting the scan across workers.
func (ix *Index) Search(q []float32, k, workers int, dot func(a, b []float32) float32) []Hit {
	if workers <= 1 || ix.n < 4096 {
		return ix.scan(q, 0, ix.n, k, dot)
	}
	parts := make([][]Hit, workers)
	var wg sync.WaitGroup
	step := (ix.n + workers - 1) / workers
	for w := 0; w < workers; w++ {
		lo, hi := w*step, min((w+1)*step, ix.n)
		if lo >= hi {
			break
		}
		wg.Add(1)
		go func() { defer wg.Done(); parts[w] = ix.scan(q, lo, hi, k, dot) }()
	}
	wg.Wait()
	var all []Hit
	for _, p := range parts {
		all = append(all, p...)
	}
	// merge: small, insertion sort is enough
	for i := 1; i < len(all); i++ {
		for j := i; j > 0 && all[j-1].Score < all[j].Score; j-- {
			all[j-1], all[j] = all[j], all[j-1]
		}
	}
	if len(all) > k {
		all = all[:k]
	}
	return all
}
