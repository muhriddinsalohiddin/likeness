package index

import (
	"fmt"
	"math/rand"
	"testing"
)

func filled(b *testing.B, dim, rows int, o Options) (*Index, [][]float32) {
	b.Helper()
	rng := rand.New(rand.NewSource(1))
	ix := mustNew(b, dim, o)
	for i := 0; i < rows; i++ {
		if err := ix.Add(fmt.Sprintf("s%d", i), 1, unit(rng, dim)); err != nil {
			b.Fatal(err)
		}
	}
	queries := make([][]float32, 64)
	for i := range queries {
		queries[i] = unit(rng, dim)
	}
	return ix, queries
}

// One search at a time, as many cores as it wants.
func BenchmarkSearch(b *testing.B) {
	for _, c := range []struct{ rows, dim int }{{7_000, 128}, {7_000, 512}, {100_000, 128}, {100_000, 512}} {
		for _, workers := range []int{1, 0} {
			name := fmt.Sprintf("%dx%d/workers=%d", c.rows, c.dim, workers)
			if workers == 0 {
				name = fmt.Sprintf("%dx%d/workers=all", c.rows, c.dim)
			}
			b.Run(name, func(b *testing.B) {
				ix, queries := filled(b, c.dim, c.rows, Options{Workers: workers})
				b.ResetTimer()
				i := 0
				for b.Loop() {
					if _, err := ix.Search(queries[i%len(queries)], 5); err != nil {
						b.Fatal(err)
					}
					i++
				}
			})
		}
	}
}

// Many searches at once, each on one core: what a busy server does.
func BenchmarkSearchParallel(b *testing.B) {
	for _, c := range []struct{ rows, dim int }{{7_000, 128}, {100_000, 128}, {100_000, 512}} {
		b.Run(fmt.Sprintf("%dx%d", c.rows, c.dim), func(b *testing.B) {
			ix, queries := filled(b, c.dim, c.rows, Options{Workers: 1})
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				i := 0
				for pb.Next() {
					if _, err := ix.Search(queries[i%len(queries)], 5); err != nil {
						b.Error(err)
						return
					}
					i++
				}
			})
		})
	}
}

func BenchmarkCompare(b *testing.B) {
	ix, queries := filled(b, 512, 10_000, Options{})
	b.ResetTimer()
	i := 0
	for b.Loop() {
		if _, ok, err := ix.Compare("s4242", queries[i%len(queries)]); err != nil || !ok {
			b.Fatal(ok, err)
		}
		i++
	}
}

func BenchmarkAdd(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	vecs := make([][]float32, 1024)
	for i := range vecs {
		vecs[i] = unit(rng, 512)
	}
	ix := mustNew(b, 512, Options{})
	b.ResetTimer()
	i := 0
	for b.Loop() {
		if err := ix.Add(fmt.Sprintf("s%d", i), 1, vecs[i%len(vecs)]); err != nil {
			b.Fatal(err)
		}
		i++
	}
}
