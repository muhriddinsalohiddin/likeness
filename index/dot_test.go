package index

import (
	"math"
	"math/rand"
	"testing"
)

func randomRows(rng *rand.Rand, n, dim int) []float32 {
	v := make([]float32, n*dim)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}

// dot64 is the reference: plain accumulation in double precision.
func dot64(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

func TestDotRowsMatchesReference(t *testing.T) {
	t.Logf("kernel: %s", Kernel())
	rng := rand.New(rand.NewSource(1))
	for _, dim := range []int{1, 3, 4, 7, 16, 31, 32, 48, 64, 96, 128, 160, 512, 1024} {
		for _, n := range []int{1, 2, 5, 64, 257} {
			q := randomRows(rng, 1, dim)
			rows := randomRows(rng, n, dim)
			got := make([]float32, n+2)
			got[n], got[n+1] = -77, -78 // canaries: the kernel must stop at n
			dotRows(got[:n], q, rows)
			portable := make([]float32, n)
			dotRowsPortable(portable, q, rows)
			for i := 0; i < n; i++ {
				want := dot64(q, rows[i*dim:(i+1)*dim])
				tol := 1e-4 * (1 + math.Abs(want))
				if math.Abs(float64(got[i])-want) > tol {
					t.Fatalf("dim %d, row %d of %d: got %v, want %v", dim, i, n, got[i], want)
				}
				if math.Abs(float64(portable[i])-want) > tol {
					t.Fatalf("portable, dim %d, row %d: got %v, want %v", dim, i, portable[i], want)
				}
			}
			if got[n] != -77 || got[n+1] != -78 {
				t.Fatalf("dim %d, n %d: the kernel wrote past the end", dim, n)
			}
		}
	}
	dotRows(nil, []float32{1}, nil) // nothing to do
}

func TestDotRowsSpecialValues(t *testing.T) {
	const dim = 64
	q := make([]float32, dim)
	rows := make([]float32, 3*dim)
	for i := range q {
		q[i] = 1
		rows[i] = 1              // row 0: all ones -> 64
		rows[dim+i] = float32(i) // row 1: 0..63 -> 2016
		rows[2*dim+i] = -0.5     // row 2 -> -32
	}
	got := make([]float32, 3)
	dotRows(got, q, rows)
	if got[0] != 64 || got[1] != 2016 || got[2] != -32 {
		t.Fatalf("got %v", got)
	}
	// each lane of each accumulator is used exactly once per step
	for hot := 0; hot < dim; hot++ {
		one := make([]float32, dim)
		one[hot] = 3
		dotRows(got[:1], q, one)
		if got[0] != 3 {
			t.Fatalf("element %d is not picked up: %v", hot, got[0])
		}
	}
}

func BenchmarkDotRows(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	for _, c := range []struct {
		name   string
		dim, n int
	}{{"128x4096", 128, 4096}, {"512x4096", 512, 4096}} {
		q := randomRows(rng, 1, c.dim)
		rows := randomRows(rng, c.n, c.dim)
		dst := make([]float32, c.n)
		b.Run(Kernel()+"/"+c.name, func(b *testing.B) {
			b.SetBytes(int64(len(rows) * 4))
			for b.Loop() {
				dotRows(dst, q, rows)
			}
		})
		b.Run("portable/"+c.name, func(b *testing.B) {
			b.SetBytes(int64(len(rows) * 4))
			for b.Loop() {
				dotRowsPortable(dst, q, rows)
			}
		})
	}
}
