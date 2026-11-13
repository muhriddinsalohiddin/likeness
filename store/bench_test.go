package store

import (
	"fmt"
	mrand "math/rand"
	"testing"
)

// One enrolment: a transaction and a flush to disk.
func BenchmarkAdd(b *testing.B) {
	rng := mrand.New(mrand.NewSource(1))
	s, _ := newStore(b, Options{Dim: 128, Fingerprint: "bench"})
	v := vector(rng, 128)
	i := 0
	for b.Loop() {
		if _, err := s.Add(ctx, fmt.Sprintf("s%d", i), v, nil); err != nil {
			b.Fatal(err)
		}
		i++
	}
}

// A bulk import, a thousand templates to the transaction.
func BenchmarkAddMany(b *testing.B) {
	rng := mrand.New(mrand.NewSource(1))
	s, _ := newStore(b, Options{Dim: 128, Fingerprint: "bench"})
	entries := make([]Entry, 1000)
	n := 0
	for b.Loop() {
		for i := range entries {
			entries[i] = Entry{Subject: fmt.Sprintf("s%d", n), Vector: vector(rng, 128)}
			n++
		}
		if _, err := s.AddMany(ctx, entries); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(b.Elapsed().Microseconds())/float64(n), "µs/template")
}
