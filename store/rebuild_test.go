package store_test

import (
	"bytes"
	"context"
	"fmt"
	"math/rand"
	"path/filepath"
	"testing"
	"time"

	"github.com/muhriddinsalohiddin/likeness/index"
	"github.com/muhriddinsalohiddin/likeness/store"
)

func randomVector(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}

// rebuild fills a fresh index from the store, the way a server does when it
// starts.
func rebuild(t testing.TB, s store.Store, o index.Options) *index.Index {
	t.Helper()
	ix, err := index.New(s.Info().Dim, o)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(ix.Close)
	if err := s.Load(context.Background(), func(tm store.Template) error {
		return ix.Add(tm.Subject, tm.ID, tm.Vector)
	}); err != nil {
		t.Fatal(err)
	}
	return ix
}

// A server keeps an index next to the store and applies every change to
// both. After a restart, the index rebuilt from the store must answer
// exactly as the old one did.
func TestIndexRebuiltFromStoreAnswersTheSame(t *testing.T) {
	const dim = 64
	ctx := context.Background()
	for name, key := range map[string][]byte{"plain": nil, "encrypted": bytes.Repeat([]byte{5}, store.KeySize)} {
		t.Run(name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(1))
			path := filepath.Join(t.TempDir(), "faces.db")
			o := store.Options{Dim: dim, Fingerprint: "test:model:1", Key: key}
			s, err := store.Open(path, o)
			if err != nil {
				t.Fatal(err)
			}
			live, err := index.New(dim, index.Options{OnHeap: true})
			if err != nil {
				t.Fatal(err)
			}
			defer live.Close()

			type enrolled struct {
				subject string
				id      uint64
			}
			var all []enrolled
			for batch := 0; batch < 9; batch++ {
				entries := make([]store.Entry, 100)
				for i := range entries {
					entries[i] = store.Entry{Subject: fmt.Sprintf("p%03d", (batch*100+i)%400), Vector: randomVector(rng, dim)}
				}
				added, err := s.AddMany(ctx, entries)
				if err != nil {
					t.Fatal(err)
				}
				for i, tm := range added {
					if err := live.Add(tm.Subject, tm.ID, entries[i].Vector); err != nil {
						t.Fatal(err)
					}
					all = append(all, enrolled{tm.Subject, tm.ID})
				}
			}
			for i, e := range all {
				switch {
				case i%11 == 0:
					if ok, err := s.Remove(ctx, e.subject, e.id); err != nil || ok != live.Remove(e.subject, e.id) {
						t.Fatalf("store and index disagree about removing %v: %v", e, err)
					}
				case i%97 == 0:
					n, err := s.RemoveSubject(ctx, e.subject)
					if err != nil || n != live.RemoveSubject(e.subject) {
						t.Fatalf("store and index disagree about removing %s: %v", e.subject, err)
					}
				}
			}
			if err := s.Close(); err != nil {
				t.Fatal(err)
			}

			s, err = store.Open(path, o)
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			restarted := rebuild(t, s, index.Options{})

			subjects, templates, _ := s.Count(ctx)
			if a, b := live.Stats(), restarted.Stats(); a.Subjects != b.Subjects || a.Templates != b.Templates || b.Subjects != subjects || b.Templates != templates {
				t.Fatalf("before the restart %d subjects / %d templates, after %d / %d, store %d / %d",
					a.Subjects, a.Templates, b.Subjects, b.Templates, subjects, templates)
			}
			for trial := 0; trial < 50; trial++ {
				q := randomVector(rng, dim)
				before, err := live.Search(q, 10)
				if err != nil {
					t.Fatal(err)
				}
				after, err := restarted.Search(q, 10)
				if err != nil {
					t.Fatal(err)
				}
				if len(before) != 10 || len(after) != 10 {
					t.Fatalf("%d and %d matches", len(before), len(after))
				}
				for i := range before {
					if before[i] != after[i] {
						t.Fatalf("trial %d, position %d: %+v before the restart, %+v after", trial, i, before[i], after[i])
					}
				}
			}
		})
	}
}

// How long a server takes to come up: reading every template and building
// the index.
func BenchmarkRebuild(b *testing.B) {
	ctx := context.Background()
	for _, c := range []struct {
		rows, dim int
		key       []byte
	}{
		{7_000, 128, nil},
		{100_000, 128, nil},
		{100_000, 128, bytes.Repeat([]byte{5}, store.KeySize)},
		{20_000, 512, nil},
	} {
		name := fmt.Sprintf("%dx%d", c.rows, c.dim)
		if c.key != nil {
			name += "/encrypted"
		}
		b.Run(name, func(b *testing.B) {
			rng := rand.New(rand.NewSource(1))
			s, err := store.Open(filepath.Join(b.TempDir(), "faces.db"), store.Options{Dim: c.dim, Fingerprint: "bench", Key: c.key})
			if err != nil {
				b.Fatal(err)
			}
			defer s.Close()
			entries := make([]store.Entry, 0, 5000)
			for i := 0; i < c.rows; i++ {
				entries = append(entries, store.Entry{Subject: fmt.Sprintf("p%06d", i), Vector: randomVector(rng, c.dim)})
				if len(entries) == cap(entries) || i == c.rows-1 {
					if _, err := s.AddMany(ctx, entries); err != nil {
						b.Fatal(err)
					}
					entries = entries[:0]
				}
			}
			b.ResetTimer()
			var took time.Duration
			for b.Loop() {
				start := time.Now()
				ix := rebuild(b, s, index.Options{})
				took += time.Since(start)
				if st := ix.Stats(); st.Templates != c.rows {
					b.Fatalf("%d templates", st.Templates)
				}
				ix.Close()
			}
			b.ReportMetric(float64(took.Milliseconds())/float64(b.N), "ms/rebuild")
		})
	}
}
