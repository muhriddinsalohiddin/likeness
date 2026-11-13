package index

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
)

// storage runs a test against both kinds of memory. The heap variant is the
// one the race detector can see into.
func storage(t *testing.T, fn func(t *testing.T, o Options)) {
	t.Run("mapped", func(t *testing.T) {
		if !canMap {
			t.Skip("no anonymous mappings on this platform")
		}
		fn(t, Options{})
	})
	t.Run("heap", func(t *testing.T) { fn(t, Options{OnHeap: true}) })
}

func mustNew(t testing.TB, dim int, o Options) *Index {
	t.Helper()
	ix, err := New(dim, o)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(ix.Close)
	return ix
}

func unit(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	var sum float64
	for i := range v {
		v[i] = float32(rng.NormFloat64())
		sum += float64(v[i]) * float64(v[i])
	}
	for i := range v {
		v[i] /= float32(math.Sqrt(sum))
	}
	return v
}

// near returns a unit vector at roughly the given cosine from v.
func near(rng *rand.Rand, v []float32, cos float64) []float32 {
	noise := unit(rng, len(v))
	out := make([]float32, len(v))
	var sum float64
	for i := range v {
		out[i] = float32(cos)*v[i] + float32(math.Sqrt(1-cos*cos))*noise[i]
		sum += float64(out[i]) * float64(out[i])
	}
	for i := range out {
		out[i] /= float32(math.Sqrt(sum))
	}
	return out
}

// oracle is the index written the obvious way.
type oracle struct {
	dim  int
	rows []oracleRow
}

type oracleRow struct {
	subject  string
	template uint64
	vec      []float32
}

func (o *oracle) add(subject string, template uint64, vec []float32) {
	o.rows = append(o.rows, oracleRow{subject, template, vec})
}

func (o *oracle) remove(subject string, template uint64, all bool) {
	kept := o.rows[:0]
	for _, r := range o.rows {
		if r.subject == subject && (all || r.template == template) {
			continue
		}
		kept = append(kept, r)
	}
	o.rows = kept
}

type oracleMatch struct {
	subject  string
	template uint64
	score    float64
}

func (o *oracle) search(q []float32) []oracleMatch {
	bestOf := map[string]oracleMatch{}
	for _, r := range o.rows {
		s := dot64(q, r.vec)
		if m, ok := bestOf[r.subject]; !ok || s > m.score {
			bestOf[r.subject] = oracleMatch{r.subject, r.template, s}
		}
	}
	out := make([]oracleMatch, 0, len(bestOf))
	for _, m := range bestOf {
		out = append(out, m)
	}
	sort.Slice(out, func(a, b int) bool { return out[a].score > out[b].score })
	return out
}

// agree checks a search result against the oracle's, allowing two subjects
// whose scores are within rounding of each other to swap places.
func agree(t *testing.T, got []Match, want []oracleMatch, limit int) {
	t.Helper()
	const tol = 2e-5
	n := min(limit, len(want))
	if len(got) != n {
		t.Fatalf("got %d matches, want %d", len(got), n)
	}
	scoreOf := map[string]oracleMatch{}
	for _, m := range want {
		scoreOf[m.subject] = m
	}
	seen := map[string]bool{}
	for i, g := range got {
		w, ok := scoreOf[g.Subject]
		if !ok {
			t.Fatalf("match %d: subject %q is not enrolled", i, g.Subject)
		}
		if seen[g.Subject] {
			t.Fatalf("subject %q reported twice", g.Subject)
		}
		seen[g.Subject] = true
		if math.Abs(float64(g.Score)-w.score) > tol {
			t.Fatalf("match %d (%s): score %v, want %v", i, g.Subject, g.Score, w.score)
		}
		if math.Abs(float64(g.Score)-want[i].score) > tol {
			t.Fatalf("match %d: score %v, but the %dth best subject scores %v", i, g.Score, i+1, want[i].score)
		}
		if i > 0 && g.Score > got[i-1].Score {
			t.Fatalf("matches are not sorted at %d", i)
		}
	}
	// nothing better was left out
	if n < len(want) && n > 0 && want[n].score > float64(got[n-1].Score)+tol {
		t.Fatalf("subject %q (%.5f) is missing from the top %d", want[n].subject, want[n].score, n)
	}
}

func TestSearchIsExact(t *testing.T) {
	storage(t, func(t *testing.T, o Options) {
		for _, dim := range []int{128, 512, 48} { // 48 takes the portable path on every machine
			rng := rand.New(rand.NewSource(int64(dim)))
			ix := mustNew(t, dim, o)
			ref := &oracle{dim: dim}
			var bases [][]float32
			for s := 0; s < 700; s++ {
				base := unit(rng, dim)
				bases = append(bases, base)
				id := fmt.Sprintf("s%04d", s)
				for tmpl := 0; tmpl < 1+s%3; tmpl++ {
					v := near(rng, base, 0.8)
					if err := ix.Add(id, uint64(tmpl+1), v); err != nil {
						t.Fatal(err)
					}
					ref.add(id, uint64(tmpl+1), v)
				}
			}
			if st := ix.Stats(); st.Subjects != 700 || st.Templates != len(ref.rows) || st.Dead != 0 || st.Dim != dim {
				t.Fatalf("stats %+v, want 700 subjects and %d templates", st, len(ref.rows))
			}
			for trial := 0; trial < 60; trial++ {
				q := unit(rng, dim)
				if trial%2 == 0 {
					q = near(rng, bases[rng.Intn(len(bases))], 0.7) // someone who is enrolled
				}
				for _, limit := range []int{1, 5, 40} {
					got, err := ix.Search(q, limit)
					if err != nil {
						t.Fatal(err)
					}
					agree(t, got, ref.search(q), limit)
				}
			}
		}
	})
}

func TestBestTemplateRepresentsTheSubject(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	ix := mustNew(t, 64, Options{OnHeap: true})
	q := unit(rng, 64)
	for i, cos := range []float64{0.2, 0.9, 0.5} {
		if err := ix.Add("anna", uint64(10+i), near(rng, q, cos)); err != nil {
			t.Fatal(err)
		}
	}
	if err := ix.Add("boris", 1, near(rng, q, 0.7)); err != nil {
		t.Fatal(err)
	}
	got, err := ix.Search(q, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Subject != "anna" || got[0].Template != 11 || got[1].Subject != "boris" {
		t.Fatalf("got %+v", got)
	}
	if math.Abs(float64(got[0].Score)-0.9) > 0.05 || math.Abs(float64(got[1].Score)-0.7) > 0.05 {
		t.Fatalf("scores %v and %v", got[0].Score, got[1].Score)
	}

	m, ok, err := ix.Compare("anna", q)
	if err != nil || !ok || m.Template != 11 || m.Subject != "anna" || math.Abs(float64(m.Score-got[0].Score)) > 1e-5 {
		t.Fatalf("Compare: %+v %v %v", m, ok, err)
	}
	if _, ok, err := ix.Compare("nobody", q); ok || err != nil {
		t.Fatalf("Compare with an unknown subject: %v %v", ok, err)
	}
	if ids := ix.Templates("anna"); len(ids) != 3 || ids[0] != 10 || ids[2] != 12 {
		t.Fatalf("templates %v", ids)
	}
	if ix.Templates("nobody") != nil || !ix.Has("boris") || ix.Has("nobody") {
		t.Fatal("subject lookups are off")
	}
}

func TestVectorsAreScaledToUnitLength(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	ix := mustNew(t, 32, Options{OnHeap: true})
	v := unit(rng, 32)
	long := make([]float32, 32)
	for i := range v {
		long[i] = v[i] * 37.5
	}
	if err := ix.Add("x", 1, long); err != nil {
		t.Fatal(err)
	}
	short := make([]float32, 32)
	for i := range v {
		short[i] = v[i] * 0.01
	}
	got, err := ix.Search(short, 1)
	if err != nil || len(got) != 1 || math.Abs(float64(got[0].Score)-1) > 1e-5 {
		t.Fatalf("got %+v %v", got, err)
	}
	if m, _, _ := ix.Compare("x", short); math.Abs(float64(m.Score)-1) > 1e-5 {
		t.Fatalf("Compare score %v", m.Score)
	}
	if long[0] != v[0]*37.5 {
		t.Fatal("Add changed the caller's vector")
	}
}

func TestBadInput(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	ix := mustNew(t, 16, Options{OnHeap: true})
	good := unit(rng, 16)
	nan := append([]float32(nil), good...)
	nan[3] = float32(math.NaN())
	inf := append([]float32(nil), good...)
	inf[0] = float32(math.Inf(1))
	for name, c := range map[string]struct {
		subject string
		vec     []float32
		want    error
	}{
		"empty subject": {"", good, ErrSubject},
		"short vector":  {"a", good[:15], ErrDimension},
		"long vector":   {"a", append(good, 1), ErrDimension},
		"zero vector":   {"a", make([]float32, 16), ErrVector},
		"NaN":           {"a", nan, ErrVector},
		"infinity":      {"a", inf, ErrVector},
	} {
		if err := ix.Add(c.subject, 1, c.vec); !errors.Is(err, c.want) {
			t.Errorf("Add, %s: got %v, want %v", name, err, c.want)
		}
		if c.subject == "" {
			continue
		}
		if _, err := ix.Search(c.vec, 3); !errors.Is(err, c.want) {
			t.Errorf("Search, %s: got %v, want %v", name, err, c.want)
		}
		if _, _, err := ix.Compare("a", c.vec); !errors.Is(err, c.want) {
			t.Errorf("Compare, %s: got %v, want %v", name, err, c.want)
		}
	}
	if st := ix.Stats(); st.Templates != 0 || st.Subjects != 0 {
		t.Fatalf("rejected input left something behind: %+v", st)
	}

	if err := ix.Add("a", 7, good); err != nil {
		t.Fatal(err)
	}
	if err := ix.Add("a", 7, unit(rng, 16)); !errors.Is(err, ErrExists) {
		t.Fatalf("the same template id twice: %v", err)
	}
	if err := ix.Add("b", 7, good); err != nil {
		t.Fatalf("template ids belong to their subject: %v", err)
	}

	for _, bad := range []struct {
		dim int
		o   Options
	}{{0, Options{}}, {-4, Options{}}, {8, Options{Capacity: -1}}, {8, Options{Workers: -1}}} {
		if _, err := New(bad.dim, bad.o); err == nil {
			t.Errorf("New(%d, %+v) should fail", bad.dim, bad.o)
		}
	}
}

func TestEmptyIndexAndLimits(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	ix := mustNew(t, 32, Options{OnHeap: true})
	q := unit(rng, 32)
	if got, err := ix.Search(q, 5); err != nil || got != nil {
		t.Fatalf("empty index: %v %v", got, err)
	}
	for i := 0; i < 5; i++ {
		if err := ix.Add(fmt.Sprintf("s%d", i), 1, near(rng, q, 0.1+0.2*float64(i))); err != nil {
			t.Fatal(err)
		}
	}
	if got, _ := ix.Search(q, 0); got != nil {
		t.Fatalf("limit 0: %v", got)
	}
	if got, _ := ix.Search(q, -3); got != nil {
		t.Fatalf("negative limit: %v", got)
	}
	got, err := ix.Search(q, 100)
	if err != nil || len(got) != 5 {
		t.Fatalf("limit beyond the population: %d matches, %v", len(got), err)
	}
	for i, m := range got {
		if want := fmt.Sprintf("s%d", 4-i); m.Subject != want {
			t.Fatalf("position %d: %s, want %s", i, m.Subject, want)
		}
	}
}

func TestTiesGoToTheEarlierTemplate(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	ix := mustNew(t, 32, Options{OnHeap: true})
	v := unit(rng, 32)
	for _, id := range []string{"first", "second", "third"} {
		if err := ix.Add(id, 1, v); err != nil {
			t.Fatal(err)
		}
	}
	for round := 0; round < 5; round++ {
		got, _ := ix.Search(v, 3)
		if len(got) != 3 || got[0].Subject != "first" || got[1].Subject != "second" || got[2].Subject != "third" {
			t.Fatalf("round %d: %+v", round, got)
		}
	}
}

func TestRemove(t *testing.T) {
	storage(t, func(t *testing.T, o Options) {
		rng := rand.New(rand.NewSource(8))
		ix := mustNew(t, 64, o)
		q := unit(rng, 64)
		must := func(err error) {
			t.Helper()
			if err != nil {
				t.Fatal(err)
			}
		}
		must(ix.Add("anna", 1, near(rng, q, 0.9)))
		must(ix.Add("anna", 2, near(rng, q, 0.6)))
		must(ix.Add("boris", 1, near(rng, q, 0.75)))
		top := func() Match {
			t.Helper()
			got, err := ix.Search(q, 1)
			if err != nil || len(got) != 1 {
				t.Fatalf("%v %v", got, err)
			}
			return got[0]
		}

		if m := top(); m.Subject != "anna" || m.Template != 1 {
			t.Fatalf("%+v", m)
		}
		if !ix.Remove("anna", 1) || ix.Remove("anna", 1) || ix.Remove("anna", 99) || ix.Remove("nobody", 1) {
			t.Fatal("Remove reported the wrong outcome")
		}
		// anna is still enrolled, by her weaker template, and boris now leads
		if m := top(); m.Subject != "boris" {
			t.Fatalf("%+v", m)
		}
		if m, ok, _ := ix.Compare("anna", q); !ok || m.Template != 2 {
			t.Fatalf("%+v %v", m, ok)
		}
		if st := ix.Stats(); st.Subjects != 2 || st.Templates != 2 || st.Dead != 1 {
			t.Fatalf("%+v", st)
		}

		if n := ix.RemoveSubject("boris"); n != 1 {
			t.Fatalf("removed %d templates", n)
		}
		if ix.RemoveSubject("boris") != 0 || ix.Has("boris") {
			t.Fatal("boris should be gone")
		}
		if m := top(); m.Subject != "anna" || m.Template != 2 {
			t.Fatalf("%+v", m)
		}
		if !ix.Remove("anna", 2) || ix.Has("anna") {
			t.Fatal("removing the last template removes the subject")
		}
		if got, _ := ix.Search(q, 5); len(got) != 0 {
			t.Fatalf("an emptied index answered %+v", got)
		}

		// a removed subject can come back, under the same template id
		must(ix.Add("anna", 2, near(rng, q, 0.8)))
		if m := top(); m.Subject != "anna" || m.Template != 2 || m.Score < 0.7 {
			t.Fatalf("%+v", m)
		}
		if st := ix.Stats(); st.Subjects != 1 || st.Templates != 1 || st.Dead != 3 {
			t.Fatalf("%+v", st)
		}
		must(ix.Compact())
		if st := ix.Stats(); st.Subjects != 1 || st.Templates != 1 || st.Dead != 0 {
			t.Fatalf("after compaction: %+v", st)
		}
		if m := top(); m.Subject != "anna" || m.Template != 2 {
			t.Fatalf("after compaction: %+v", m)
		}
		must(ix.Compact()) // nothing to do
	})
}

func TestGrowth(t *testing.T) {
	storage(t, func(t *testing.T, o Options) {
		o.Capacity = 4
		rng := rand.New(rand.NewSource(9))
		ix := mustNew(t, 32, o)
		ref := &oracle{dim: 32}
		for i := 0; i < 1500; i++ {
			v := unit(rng, 32)
			id := fmt.Sprintf("s%d", i/2)
			if err := ix.Add(id, uint64(i), v); err != nil {
				t.Fatal(err)
			}
			ref.add(id, uint64(i), v)
		}
		st := ix.Stats()
		if st.Capacity < 1500 || st.Templates != 1500 || st.Subjects != 750 || st.OnHeap != o.OnHeap {
			t.Fatalf("%+v", st)
		}
		for trial := 0; trial < 20; trial++ {
			q := unit(rng, 32)
			got, err := ix.Search(q, 10)
			if err != nil {
				t.Fatal(err)
			}
			agree(t, got, ref.search(q), 10)
		}
	})
}

func TestCompactionKeepsEverythingElse(t *testing.T) {
	storage(t, func(t *testing.T, o Options) {
		rng := rand.New(rand.NewSource(10))
		ix := mustNew(t, 64, o)
		ref := &oracle{dim: 64}
		const subjects = 3000
		for s := 0; s < subjects; s++ {
			id := fmt.Sprintf("s%04d", s)
			for tmpl := uint64(1); tmpl <= 2; tmpl++ {
				v := unit(rng, 64)
				if err := ix.Add(id, tmpl, v); err != nil {
					t.Fatal(err)
				}
				ref.add(id, tmpl, v)
			}
		}
		// drop a third of the subjects whole, and one template of another third
		compacted := false
		for s := 0; s < subjects; s++ {
			id := fmt.Sprintf("s%04d", s)
			switch s % 3 {
			case 0:
				if ix.RemoveSubject(id) != 2 {
					t.Fatal("RemoveSubject")
				}
				ref.remove(id, 0, true)
			case 1:
				if !ix.Remove(id, 1) {
					t.Fatal("Remove")
				}
				ref.remove(id, 1, false)
			}
			if ix.Stats().Dead == 0 && s > 0 {
				compacted = true
			}
		}
		if !compacted {
			t.Fatal("removing half the templates never triggered a compaction")
		}
		st := ix.Stats()
		if st.Subjects != subjects*2/3 || st.Templates != len(ref.rows) {
			t.Fatalf("%+v, want %d subjects and %d templates", st, subjects*2/3, len(ref.rows))
		}
		for trial := 0; trial < 30; trial++ {
			q := unit(rng, 64)
			got, err := ix.Search(q, 8)
			if err != nil {
				t.Fatal(err)
			}
			agree(t, got, ref.search(q), 8)
			// the template reported is the one that scored
			want := ref.search(q)[0]
			if got[0].Subject == want.subject && got[0].Template != want.template {
				t.Fatalf("subject %s: template %d, want %d", want.subject, got[0].Template, want.template)
			}
		}
		// and additions after a compaction land where they should
		v := unit(rng, 64)
		if err := ix.Add("late", 1, v); err != nil {
			t.Fatal(err)
		}
		if got, _ := ix.Search(v, 1); len(got) != 1 || got[0].Subject != "late" {
			t.Fatalf("%+v", got)
		}
	})
}

func TestWorkersDoNotChangeTheAnswer(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	const dim, rows = 128, 30000
	serial := mustNew(t, dim, Options{Workers: 1})
	parallel := mustNew(t, dim, Options{Workers: 7})
	for i := 0; i < rows; i++ {
		v := unit(rng, dim)
		id := fmt.Sprintf("s%d", i%9000) // templates of one subject are spread over the whole index
		if err := serial.Add(id, uint64(i), v); err != nil {
			t.Fatal(err)
		}
		if err := parallel.Add(id, uint64(i), v); err != nil {
			t.Fatal(err)
		}
	}
	for trial := 0; trial < 25; trial++ {
		q := unit(rng, dim)
		a, _ := serial.Search(q, 20)
		b, _ := parallel.Search(q, 20)
		if len(a) != 20 || len(b) != 20 {
			t.Fatalf("%d and %d matches", len(a), len(b))
		}
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("trial %d, position %d: one worker says %+v, seven say %+v", trial, i, a[i], b[i])
			}
		}
	}
}

func TestClose(t *testing.T) {
	storage(t, func(t *testing.T, o Options) {
		rng := rand.New(rand.NewSource(12))
		ix, err := New(16, o)
		if err != nil {
			t.Fatal(err)
		}
		v := unit(rng, 16)
		if err := ix.Add("a", 1, v); err != nil {
			t.Fatal(err)
		}
		ix.Close()
		ix.Close()
		if _, err := ix.Search(v, 1); !errors.Is(err, ErrClosed) {
			t.Errorf("Search: %v", err)
		}
		if _, _, err := ix.Compare("a", v); !errors.Is(err, ErrClosed) {
			t.Errorf("Compare: %v", err)
		}
		if err := ix.Add("b", 1, v); !errors.Is(err, ErrClosed) {
			t.Errorf("Add: %v", err)
		}
		if err := ix.Compact(); !errors.Is(err, ErrClosed) {
			t.Errorf("Compact: %v", err)
		}
		if ix.Remove("a", 1) || ix.RemoveSubject("a") != 0 || ix.Templates("a") != nil {
			t.Error("a closed index still removes or lists templates")
		}
		if st := ix.Stats(); st.Capacity != 0 {
			t.Errorf("%+v", st)
		}
	})
}

// Searches run while templates come and go, the index grows and is
// compacted. One subject is never touched: every search must find it first,
// with a perfect score, or a reader saw memory it should not have.
func TestSearchesDuringChanges(t *testing.T) {
	storage(t, func(t *testing.T, o Options) {
		const dim = 64
		o.Capacity = 8 // grow often
		rng := rand.New(rand.NewSource(13))
		ix := mustNew(t, dim, o)
		anchor := unit(rng, dim)
		if err := ix.Add("anchor", 1, anchor); err != nil {
			t.Fatal(err)
		}

		var stop atomic.Bool
		var searches atomic.Int64
		var wg sync.WaitGroup
		errs := make(chan string, 64)
		for r := 0; r < 4; r++ {
			wg.Add(1)
			go func(seed int64) {
				defer wg.Done()
				rng := rand.New(rand.NewSource(seed))
				for !stop.Load() {
					got, err := ix.Search(anchor, 3)
					if err != nil || len(got) == 0 || got[0].Subject != "anchor" || got[0].Score < 0.9999 {
						errs <- fmt.Sprintf("anchor search: %+v %v", got, err)
						return
					}
					for _, m := range got[1:] {
						if m.Score > 0.9 || m.Subject == "" {
							errs <- fmt.Sprintf("an unrelated subject scored like the anchor: %+v", m)
							return
						}
					}
					if m, ok, err := ix.Compare("anchor", anchor); err != nil || !ok || m.Score < 0.9999 {
						errs <- fmt.Sprintf("anchor compare: %+v %v %v", m, ok, err)
						return
					}
					ix.Search(unit(rng, dim), 5)
					ix.Has("s17")
					ix.Templates("s17")
					searches.Add(1)
				}
			}(int64(100 + r))
		}

		ref := map[string]bool{}
		for round := 0; round < 6000; round++ {
			id := fmt.Sprintf("s%d", rng.Intn(400))
			switch op := rng.Intn(10); {
			case op < 6:
				if err := ix.Add(id, uint64(round), unit(rng, dim)); err != nil {
					t.Fatal(err)
				}
				ref[id] = true
			case op < 9:
				if ix.RemoveSubject(id) > 0 != ref[id] {
					t.Fatalf("round %d: RemoveSubject(%s) disagrees with the bookkeeping", round, id)
				}
				delete(ref, id)
			default:
				if err := ix.Compact(); err != nil {
					t.Fatal(err)
				}
			}
			if len(errs) > 0 {
				break
			}
		}
		stop.Store(true)
		wg.Wait()
		close(errs)
		for msg := range errs {
			t.Error(msg)
		}
		if searches.Load() == 0 {
			t.Fatal("no search completed while the index was changing")
		}
		if st := ix.Stats(); st.Subjects != len(ref)+1 {
			t.Fatalf("%d subjects, want %d", st.Subjects, len(ref)+1)
		}
		for id := range ref {
			if !ix.Has(id) {
				t.Fatalf("%s went missing", id)
			}
		}
	})
}

func TestConcurrentWriters(t *testing.T) {
	ix := mustNew(t, 32, Options{OnHeap: true, Capacity: 16})
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(w)))
			for i := 0; i < 300; i++ {
				id := fmt.Sprintf("w%d-%d", w, i)
				if err := ix.Add(id, 1, unit(rng, 32)); err != nil {
					t.Error(err)
					return
				}
				if i%3 == 0 {
					ix.RemoveSubject(id)
				}
			}
		}(w)
	}
	wg.Wait()
	if st := ix.Stats(); st.Subjects != 8*200 || st.Templates != 8*200 {
		t.Fatalf("%+v", st)
	}
}
