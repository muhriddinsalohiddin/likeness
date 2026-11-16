package gallery

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/muhriddinsalohiddin/likeness/index"
	"github.com/muhriddinsalohiddin/likeness/store"
)

const dim = 32

var ctx = context.Background()

func unit(rng *rand.Rand) []float32 {
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

// like returns a unit vector at about the given cosine from v.
func like(rng *rand.Rand, v []float32, cos float64) []float32 {
	noise := unit(rng)
	out := make([]float32, dim)
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

func openAt(t testing.TB, path string, o store.Options) *Gallery {
	t.Helper()
	if o.Dim == 0 && !o.ReadOnly {
		o.Dim, o.Fingerprint = dim, "test:model:1"
	}
	st, err := store.Open(path, o)
	if err != nil {
		t.Fatal(err)
	}
	g, err := Open(ctx, st, Options{Index: index.Options{OnHeap: true}})
	if err != nil {
		st.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() { g.Close() })
	return g
}

func newGallery(t testing.TB) (*Gallery, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "faces.db")
	return openAt(t, path, store.Options{}), path
}

// inStep checks that the index holds exactly what the store holds.
func inStep(t testing.TB, g *Gallery) {
	t.Helper()
	templates := 0
	subjects := map[string]bool{}
	err := g.st.Load(ctx, func(tm store.Template) error {
		templates++
		subjects[tm.Subject] = true
		m, ok, err := g.ix.Compare(tm.Subject, tm.Vector)
		if err != nil {
			return err
		}
		if !ok || m.Score < 0.9999 {
			return fmt.Errorf("template %d of %s is on disk but not searchable (%v %v)", tm.ID, tm.Subject, m, ok)
		}
		found := false
		for _, id := range g.ix.Templates(tm.Subject) {
			found = found || id == tm.ID
		}
		if !found {
			return fmt.Errorf("the index does not list template %d of %s", tm.ID, tm.Subject)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if info := g.Info(); info.Templates != templates || info.Subjects != len(subjects) {
		t.Fatalf("the index holds %d subjects / %d templates, the store %d / %d", info.Subjects, info.Templates, len(subjects), templates)
	}
}

func TestEnrolSearchRemove(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	g, path := newGallery(t)
	anna, boris := unit(rng), unit(rng)

	got, err := g.Enrol(ctx, "anna", anna, []byte("first"), EnrolOptions{})
	if err != nil || got.Templates != 1 || got.Template.Subject != "anna" || got.Template.ID == 0 {
		t.Fatalf("%+v %v", got, err)
	}
	if _, err := g.Enrol(ctx, "boris", boris, nil, EnrolOptions{}); err != nil {
		t.Fatal(err)
	}
	second, err := g.Enrol(ctx, "anna", like(rng, anna, 0.9), nil, EnrolOptions{})
	if err != nil || second.Templates != 2 {
		t.Fatalf("%+v %v", second, err)
	}

	matches, err := g.Search(like(rng, anna, 0.95), 5)
	if err != nil || len(matches) != 2 || matches[0].Subject != "anna" || matches[0].Score < 0.8 {
		t.Fatalf("%+v %v", matches, err)
	}
	if m, ok, err := g.Compare("boris", boris); err != nil || !ok || m.Score < 0.9999 {
		t.Fatalf("%+v %v %v", m, ok, err)
	}
	if _, ok, _ := g.Compare("nobody", boris); ok {
		t.Fatal("Compare found someone who is not enrolled")
	}
	sub, err := g.Subject(ctx, "anna")
	if err != nil || len(sub.Templates) != 2 || string(sub.Templates[0].Meta) != "first" {
		t.Fatalf("%+v %v", sub, err)
	}
	if list, err := g.Subjects(ctx, "", 10); err != nil || len(list) != 2 || list[0].ID != "anna" || list[0].Templates != 2 {
		t.Fatalf("%+v %v", list, err)
	}
	info := g.Info()
	if info.Subjects != 2 || info.Templates != 3 || info.Store.Dim != dim || info.Kernel == "" {
		t.Fatalf("%+v", info)
	}
	if !g.Has("anna") || g.Has("nobody") || g.Templates("anna") != 2 || g.Templates("nobody") != 0 {
		t.Fatalf("anna: %v with %d templates; nobody: %v with %d", g.Has("anna"), g.Templates("anna"), g.Has("nobody"), g.Templates("nobody"))
	}
	inStep(t, g)

	// a backup is a store of its own
	var copied bytes.Buffer
	if err := g.Backup(ctx, &copied); err != nil {
		t.Fatal(err)
	}
	backup := filepath.Join(t.TempDir(), "backup.db")
	if err := os.WriteFile(backup, copied.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	if restored := openAt(t, backup, store.Options{}); restored.Info().Templates != 3 {
		t.Fatalf("the backup holds %d templates", restored.Info().Templates)
	}

	if ok, err := g.Remove(ctx, "anna", got.Template.ID); err != nil || !ok {
		t.Fatal(ok, err)
	}
	if ok, _ := g.Remove(ctx, "anna", got.Template.ID); ok {
		t.Fatal("removed the same template twice")
	}
	if n, err := g.RemoveSubject(ctx, "boris"); err != nil || n != 1 {
		t.Fatal(n, err)
	}
	if n, _ := g.RemoveSubject(ctx, "boris"); n != 0 {
		t.Fatal("removed the same subject twice")
	}
	if matches, _ := g.Search(boris, 5); len(matches) != 1 || matches[0].Subject != "anna" || matches[0].Template != second.Template.ID {
		t.Fatalf("%+v", matches)
	}
	if _, err := g.Subject(ctx, "boris"); !errors.Is(err, store.ErrNotFound) {
		t.Fatalf("%v", err)
	}
	inStep(t, g)

	// everything is still there after a restart
	if err := g.Close(); err != nil {
		t.Fatal(err)
	}
	again := openAt(t, path, store.Options{})
	if info := again.Info(); info.Subjects != 1 || info.Templates != 1 {
		t.Fatalf("%+v", info)
	}
	if matches, _ := again.Search(anna, 5); len(matches) != 1 || matches[0].Subject != "anna" || matches[0].Template != second.Template.ID {
		t.Fatalf("%+v", matches)
	}
	inStep(t, again)
}

func TestDuplicateFace(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	g, _ := newGallery(t)
	anna := unit(rng)
	strict := EnrolOptions{DuplicateAbove: 0.6}
	if _, err := g.Enrol(ctx, "anna", anna, nil, strict); err != nil {
		t.Fatal(err)
	}

	// the same face under another name
	_, err := g.Enrol(ctx, "anna-again", like(rng, anna, 0.9), nil, strict)
	var dup *DuplicateError
	if !errors.As(err, &dup) || dup.Subject != "anna" || dup.Score < 0.8 || dup.Error() == "" {
		t.Fatalf("%v", err)
	}
	if info := g.Info(); info.Subjects != 1 || info.Templates != 1 {
		t.Fatalf("a refused enrolment left something behind: %+v", info)
	}
	// a second template for anna herself is not a duplicate
	if _, err := g.Enrol(ctx, "anna", like(rng, anna, 0.9), nil, strict); err != nil {
		t.Fatal(err)
	}
	// someone else is fine, and so is the duplicate when nobody checks
	if _, err := g.Enrol(ctx, "boris", unit(rng), nil, strict); err != nil {
		t.Fatal(err)
	}
	if _, err := g.Enrol(ctx, "twin", like(rng, anna, 0.9), nil, EnrolOptions{}); err != nil {
		t.Fatal(err)
	}
	// with anna and her twin both enrolled, a third copy is still caught,
	// even for anna herself being the best match
	_, err = g.Enrol(ctx, "third", like(rng, anna, 0.95), nil, strict)
	if !errors.As(err, &dup) {
		t.Fatalf("%v", err)
	}
	inStep(t, g)
}

func TestSubjectMismatch(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	g, _ := newGallery(t)
	anna := unit(rng)
	careful := EnrolOptions{MismatchBelow: 0.4}
	if _, err := g.Enrol(ctx, "anna", anna, nil, careful); err != nil {
		t.Fatal(err)
	}
	// somebody else's face filed under anna
	_, err := g.Enrol(ctx, "anna", unit(rng), nil, careful)
	var mm *MismatchError
	if !errors.As(err, &mm) || mm.Score > 0.4 || mm.Error() == "" {
		t.Fatalf("%v", err)
	}
	if _, err := g.Enrol(ctx, "anna", like(rng, anna, 0.8), nil, careful); err != nil {
		t.Fatalf("anna's own face: %v", err)
	}
	// replacing is a deliberate fresh start and is not held to the old face
	stranger := unit(rng)
	got, err := g.Enrol(ctx, "anna", stranger, nil, EnrolOptions{MismatchBelow: 0.4, Replace: true})
	if err != nil || got.Templates != 1 {
		t.Fatalf("%+v %v", got, err)
	}
	if m, _, _ := g.Compare("anna", stranger); m.Score < 0.9999 || m.Template != got.Template.ID {
		t.Fatalf("%+v", m)
	}
	if m, _, _ := g.Compare("anna", anna); m.Score > 0.7 {
		t.Fatalf("the replaced templates still answer: %+v", m)
	}
	inStep(t, g)
}

func TestTemplateLimitAndReplace(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	g, _ := newGallery(t)
	anna := unit(rng)
	limited := EnrolOptions{MaxTemplates: 3}
	var ids []uint64
	for i := 0; i < 3; i++ {
		got, err := g.Enrol(ctx, "anna", like(rng, anna, 0.9), nil, limited)
		if err != nil || got.Templates != i+1 {
			t.Fatalf("%+v %v", got, err)
		}
		ids = append(ids, got.Template.ID)
	}
	if _, err := g.Enrol(ctx, "anna", like(rng, anna, 0.9), nil, limited); !errors.Is(err, ErrTooManyTemplates) {
		t.Fatalf("%v", err)
	}
	// the limit is per subject
	if _, err := g.Enrol(ctx, "boris", unit(rng), nil, limited); err != nil {
		t.Fatal(err)
	}

	fresh := like(rng, anna, 0.9)
	got, err := g.Enrol(ctx, "anna", fresh, nil, EnrolOptions{MaxTemplates: 3, Replace: true})
	if err != nil || got.Templates != 1 {
		t.Fatalf("%+v %v", got, err)
	}
	sub, _ := g.Subject(ctx, "anna")
	if len(sub.Templates) != 1 || sub.Templates[0].ID != got.Template.ID {
		t.Fatalf("%+v", sub)
	}
	for _, id := range ids {
		if ok, _ := g.Remove(ctx, "anna", id); ok {
			t.Fatalf("template %d survived the replacement", id)
		}
	}
	// replacing a subject that does not exist yet simply enrols it
	if got, err := g.Enrol(ctx, "carol", unit(rng), nil, EnrolOptions{Replace: true}); err != nil || got.Templates != 1 {
		t.Fatalf("%+v %v", got, err)
	}
	inStep(t, g)
}

func TestBadInputChangesNothing(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	g, _ := newGallery(t)
	for name, c := range map[string]struct {
		subject string
		vec     []float32
		want    error
	}{
		"bad id":       {"no spaces please", unit(rng), store.ErrSubject},
		"short vector": {"anna", unit(rng)[:dim-1], nil},
		"zero vector":  {"anna", make([]float32, dim), nil},
	} {
		_, err := g.Enrol(ctx, c.subject, c.vec, nil, EnrolOptions{DuplicateAbove: 0.5})
		if err == nil || (c.want != nil && !errors.Is(err, c.want)) {
			t.Errorf("%s: %v", name, err)
		}
	}
	if _, err := g.Enrol(ctx, "anna", unit(rng), make([]byte, store.MaxMeta+1), EnrolOptions{}); !errors.Is(err, store.ErrMeta) {
		t.Errorf("oversized metadata: %v", err)
	}
	if info := g.Info(); info.Subjects != 0 || info.Templates != 0 {
		t.Fatalf("%+v", info)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := g.Enrol(cancelled, "anna", unit(rng), nil, EnrolOptions{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("%v", err)
	}
	inStep(t, g)
}

// If the index refuses a template the store has already taken, the store
// must give it back: otherwise the next start would serve a template this
// run never did.
func TestStoreIsRolledBackWhenTheIndexRefuses(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	g, path := newGallery(t)
	if _, err := g.Enrol(ctx, "anna", unit(rng), nil, EnrolOptions{}); err != nil {
		t.Fatal(err)
	}
	g.ix.Close() // the index is gone; the store still works
	if _, err := g.Enrol(ctx, "boris", unit(rng), nil, EnrolOptions{}); !errors.Is(err, index.ErrClosed) {
		t.Fatalf("%v", err)
	}
	if subjects, templates, _ := g.st.Count(ctx); subjects != 1 || templates != 1 {
		t.Fatalf("the store kept a template the index never had: %d subjects, %d templates", subjects, templates)
	}
	g.Close()
	again := openAt(t, path, store.Options{})
	if info := again.Info(); info.Subjects != 1 || info.Templates != 1 {
		t.Fatalf("%+v", info)
	}
}

func TestReadOnlyGallery(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	g, path := newGallery(t)
	anna := unit(rng)
	got, err := g.Enrol(ctx, "anna", anna, nil, EnrolOptions{})
	if err != nil {
		t.Fatal(err)
	}
	g.Close()

	ro := openAt(t, path, store.Options{ReadOnly: true})
	if matches, err := ro.Search(anna, 1); err != nil || len(matches) != 1 || matches[0].Subject != "anna" {
		t.Fatalf("%+v %v", matches, err)
	}
	if _, err := ro.Enrol(ctx, "boris", unit(rng), nil, EnrolOptions{}); !errors.Is(err, store.ErrReadOnly) {
		t.Fatalf("%v", err)
	}
	if _, err := ro.Remove(ctx, "anna", got.Template.ID); !errors.Is(err, store.ErrReadOnly) {
		t.Fatalf("%v", err)
	}
	if _, err := ro.RemoveSubject(ctx, "anna"); !errors.Is(err, store.ErrReadOnly) {
		t.Fatalf("%v", err)
	}
	inStep(t, ro)
}

func TestOpenFailsOnACorruptStore(t *testing.T) {
	path := filepath.Join(t.TempDir(), "faces.db")
	st, err := store.Open(path, store.Options{Dim: dim, Fingerprint: "test:model:1"})
	if err != nil {
		t.Fatal(err)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := Open(cancelled, st, Options{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("%v", err)
	}
	if _, err := Open(ctx, st, Options{Index: index.Options{Workers: -1}}); err == nil {
		t.Fatal("bad index options were accepted")
	}
	st.Close()
}

// Enrolments and removals from many goroutines, with searches in between:
// afterwards the index and the store must hold exactly the same.
func TestStoreAndIndexStayInStep(t *testing.T) {
	g, _ := newGallery(t)
	var wg sync.WaitGroup
	for w := 0; w < 6; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(w)))
			for i := 0; i < 25; i++ {
				// the same few subjects from every goroutine, on purpose
				subject := fmt.Sprintf("s%d", rng.Intn(8))
				switch rng.Intn(5) {
				case 0:
					if _, err := g.RemoveSubject(ctx, subject); err != nil {
						t.Error(err)
						return
					}
				case 1:
					if _, err := g.Enrol(ctx, subject, unit(rng), nil, EnrolOptions{Replace: true}); err != nil {
						t.Error(err)
						return
					}
				default:
					got, err := g.Enrol(ctx, subject, unit(rng), nil, EnrolOptions{MaxTemplates: 4})
					if err != nil && !errors.Is(err, ErrTooManyTemplates) {
						t.Error(err)
						return
					}
					if err == nil && rng.Intn(3) == 0 {
						if _, err := g.Remove(ctx, subject, got.Template.ID); err != nil {
							t.Error(err)
							return
						}
					}
				}
				if _, err := g.Search(unit(rng), 3); err != nil {
					t.Error(err)
					return
				}
			}
		}(w)
	}
	wg.Wait()
	inStep(t, g)
	for _, sm := range mustList(t, g) {
		if sm.Templates > 4 {
			t.Fatalf("%s has %d templates, the limit was 4", sm.ID, sm.Templates)
		}
	}
}

func mustList(t *testing.T, g *Gallery) []store.Summary {
	t.Helper()
	list, err := g.Subjects(ctx, "", 1000)
	if err != nil {
		t.Fatal(err)
	}
	return list
}
