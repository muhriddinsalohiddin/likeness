// Package gallery is the set of enrolled people: a store on disk and a
// search index in memory, kept in step.
//
// The store is the source of truth. Every change goes to the store first
// and to the index second, one change at a time, so the two cannot drift
// apart while the process runs; and since the index is rebuilt from the
// store at start-up, they cannot stay apart after a crash either.
package gallery

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/muhriddinsalohiddin/likeness/index"
	"github.com/muhriddinsalohiddin/likeness/store"
)

// ErrTooManyTemplates: the subject already has as many templates as allowed.
var ErrTooManyTemplates = errors.New("gallery: the subject has as many templates as allowed")

// DuplicateError: the face being enrolled already belongs to someone else.
type DuplicateError struct {
	Subject string
	Score   float32
}

func (e *DuplicateError) Error() string {
	return fmt.Sprintf("gallery: the face is already enrolled as another subject (similarity %.2f)", e.Score)
}

// MismatchError: the face being added does not look like the subject's
// existing templates.
type MismatchError struct {
	Score float32
}

func (e *MismatchError) Error() string {
	return fmt.Sprintf("gallery: the face does not match the subject's templates (similarity %.2f)", e.Score)
}

// Options tune a Gallery.
type Options struct {
	Index index.Options
}

// Gallery is safe for concurrent use.
type Gallery struct {
	st store.Store
	ix *index.Index

	// mu admits one change at a time. Searches do not take it.
	mu     sync.Mutex
	loaded time.Duration
}

// Open builds the index from the store. The gallery owns the store from
// here on and closes it with Close.
func Open(ctx context.Context, st store.Store, o Options) (*Gallery, error) {
	ix, err := index.New(st.Info().Dim, o.Index)
	if err != nil {
		return nil, fmt.Errorf("gallery: %w", err)
	}
	start := time.Now()
	err = st.Load(ctx, func(tm store.Template) error {
		return ix.Add(tm.Subject, tm.ID, tm.Vector)
	})
	if err != nil {
		ix.Close()
		return nil, fmt.Errorf("gallery: loading the store: %w", err)
	}
	return &Gallery{st: st, ix: ix, loaded: time.Since(start)}, nil
}

// Info describes the gallery.
type Info struct {
	Store     store.Info
	Subjects  int
	Templates int
	// LoadTime is how long building the index took at start-up.
	LoadTime time.Duration
	Kernel   string
}

func (g *Gallery) Info() Info {
	st := g.ix.Stats()
	return Info{Store: g.st.Info(), Subjects: st.Subjects, Templates: st.Templates, LoadTime: g.loaded, Kernel: st.Kernel}
}

// EnrolOptions say what an enrolment has to respect. The zero value checks
// nothing.
type EnrolOptions struct {
	// Replace drops the subject's earlier templates once the new one is in.
	Replace bool
	// MaxTemplates bounds the templates a subject may hold; 0 is no bound.
	MaxTemplates int
	// DuplicateAbove refuses a face that another subject matches with at
	// least this similarity; 0 turns the check off.
	DuplicateAbove float32
	// MismatchBelow refuses a face for a subject whose existing templates
	// all score lower than this against it; 0 turns the check off.
	MismatchBelow float32
}

// Enrolled is the result of an enrolment.
type Enrolled struct {
	Template store.Template
	// Templates is how many templates the subject has now.
	Templates int
}

// Enrol adds a template for a subject. When it returns without error the
// template is on disk and searchable.
func (g *Gallery) Enrol(ctx context.Context, subject string, vector []float32, meta []byte, o EnrolOptions) (Enrolled, error) {
	if err := store.ValidSubject(subject); err != nil {
		return Enrolled{}, err
	}
	g.mu.Lock()
	defer g.mu.Unlock()

	if o.DuplicateAbove > 0 {
		// the two best subjects: the first may be the subject itself
		matches, err := g.ix.Search(vector, 2)
		if err != nil {
			return Enrolled{}, err
		}
		for _, m := range matches {
			if m.Subject != subject && m.Score >= o.DuplicateAbove {
				return Enrolled{}, &DuplicateError{Subject: m.Subject, Score: m.Score}
			}
		}
	}
	earlier := g.ix.Templates(subject)
	if len(earlier) > 0 && !o.Replace {
		if o.MismatchBelow > 0 {
			m, ok, err := g.ix.Compare(subject, vector)
			if err != nil {
				return Enrolled{}, err
			}
			if ok && m.Score < o.MismatchBelow {
				return Enrolled{}, &MismatchError{Score: m.Score}
			}
		}
		if o.MaxTemplates > 0 && len(earlier) >= o.MaxTemplates {
			return Enrolled{}, ErrTooManyTemplates
		}
	}

	tm, err := g.st.Add(ctx, subject, vector, meta)
	if err != nil {
		return Enrolled{}, err
	}
	if err := g.ix.Add(subject, tm.ID, vector); err != nil {
		// The index would not take it: take it back out of the store, or
		// the next start would bring up a template this run never served.
		if _, undo := g.st.Remove(context.WithoutCancel(ctx), subject, tm.ID); undo != nil {
			return Enrolled{}, fmt.Errorf("%w (and the store could not be rolled back: %w)", err, undo)
		}
		return Enrolled{}, err
	}
	if o.Replace {
		// The new template is in before the old ones go, so the subject is
		// never without one.
		for _, id := range earlier {
			if _, err := g.st.Remove(ctx, subject, id); err != nil {
				return Enrolled{}, fmt.Errorf("gallery: the new template is enrolled, but an earlier one could not be removed: %w", err)
			}
			g.ix.Remove(subject, id)
		}
	}
	return Enrolled{Template: tm, Templates: len(g.ix.Templates(subject))}, nil
}

// Remove deletes one template and reports whether it existed.
func (g *Gallery) Remove(ctx context.Context, subject string, template uint64) (bool, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	ok, err := g.st.Remove(ctx, subject, template)
	if err != nil || !ok {
		return false, err
	}
	g.ix.Remove(subject, template)
	return true, nil
}

// RemoveSubject deletes a subject and returns how many templates it had.
func (g *Gallery) RemoveSubject(ctx context.Context, subject string) (int, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	n, err := g.st.RemoveSubject(ctx, subject)
	if err != nil || n == 0 {
		return 0, err
	}
	g.ix.RemoveSubject(subject)
	return n, nil
}

// Search returns up to limit subjects, most similar first.
func (g *Gallery) Search(vector []float32, limit int) ([]index.Match, error) {
	return g.ix.Search(vector, limit)
}

// Compare scores a vector against one subject. The boolean is false when
// the subject is not enrolled.
func (g *Gallery) Compare(subject string, vector []float32) (index.Match, bool, error) {
	return g.ix.Compare(subject, vector)
}

// Subject returns one subject with its templates, or store.ErrNotFound.
func (g *Gallery) Subject(ctx context.Context, id string) (store.Subject, error) {
	return g.st.Subject(ctx, id)
}

// Subjects lists subjects in id order, starting after the given id.
func (g *Gallery) Subjects(ctx context.Context, after string, limit int) ([]store.Summary, error) {
	return g.st.Subjects(ctx, after, limit)
}

// Has reports whether the subject is enrolled.
func (g *Gallery) Has(subject string) bool { return g.ix.Has(subject) }

// Templates is the number of templates the subject holds; 0 if it is not
// enrolled.
func (g *Gallery) Templates(subject string) int { return len(g.ix.Templates(subject)) }

// Backup writes a consistent copy of the store.
func (g *Gallery) Backup(ctx context.Context, w io.Writer) error {
	return g.st.Backup(ctx, w)
}

// Close releases the index and closes the store. Changes in flight finish
// first.
func (g *Gallery) Close() error {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.ix.Close()
	return g.st.Close()
}
