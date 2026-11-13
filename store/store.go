// Package store keeps enrolled faces on disk.
//
// The store is the source of truth: a search index holds the same vectors
// in memory and is rebuilt from the store when a server starts. A change is
// written here first, so a crash between the two leaves nothing that the
// next start does not repair.
//
// A store remembers which model made its vectors. Vectors from different
// models cannot be compared, so opening a store for another model is an
// error rather than a source of wrong answers.
package store

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"time"
	"unicode"
	"unicode/utf8"
)

var (
	// ErrNotFound: the subject is not enrolled.
	ErrNotFound = errors.New("store: no such subject")
	// ErrSubject: the subject id is not acceptable; see ValidSubject.
	ErrSubject = errors.New("store: invalid subject id")
	// ErrVector: the vector has the wrong length, is all zero or not finite.
	ErrVector = errors.New("store: invalid vector")
	// ErrMeta: the metadata exceeds MaxMeta.
	ErrMeta = errors.New("store: metadata too large")
	// ErrMismatch: the store was made for another model.
	ErrMismatch = errors.New("store: made for a different model")
	// ErrKey: the encryption key is wrong, missing or not wanted.
	ErrKey = errors.New("store: encryption key")
	// ErrBusy: another process holds the store.
	ErrBusy = errors.New("store: in use by another process")
	// ErrCorrupt: a record cannot be read back.
	ErrCorrupt = errors.New("store: corrupt record")
	// ErrReadOnly: the store was opened read-only.
	ErrReadOnly = errors.New("store: opened read-only")
	// ErrClosed: the store has been closed.
	ErrClosed = errors.New("store: closed")
)

const (
	// MaxSubject is the longest subject id, in bytes.
	MaxSubject = 128
	// MaxMeta is the largest metadata blob a template may carry, in bytes.
	MaxMeta = 64 << 10
	// KeySize is the length of an encryption key, in bytes.
	KeySize = 32
)

// Info describes what a store holds.
type Info struct {
	// Dim is the length of every vector.
	Dim int
	// Fingerprint identifies the model that made the vectors.
	Fingerprint string
	Created     time.Time
	// Encrypted reports whether vectors and metadata are encrypted on disk.
	Encrypted bool
}

// Template is one enrolled vector.
type Template struct {
	// ID is assigned by the store and never reused.
	ID      uint64
	Subject string
	Created time.Time
	Vector  []float32
	// Meta is whatever the caller attached: where the picture came from,
	// its quality. The store does not look inside.
	Meta []byte
}

// TemplateInfo is a template without its vector.
type TemplateInfo struct {
	ID      uint64
	Created time.Time
	Meta    []byte
}

// Subject is one enrolled person with their templates, oldest first.
type Subject struct {
	ID        string
	Created   time.Time
	Templates []TemplateInfo
}

// Summary is one line of a subject listing.
type Summary struct {
	ID        string
	Created   time.Time
	Templates int
}

// Entry is a template to add.
type Entry struct {
	Subject string
	Vector  []float32
	Meta    []byte
}

// Store is safe for concurrent use.
type Store interface {
	// Info reports what the store was created for.
	Info() Info

	// Add enrols a vector for a subject, creating the subject if need be.
	// When it returns without error the template is on disk.
	Add(ctx context.Context, subject string, vector []float32, meta []byte) (Template, error)
	// AddMany enrols several vectors at once: all of them, or on error none.
	AddMany(ctx context.Context, entries []Entry) ([]Template, error)

	// Remove deletes one template and reports whether it existed. A subject
	// whose last template is removed is gone.
	Remove(ctx context.Context, subject string, template uint64) (bool, error)
	// RemoveSubject deletes a subject and returns how many templates it had.
	RemoveSubject(ctx context.Context, subject string) (int, error)

	// Subject returns one subject, or ErrNotFound.
	Subject(ctx context.Context, id string) (Subject, error)
	// Subjects lists subjects in id order, starting after the given id
	// ("" for the beginning). limit <= 0 means 100; the most is 1000.
	Subjects(ctx context.Context, after string, limit int) ([]Summary, error)
	// Count returns the number of subjects and templates.
	Count(ctx context.Context) (subjects, templates int, err error)

	// Load calls fn for every template, grouped by subject. The Vector and
	// Meta of the template passed to fn are reused for the next one: copy
	// what must outlive the call. An error from fn stops the load.
	Load(ctx context.Context, fn func(Template) error) error

	// Backup writes a consistent copy of the store, usable as a store file,
	// while the store stays in use.
	Backup(ctx context.Context, w io.Writer) error

	Close() error
}

// ValidSubject reports whether id can name a subject: 1 to MaxSubject bytes
// of UTF-8 without spaces, control characters or slashes. Ids are stored in
// the clear even in an encrypted store; use opaque ones if they are
// sensitive.
func ValidSubject(id string) error {
	switch {
	case id == "":
		return fmt.Errorf("%w: empty", ErrSubject)
	case len(id) > MaxSubject:
		return fmt.Errorf("%w: longer than %d bytes", ErrSubject, MaxSubject)
	case !utf8.ValidString(id):
		return fmt.Errorf("%w: not valid UTF-8", ErrSubject)
	}
	for _, r := range id {
		if r == '/' || unicode.IsControl(r) || unicode.IsSpace(r) {
			return fmt.Errorf("%w: contains %q", ErrSubject, r)
		}
	}
	return nil
}

func nameable(id string) bool { return ValidSubject(id) == nil }

func validVector(v []float32, dim int) error {
	if len(v) != dim {
		return fmt.Errorf("%w: %d values, want %d", ErrVector, len(v), dim)
	}
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if !(sum > 0) || math.IsInf(sum, 0) {
		return fmt.Errorf("%w: zero or not finite", ErrVector)
	}
	return nil
}

func validEntry(e Entry, dim int) error {
	if err := ValidSubject(e.Subject); err != nil {
		return err
	}
	if err := validVector(e.Vector, dim); err != nil {
		return err
	}
	if len(e.Meta) > MaxMeta {
		return fmt.Errorf("%w: %d bytes, at most %d", ErrMeta, len(e.Meta), MaxMeta)
	}
	return nil
}
