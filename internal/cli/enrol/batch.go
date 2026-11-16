package enrol

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"os"
	"slices"
	"sync"

	"github.com/muhriddinsalohiddin/likeness"
	"github.com/muhriddinsalohiddin/likeness/internal/cli"
	"github.com/muhriddinsalohiddin/likeness/internal/gallery"
	"github.com/muhriddinsalohiddin/likeness/internal/server"
	"github.com/muhriddinsalohiddin/likeness/store"
)

const (
	// sameAbove is the similarity from which a picture is the very one a
	// template was made from. Analysing a file again gives the same vector;
	// another picture of the same person does not come this close.
	sameAbove = 0.9999
	// maxPicture is the largest file read as a picture, in bytes.
	maxPicture = 64 << 20
)

// outcome is what became of a picture.
type outcome string

const (
	enrolled        outcome = "enrolled"
	wouldEnrol      outcome = "would be enrolled"
	alreadyEnrolled outcome = "already enrolled"
	turnedDown      outcome = "turned down"
	duplicate       outcome = "duplicate"
	mismatch        outcome = "mismatch"
	full            outcome = "full"
	unreadable      outcome = "unreadable"
	invalidSubject  outcome = "invalid subject"
	invalidLine     outcome = "invalid line"
)

// verdict is an outcome and what explains it.
type verdict struct {
	outcome outcome
	// reason is why a picture was turned down or could not be used.
	reason string
	// score and other belong to a duplicate or a mismatch: the similarity,
	// and the subject the face is enrolled as.
	score float32
	other string
	// template is the template the picture became, or is already.
	template uint64
}

// seen is what the engine made of a picture.
type seen struct {
	vector  []float32
	quality likeness.Quality
	reject  likeness.Reason
	// unreadable says what is wrong with the file.
	unreadable string
	// err is a failure that is not the picture's fault and ends the run.
	err error
}

// batch is one run over the pictures.
type batch struct {
	engine cli.Engine
	// faces is nil in a dry run on a store that does not exist yet.
	faces   *gallery.Gallery
	rules   gallery.EnrolOptions
	replace bool
	dryRun  bool
	workers int

	report   *report
	progress *progress

	// replaced holds the subjects whose earlier templates this run has
	// replaced, and added what a dry run would have given each subject.
	replaced map[string]bool
	added    map[string]int

	done    int
	count   map[outcome]int
	rejects map[string]int
}

// run analyses the pictures, several at a time, and settles them one by one
// in the order of the input: what becomes of a picture must not depend on
// which analysis happened to finish first.
func (b *batch) run(ctx context.Context, pictures []picture) error {
	ctx, stop := context.WithCancel(ctx)
	var wg sync.WaitGroup
	defer func() {
		stop()
		wg.Wait() // the engine is closed next, and has to be idle by then
	}()

	type job struct {
		picture *picture
		result  chan<- seen
	}
	jobs := make(chan job)
	// queue holds the results to come, in the order of the input. Its
	// length bounds how far the analysis runs ahead of the store.
	queue := make(chan (<-chan seen), 2*b.workers)
	for range b.workers {
		wg.Go(func() {
			for j := range jobs {
				j.result <- b.look(ctx, j.picture)
			}
		})
	}
	wg.Go(func() {
		defer close(jobs)
		for i := range pictures {
			if pictures[i].refused != "" {
				continue
			}
			result := make(chan seen, 1)
			select {
			case queue <- result:
			case <-ctx.Done():
				return
			}
			select {
			case jobs <- job{&pictures[i], result}:
			case <-ctx.Done():
				return
			}
		}
	})

	for i := range pictures {
		p := &pictures[i]
		v := verdict{outcome: p.refused, reason: p.reason}
		if p.refused == "" {
			result, err := await(ctx, queue)
			if err != nil {
				return err
			}
			s, err := await(ctx, result)
			if err != nil {
				return err
			}
			if err := ctx.Err(); err != nil {
				return err
			}
			if v, err = b.settle(ctx, p, s); err != nil {
				return fmt.Errorf("%s: %w", p.path, err)
			}
		}
		b.done++
		b.count[v.outcome]++
		if v.outcome == turnedDown {
			b.rejects[v.reason]++
		}
		b.report.add(p, v)
		b.progress.step(b.done)
	}
	return nil
}

// await takes the next value from c, unless ctx ends first.
func await[T any](ctx context.Context, c <-chan T) (T, error) {
	select {
	case v := <-c:
		return v, nil
	case <-ctx.Done():
		var none T
		return none, ctx.Err()
	}
}

// look reads a picture and has the engine analyse it.
func (b *batch) look(ctx context.Context, p *picture) seen {
	st, err := os.Stat(p.path)
	if err != nil {
		return seen{unreadable: cause(err)}
	}
	if st.Size() > maxPicture {
		return seen{unreadable: fmt.Sprintf("larger than %d MiB", maxPicture>>20)}
	}
	file, err := os.ReadFile(p.path)
	if err != nil {
		return seen{unreadable: cause(err)}
	}
	res, err := b.engine.Analyze(ctx, file, likeness.ForEnrolment())
	switch {
	case errors.Is(err, likeness.ErrUnsupportedImage):
		return seen{unreadable: "not a JPEG, PNG or WebP image"}
	case errors.Is(err, likeness.ErrImageTooLarge):
		return seen{unreadable: "more pixels than the engine accepts"}
	case errors.Is(err, likeness.ErrImageTooSmall):
		return seen{unreadable: "too small to hold a face"}
	case errors.Is(err, likeness.ErrCorruptImage):
		return seen{unreadable: "damaged or cut short"}
	case err != nil:
		return seen{err: err}
	}
	face, reason := res.Best()
	if reason != likeness.Accepted {
		return seen{reject: reason}
	}
	return seen{vector: face.Embedding, quality: face.Quality}
}

// cause is the reason a file could not be read, without its path: the
// path is told separately.
func cause(err error) string {
	var failed *fs.PathError
	if errors.As(err, &failed) {
		return failed.Err.Error()
	}
	return err.Error()
}

// settle decides what becomes of a picture the engine has seen.
func (b *batch) settle(ctx context.Context, p *picture, s seen) (verdict, error) {
	switch {
	case s.err != nil:
		return verdict{}, s.err
	case s.unreadable != "":
		return verdict{outcome: unreadable, reason: s.unreadable}, nil
	case s.reject != likeness.Accepted:
		return verdict{outcome: turnedDown, reason: string(s.reject)}, nil
	case b.dryRun:
		return b.rehearse(p.subject, s.vector)
	}
	return b.enrol(ctx, p.subject, s)
}

// enrol puts a face into the gallery under the server's rules, unless the
// subject holds this very picture already.
func (b *batch) enrol(ctx context.Context, subject string, s seen) (verdict, error) {
	// With --replace the first picture of a subject is enrolled even if
	// the subject holds it already: that is what makes the others go.
	replace := b.replace && !b.replaced[subject]
	if !replace {
		m, held, err := b.faces.Compare(subject, s.vector)
		if err != nil {
			return verdict{}, err
		}
		if held && m.Score >= sameAbove {
			return verdict{outcome: alreadyEnrolled, template: m.Template}, nil
		}
	}
	meta, err := server.TemplateMeta(s.quality)
	if err != nil {
		return verdict{}, err
	}
	rules := b.rules
	rules.Replace = replace
	// Not the run's context: an interrupt must not leave a replacement
	// half done.
	done, err := b.faces.Enrol(context.WithoutCancel(ctx), subject, s.vector, meta, rules)
	if err != nil {
		return refusal(err)
	}
	if replace {
		b.replaced[subject] = true
	}
	return verdict{outcome: enrolled, template: done.Template.ID}, nil
}

// refusal turns the gallery's objection to a face into an outcome. Any
// other error is returned as it is.
func refusal(err error) (verdict, error) {
	var (
		taken *gallery.DuplicateError
		odd   *gallery.MismatchError
	)
	switch {
	case errors.As(err, &taken):
		return verdict{outcome: duplicate, other: taken.Subject, score: taken.Score}, nil
	case errors.As(err, &odd):
		return verdict{outcome: mismatch, score: odd.Score}, nil
	case errors.Is(err, gallery.ErrTooManyTemplates):
		return verdict{outcome: full}, nil
	case errors.Is(err, store.ErrSubject):
		return verdict{outcome: invalidSubject, reason: whyInvalid(err)}, nil
	}
	return verdict{}, err
}

// rehearse is enrol for a dry run: the gallery's checks in the gallery's
// order, against what the store holds, with nothing written. What the run
// would have added it knows by count alone, so pictures of one batch are
// not compared with each other.
func (b *batch) rehearse(subject string, vector []float32) (verdict, error) {
	// Once --replace has taken a picture for the subject, what the store
	// holds for it would be gone.
	replace := b.replace && !b.replaced[subject]
	inStore := b.faces != nil && !b.replaced[subject]
	if inStore && !replace {
		m, held, err := b.faces.Compare(subject, vector)
		if err != nil {
			return verdict{}, err
		}
		if held && m.Score >= sameAbove {
			return verdict{outcome: alreadyEnrolled, template: m.Template}, nil
		}
	}
	if b.faces != nil && b.rules.DuplicateAbove > 0 {
		matches, err := b.faces.Search(vector, 2)
		if err != nil {
			return verdict{}, err
		}
		for _, m := range matches {
			if m.Subject != subject && m.Score >= b.rules.DuplicateAbove {
				return verdict{outcome: duplicate, other: m.Subject, score: m.Score}, nil
			}
		}
	}
	if !replace {
		templates := b.added[subject]
		if inStore {
			templates += b.faces.Templates(subject)
			if b.rules.MismatchBelow > 0 {
				m, held, err := b.faces.Compare(subject, vector)
				if err != nil {
					return verdict{}, err
				}
				if held && m.Score < b.rules.MismatchBelow {
					return verdict{outcome: mismatch, score: m.Score}, nil
				}
			}
		}
		if b.rules.MaxTemplates > 0 && templates >= b.rules.MaxTemplates {
			return verdict{outcome: full}, nil
		}
	}
	if replace {
		b.replaced[subject] = true
	}
	b.added[subject]++
	return verdict{outcome: wouldEnrol}, nil
}

// subjects is the number of subjects in the store.
func (b *batch) subjects() int {
	if b.faces == nil {
		return 0
	}
	return b.faces.Info().Subjects
}

func sortedKeys(m map[string]int) []string { return slices.Sorted(maps.Keys(m)) }
