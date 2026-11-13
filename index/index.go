// Package index finds the enrolled faces most similar to a query vector.
//
// The search is exact: every stored vector is compared with the query, so
// there is nothing to tune and no recall to lose. One comparison is a dot
// product of unit vectors, done in blocks by a SIMD kernel and split across
// cores; a hundred thousand 512-value vectors take a few milliseconds.
//
// A subject (a person) may have several templates (vectors). A search
// reports each subject once, by its best template.
//
// Searches never wait for enrolments: a writer prepares the new row out of
// sight and publishes it with one atomic store.
package index

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
)

var (
	ErrClosed    = errors.New("index: closed")
	ErrDimension = errors.New("index: vector has the wrong length")
	ErrVector    = errors.New("index: vector is zero or not finite")
	ErrExists    = errors.New("index: the subject already has this template")
	ErrSubject   = errors.New("index: empty subject")
)

// Options tune an Index. The zero value is right for a server.
type Options struct {
	// Capacity is the number of templates to make room for at the start.
	// The index grows when it is full, so this is an optimisation, not a
	// limit. Zero means about a million off the heap (address space only;
	// memory is committed as templates arrive) or a thousand on it.
	Capacity int
	// Workers bounds the goroutines one search may use; 0 means one per CPU.
	Workers int
	// OnHeap keeps the vectors in ordinary Go memory instead of mapping
	// them outside the heap. It is what platforms without anonymous
	// mappings get anyway.
	OnHeap bool
}

// Match is one subject found by a search.
type Match struct {
	Subject string
	// Score is the cosine similarity of the query with the subject's best
	// template, in [-1, 1].
	Score float32
	// Template is the id of that template.
	Template uint64
}

// Stats describes the index at one moment.
type Stats struct {
	Subjects  int
	Templates int
	// Dead counts removed templates whose rows have not been reclaimed yet.
	Dead     int
	Capacity int
	Dim      int
	OnHeap   bool
	Kernel   string
}

// Index is safe for concurrent use.
type Index struct {
	dim     int
	workers int
	onHeap  bool

	// wmu admits one writer at a time. Readers never take it.
	wmu      sync.Mutex
	deadRows int

	// life is read-held by anything looking at a block and write-held, for
	// a moment, when a block is retired. Adding and removing templates does
	// not retire blocks; growing and compacting do.
	life sync.RWMutex
	snap atomic.Pointer[snapshot]

	// smu guards the subject table. It is taken after life, never before.
	smu      sync.RWMutex
	byID     map[string]uint32
	slots    []slot
	subjects int
	live     int

	queries sync.Pool // *[]float32 of dim values
	scores  sync.Pool // *[]float32 of chunkRows values
}

// slot is one subject: its id and the rows of its live templates. The slot
// number is what rows store as their owner. A removed subject keeps its
// number, with an empty id, until the next compaction.
type slot struct {
	id   string
	rows []int32
}

// snapshot is what a reader sees: the first rows rows of b, and the id of
// every subject slot those rows mention. It is never modified; a writer
// publishes a new one.
type snapshot struct {
	b     *block
	rows  int
	names []string
}

// New makes an empty index for vectors of dim values.
func New(dim int, o Options) (*Index, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("index: dimension %d", dim)
	}
	if o.Capacity < 0 || o.Workers < 0 {
		return nil, errors.New("index: negative option")
	}
	ix := &Index{dim: dim, workers: o.Workers, onHeap: o.OnHeap || !canMap, byID: map[string]uint32{}}
	if ix.workers == 0 {
		ix.workers = runtime.GOMAXPROCS(0)
	}
	b, err := ix.newBlock(o.Capacity)
	if err != nil {
		return nil, err
	}
	ix.snap.Store(&snapshot{b: b})
	ix.queries.New = func() any { v := make([]float32, dim); return &v }
	ix.scores.New = func() any { v := make([]float32, chunkRows); return &v }
	return ix, nil
}

// newBlock allocates room for rows templates (0 picks the default). If the
// kernel refuses the mapping, the index carries on with heap memory.
func (ix *Index) newBlock(rows int) (*block, error) {
	if !ix.onHeap {
		if rows == 0 {
			rows = defaultMappedRows
		}
		if b, err := mappedBlock(ix.dim, rows); err == nil {
			return b, nil
		}
		ix.onHeap = true
		rows = min(rows, defaultHeapRows)
	}
	if rows == 0 {
		rows = defaultHeapRows
	}
	if rows > math.MaxInt32 || rows*ix.dim/ix.dim != rows {
		return nil, fmt.Errorf("index: cannot hold %d rows of %d values", rows, ix.dim)
	}
	return heapBlock(ix.dim, rows), nil
}

// Dim is the length of the vectors the index holds.
func (ix *Index) Dim() int { return ix.dim }

// Stats reports the size of the index.
func (ix *Index) Stats() Stats {
	ix.wmu.Lock()
	defer ix.wmu.Unlock()
	st := Stats{Dim: ix.dim, OnHeap: ix.onHeap, Kernel: kernelName, Dead: ix.deadRows}
	if s := ix.snap.Load(); s != nil {
		st.Capacity = s.b.cap
	}
	ix.smu.RLock()
	st.Subjects, st.Templates = ix.subjects, ix.live
	ix.smu.RUnlock()
	return st
}

// Has reports whether the subject has at least one template.
func (ix *Index) Has(subject string) bool {
	ix.smu.RLock()
	defer ix.smu.RUnlock()
	_, ok := ix.byID[subject]
	return ok
}

// Templates lists the template ids of a subject, oldest first.
func (ix *Index) Templates(subject string) []uint64 {
	ix.life.RLock()
	defer ix.life.RUnlock()
	s := ix.snap.Load()
	if s == nil {
		return nil
	}
	ix.smu.RLock()
	defer ix.smu.RUnlock()
	si, ok := ix.byID[subject]
	if !ok {
		return nil
	}
	ids := make([]uint64, len(ix.slots[si].rows))
	for i, r := range ix.slots[si].rows {
		ids[i] = s.b.tmpl[r]
	}
	return ids
}

// unitScale returns the factor that makes v a unit vector.
func unitScale(v []float32) (float32, error) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if !(sum > 0) || math.IsInf(sum, 0) {
		return 0, ErrVector
	}
	return float32(1 / math.Sqrt(sum)), nil
}

// Add stores a template for a subject, creating the subject if need be. The
// vector is scaled to unit length; it must have Dim values, not all zero.
// The template id is the caller's and must be new for this subject.
func (ix *Index) Add(subject string, template uint64, vector []float32) error {
	if subject == "" {
		return ErrSubject
	}
	if len(vector) != ix.dim {
		return fmt.Errorf("%w: %d values, want %d", ErrDimension, len(vector), ix.dim)
	}
	scale, err := unitScale(vector)
	if err != nil {
		return err
	}

	ix.wmu.Lock()
	defer ix.wmu.Unlock()
	s := ix.snap.Load()
	if s == nil {
		return ErrClosed
	}
	// Only writers change the subject table, and this is the only writer.
	si, known := ix.byID[subject]
	if known {
		for _, r := range ix.slots[si].rows {
			if s.b.tmpl[r] == template {
				return fmt.Errorf("%w: %s/%d", ErrExists, subject, template)
			}
		}
	}
	if s.rows == s.b.cap {
		if s, err = ix.grow(s); err != nil {
			return err
		}
	}

	// Fill in the row past the published ones, where no reader looks yet.
	row := s.rows
	dst := s.b.vec[row*ix.dim : (row+1)*ix.dim]
	for i, v := range vector {
		dst[i] = v * scale
	}
	s.b.tmpl[row] = template
	names := s.names
	if !known {
		si = uint32(len(names))
		names = append(names, subject)
	}
	atomic.StoreUint32(&s.b.owner[row], si)
	ix.snap.Store(&snapshot{b: s.b, rows: row + 1, names: names})

	// The subject table lists a row only once readers can see it.
	ix.smu.Lock()
	if !known {
		ix.byID[subject] = si
		ix.slots = append(ix.slots, slot{id: subject})
		ix.subjects++
	}
	ix.slots[si].rows = append(ix.slots[si].rows, int32(row))
	ix.live++
	ix.smu.Unlock()
	return nil
}

// Remove deletes one template and reports whether it was there. A subject
// whose last template is removed is gone.
func (ix *Index) Remove(subject string, template uint64) bool {
	return ix.remove(subject, func(id uint64) bool { return id == template }) > 0
}

// RemoveSubject deletes a subject with all its templates and returns how
// many there were.
func (ix *Index) RemoveSubject(subject string) int {
	return ix.remove(subject, func(uint64) bool { return true })
}

func (ix *Index) remove(subject string, doomed func(template uint64) bool) int {
	ix.wmu.Lock()
	defer ix.wmu.Unlock()
	s := ix.snap.Load()
	if s == nil {
		return 0
	}
	si, ok := ix.byID[subject]
	if !ok {
		return 0
	}
	var keep, drop []int32
	for _, r := range ix.slots[si].rows {
		if doomed(s.b.tmpl[r]) {
			drop = append(drop, r)
		} else {
			keep = append(keep, r)
		}
	}
	if len(drop) == 0 {
		return 0
	}

	ix.smu.Lock()
	ix.slots[si].rows = keep
	ix.live -= len(drop)
	if len(keep) == 0 {
		delete(ix.byID, subject)
		ix.slots[si].id = ""
		ix.subjects--
	}
	ix.smu.Unlock()
	for _, r := range drop {
		atomic.StoreUint32(&s.b.owner[r], dead)
	}
	ix.deadRows += len(drop)

	// Reclaim the rows once a quarter of the index is dead weight.
	if ix.deadRows > 1024 && ix.deadRows*4 > s.rows {
		_ = ix.compact() // on failure the index simply stays as it is
	}
	return len(drop)
}

// Compact reclaims the rows of removed templates. It happens by itself
// when enough have piled up; searches carry on meanwhile.
func (ix *Index) Compact() error {
	ix.wmu.Lock()
	defer ix.wmu.Unlock()
	if ix.snap.Load() == nil {
		return ErrClosed
	}
	if ix.deadRows == 0 {
		return nil
	}
	return ix.compact()
}

// compact copies the live rows into a fresh block and swaps it in. The
// writer lock is held, so nothing changes during the copy; readers keep
// using the old block until the swap, and the swap waits for them.
func (ix *Index) compact() error {
	s := ix.snap.Load()
	nb, err := ix.newBlock(s.b.cap)
	if err != nil {
		return err
	}
	remap := make([]uint32, len(s.names))
	for i := range remap {
		remap[i] = dead
	}
	names := make([]string, 0, ix.subjects)
	slots := make([]slot, 0, ix.subjects)
	byID := make(map[string]uint32, ix.subjects)
	n := 0
	for r := 0; r < s.rows; r++ {
		owner := atomic.LoadUint32(&s.b.owner[r])
		if owner == dead {
			continue
		}
		to := remap[owner]
		if to == dead {
			to = uint32(len(names))
			remap[owner] = to
			names = append(names, s.names[owner])
			slots = append(slots, slot{id: s.names[owner]})
			byID[s.names[owner]] = to
		}
		copy(nb.vec[n*ix.dim:(n+1)*ix.dim], s.b.vec[r*ix.dim:(r+1)*ix.dim])
		nb.tmpl[n] = s.b.tmpl[r]
		nb.owner[n] = to
		slots[to].rows = append(slots[to].rows, int32(n))
		n++
	}

	ix.life.Lock()
	ix.smu.Lock()
	ix.byID, ix.slots = byID, slots
	ix.snap.Store(&snapshot{b: nb, rows: n, names: names})
	ix.smu.Unlock()
	s.b.release()
	ix.life.Unlock()
	ix.deadRows = 0
	return nil
}

// grow moves the rows into a block twice the size. Row numbers stay, so
// the subject table is untouched.
func (ix *Index) grow(s *snapshot) (*snapshot, error) {
	nb, err := ix.newBlock(s.b.cap * 2)
	if err != nil {
		return nil, err
	}
	copy(nb.vec, s.b.vec[:s.rows*ix.dim])
	copy(nb.tmpl, s.b.tmpl[:s.rows])
	for r := 0; r < s.rows; r++ {
		nb.owner[r] = atomic.LoadUint32(&s.b.owner[r])
	}
	grown := &snapshot{b: nb, rows: s.rows, names: s.names}
	ix.life.Lock()
	ix.snap.Store(grown)
	s.b.release()
	ix.life.Unlock()
	return grown, nil
}

// Close releases the index's memory. Calls in flight finish first; later
// ones fail with ErrClosed.
func (ix *Index) Close() {
	ix.wmu.Lock()
	defer ix.wmu.Unlock()
	ix.life.Lock()
	defer ix.life.Unlock()
	if s := ix.snap.Swap(nil); s != nil {
		s.b.release()
	}
}
