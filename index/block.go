package index

// dead marks a row whose template has been removed.
const dead = ^uint32(0)

// block is the storage behind one generation of the index: room for cap
// rows, each a vector with the slot of the subject it belongs to and the id
// of its template.
type block struct {
	cap   int
	vec   []float32 // cap * dim values
	owner []uint32  // subject slot of each row, or dead; read and written atomically once published
	tmpl  []uint64
	free  func() // returns the memory; nil when the garbage collector owns it
}

func (b *block) release() {
	if b.free != nil {
		b.free()
	}
}

// Rows reserved by default. Off the heap this is only address space: pages
// are committed as rows are written. On the heap it is real memory, so the
// start is modest and the block doubles when full.
const (
	defaultMappedRows = 1 << 20
	defaultHeapRows   = 1 << 10
)

func heapBlock(dim, rows int) *block {
	return &block{
		cap:   rows,
		vec:   make([]float32, rows*dim),
		owner: make([]uint32, rows),
		tmpl:  make([]uint64, rows),
	}
}
