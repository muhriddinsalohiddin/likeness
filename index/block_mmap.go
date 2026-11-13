//go:build linux || darwin

package index

import (
	"fmt"
	"math"
	"unsafe"

	"golang.org/x/sys/unix"
)

const canMap = true

// mappedBlock takes anonymous memory straight from the kernel. The garbage
// collector never sees it, so a large index does not make the collector
// let the heap grow to twice its size; and untouched pages cost nothing,
// so room can be reserved generously.
func mappedBlock(dim, rows int) (*block, error) {
	vecBytes := rows * dim * 4
	tmplAt := (vecBytes + 7) &^ 7
	ownerAt := tmplAt + rows*8
	total := ownerAt + rows*4
	if rows <= 0 || dim <= 0 || rows > math.MaxInt32 || total/rows < dim*4 {
		return nil, fmt.Errorf("index: cannot map %d rows of %d values", rows, dim)
	}
	mem, err := unix.Mmap(-1, 0, total, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_ANON|unix.MAP_PRIVATE|unix.MAP_NORESERVE)
	if err != nil {
		return nil, fmt.Errorf("index: mapping %d bytes: %w", total, err)
	}
	return &block{
		cap:   rows,
		vec:   unsafe.Slice((*float32)(unsafe.Pointer(&mem[0])), rows*dim),
		tmpl:  unsafe.Slice((*uint64)(unsafe.Pointer(&mem[tmplAt])), rows),
		owner: unsafe.Slice((*uint32)(unsafe.Pointer(&mem[ownerAt])), rows),
		free:  func() { _ = unix.Munmap(mem) },
	}, nil
}
