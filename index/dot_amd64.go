package index

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

//go:noescape
func dotBlockAVX2(q, rows, dst unsafe.Pointer, dim, n int)

func init() {
	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		kernel, kernelStep, kernelName = dotBlockAVX2, 32, "avx2"
	}
}
