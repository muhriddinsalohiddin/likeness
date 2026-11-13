package index

import "unsafe"

//go:noescape
func dotBlockNEON(q, rows, dst unsafe.Pointer, dim, n int)

func init() {
	// NEON is part of the baseline on arm64
	kernel, kernelStep, kernelName = dotBlockNEON, 16, "neon"
}
