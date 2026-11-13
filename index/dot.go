package index

import "unsafe"

// A block kernel computes dst[i] = q · rows[i*dim:(i+1)*dim] for n rows.
// dim is a positive multiple of the kernel's step.
type blockKernel func(q, rows, dst unsafe.Pointer, dim, n int)

// The kernel in use, chosen once at start-up. nil means there is no vector
// kernel for this machine and the portable loop does all the work.
var (
	kernel     blockKernel
	kernelStep int
	kernelName = "portable"
)

// Kernel names the dot-product implementation in use: "avx2", "neon" or
// "portable".
func Kernel() string { return kernelName }

// dotRows writes the dot product of q with every dim-long row of rows.
func dotRows(dst, q, rows []float32) {
	n, dim := len(dst), len(q)
	if n == 0 {
		return
	}
	_ = rows[n*dim-1] // the kernels trust this
	if kernel != nil && dim%kernelStep == 0 {
		kernel(unsafe.Pointer(&q[0]), unsafe.Pointer(&rows[0]), unsafe.Pointer(&dst[0]), dim, n)
		return
	}
	dotRowsPortable(dst, q, rows)
}

func dotRowsPortable(dst, q, rows []float32) {
	dim := len(q)
	for i := range dst {
		dst[i] = dot(q, rows[i*dim:(i+1)*dim])
	}
}

// dot is the portable dot product. Four independent sums let the processor
// overlap the multiply-adds; the compiler does not vectorise the loop.
func dot(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	b = b[:len(a)]
	n := len(a) &^ 3
	for i := 0; i < n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for i := n; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}
