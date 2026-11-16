package ortx

import "sync"

// Keep holds idle workers between calls: what a sync.Pool is used for,
// except that it never lets one go. A worker owns tensors, and tensors live
// outside the Go heap. The garbage collector empties a sync.Pool whenever
// it runs and would drop the only reference to that memory with it; under
// load that lost several megabytes a request.
//
// A Keep grows to the largest number of workers ever in use at once and
// stays there. The zero value is ready to use.
type Keep[T any] struct {
	mu   sync.Mutex
	idle []*T
}

// Get returns an idle worker, or nil when there is none and the caller has
// to make one.
func (k *Keep[T]) Get() *T {
	k.mu.Lock()
	defer k.mu.Unlock()
	n := len(k.idle)
	if n == 0 {
		return nil
	}
	w := k.idle[n-1]
	k.idle[n-1] = nil
	k.idle = k.idle[:n-1]
	return w
}

// Put hands a worker back.
func (k *Keep[T]) Put(w *T) {
	k.mu.Lock()
	k.idle = append(k.idle, w)
	k.mu.Unlock()
}

// Drain takes every idle worker out and passes it to release, which frees
// what the worker owns.
func (k *Keep[T]) Drain(release func(*T)) {
	k.mu.Lock()
	idle := k.idle
	k.idle = nil
	k.mu.Unlock()
	for _, w := range idle {
		release(w)
	}
}
