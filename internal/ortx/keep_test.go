package ortx

import (
	"runtime"
	"sync"
	"testing"
)

func TestKeepNeverDropsAWorker(t *testing.T) {
	type worker struct{ id int }
	var k Keep[worker]
	if k.Get() != nil {
		t.Fatal("an empty Keep handed out a worker")
	}
	a, b := &worker{1}, &worker{2}
	k.Put(a)
	k.Put(b)
	// a sync.Pool would have forgotten them by now
	runtime.GC()
	runtime.GC()
	runtime.GC()
	if got := k.Get(); got != b {
		t.Errorf("got %v, want the worker put back last", got)
	}
	if got := k.Get(); got != a {
		t.Errorf("got %v, want the other worker", got)
	}
	if k.Get() != nil {
		t.Error("a third worker out of two")
	}

	k.Put(a)
	k.Put(b)
	var released []int
	k.Drain(func(w *worker) { released = append(released, w.id) })
	if len(released) != 2 || k.Get() != nil {
		t.Errorf("drained %v, and the Keep still has %v", released, k.Get())
	}
	k.Drain(func(*worker) { t.Error("drained a worker out of an empty Keep") })
}

func TestKeepGrowsToThePeakAndNoFurther(t *testing.T) {
	type worker struct{}
	var (
		k    Keep[worker]
		made int
		mu   sync.Mutex
		wg   sync.WaitGroup
	)
	for round := 0; round < 50; round++ {
		gate := make(chan struct{})
		for i := 0; i < 8; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				w := k.Get()
				if w == nil {
					mu.Lock()
					made++
					mu.Unlock()
					w = &worker{}
				}
				<-gate // all eight hold a worker at once
				k.Put(w)
			}()
		}
		close(gate)
		wg.Wait()
	}
	if made > 8 {
		t.Errorf("%d workers were made for eight callers at a time", made)
	}
}
