package likeness

import (
	"runtime"
	"testing"

	"github.com/muhriddinsalohiddin/likeness/internal/fixtures"
	"github.com/muhriddinsalohiddin/likeness/internal/ortx"
)

// Tensors live outside the Go heap. They once sat in a sync.Pool, which the
// garbage collector empties without freeing them: a server under load grew
// by gigabytes a minute. However often the collector runs, the engine must
// hold a tensor or two for each picture it analyses at once, and none when
// it is closed.
func TestEngineDoesNotLeakTensors(t *testing.T) {
	for _, preset := range presets {
		t.Run(preset, func(t *testing.T) {
			before := ortx.LiveTensors()
			e := open(t, preset)
			picture, err := fixtures.Closeup("hathaway")
			if err != nil {
				t.Fatal(err)
			}
			analyze(t, e, picture, ForSearch())
			settled := ortx.LiveTensors()
			if settled == before {
				t.Fatal("the engine analysed a picture without a tensor: this test watches the wrong thing")
			}
			for i := 0; i < 40; i++ {
				runtime.GC()
				runtime.GC()
				analyze(t, e, picture, ForSearch())
			}
			if now := ortx.LiveTensors(); now != settled {
				t.Errorf("%d tensors after one picture, %d after forty more with the collector running", settled-before, now-before)
			}
			e.Close()
			if left := ortx.LiveTensors(); left != before {
				t.Errorf("%d tensors left after the engine was closed", left-before)
			}
		})
	}
}
