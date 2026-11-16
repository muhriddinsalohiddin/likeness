package bench

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// made is a report with numbers that are easy to recognise.
func made() *report {
	even := func(d time.Duration) spread {
		return spread{P50: millis(d), P90: millis(2 * d), P99: millis(3 * d), Max: millis(4 * d)}
	}
	return &report{
		Version:  "1.2.3",
		Engine:   engineInfo{Preset: "default", Detector: "yunet", Embedder: "sface", Dim: 128, Runtime: "1.22.0", Threads: 8},
		Machine:  machineInfo{OS: "linux", Arch: "amd64", CPUs: 16, Go: "go1.26.0"},
		Picture:  pictureInfo{Name: "built-in sample", Width: 1280, Height: 720, Bytes: 98765},
		Duration: millis(5 * time.Second),
		Stages: stageTimes{
			Pictures: 250,
			Decode:   even(3 * ms), Detect: even(5 * ms), Align: even(250 * time.Microsecond), Embed: even(7 * ms),
			Total: even(16 * ms),
		},
		Search: []searchTimes{
			{Gallery: 1000, Searches: 2000, Kernel: "avx2", P50: millis(4500 * time.Nanosecond), P99: millis(9 * time.Microsecond)},
			{Gallery: 100000, Searches: 1234, Kernel: "avx2", P50: millis(1250 * time.Microsecond), P99: millis(2 * ms)},
		},
		Inflight: []levelTimes{
			{Inflight: 1, Gallery: 100000, Pictures: 300, PerSecond: 60, P50: millis(16 * ms), P99: millis(21 * ms)},
			{Inflight: 2, Gallery: 100000, Pictures: 520, PerSecond: 104.25, P50: millis(19 * ms), P99: millis(64500 * time.Microsecond)},
			{Inflight: 4, Gallery: 100000, Pictures: 610, PerSecond: 122, P50: millis(33 * ms), P99: millis(140 * ms)},
		},
		Recommendation: recommendation{Target: millis(100 * ms), MaxInflight: 2, PerSecond: 104.25, P99: millis(64500 * time.Microsecond)},
	}
}

func TestText(t *testing.T) {
	var buf bytes.Buffer
	if err := made().write(&buf, false); err != nil {
		t.Fatal(err)
	}
	// the columns line up on the right
	want := `likeness 1.2.3
  models   default: yunet and sface, vectors of 128 values
  runtime  ONNX Runtime 1.22.0, 8 threads
  machine  linux/amd64, 16 processors, go1.26.0
  picture  built-in sample, 1280x720, 98765 bytes

One picture at a time: 250 pictures in 5s
   stage  p50 ms  p90 ms  p99 ms  max ms
  decode    3.00    6.00    9.00   12.00
  detect    5.00   10.00   15.00   20.00
   align    0.25    0.50    0.75    1.00
   embed    7.00   14.00   21.00   28.00
   total   16.00   32.00   48.00   64.00

Search: the face of the picture among random ones, one search at a time
  gallery  searches  p50 µs  p99 µs  kernel
     1000      2000     4.5     9.0    avx2
   100000      1234  1250.0  2000.0    avx2

Several at once: each picture analysed and searched for, 5s a level
  at once  gallery  pictures  pictures/s  p50 ms  p99 ms
        1   100000       300        60.0   16.00   21.00
        2   100000       520       104.2   19.00   64.50
        4   100000       610       122.0   33.00  140.00

Recommendation: run the server with --max-inflight 2
  2 at once: 104.2 pictures per second, 99th percentile 64.50 ms, within the 100 ms asked for.
`
	if got := buf.String(); got != want {
		t.Errorf("the report reads:\n%s\nwant:\n%s", got, want)
	}
}

func TestTextWithoutARecommendation(t *testing.T) {
	r := made()
	r.Recommendation = recommend(r.Inflight, 20*ms)
	var buf bytes.Buffer
	if err := r.write(&buf, false); err != nil {
		t.Fatal(err)
	}
	want := "\nRecommendation: none. No level keeps the 99th percentile within 20 ms on this machine;\n" +
		"  the lowest is 21.00 ms, with 1 at once.\n"
	if got := buf.String(); !strings.HasSuffix(got, want) || strings.Contains(got, "--max-inflight") {
		t.Errorf("the report ends:\n%s\nwant:\n%s", got, want)
	}

	// a report with nothing in it is still a report
	buf.Reset()
	if err := (&report{}).write(&buf, false); err != nil {
		t.Fatal(err)
	}
	if got := buf.String(); !strings.HasSuffix(got, "within 0 ms on this machine.\n") {
		t.Errorf("an empty report ends:\n%s", got)
	}
}

func TestJSON(t *testing.T) {
	var buf bytes.Buffer
	if err := made().write(&buf, true); err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(buf.Bytes(), &got); err != nil {
		t.Fatalf("%v\n%s", err, buf.String())
	}
	// at walks the document: names for objects, numbers for lists
	at := func(path ...any) any {
		t.Helper()
		var v any = got
		for _, step := range path {
			switch step := step.(type) {
			case string:
				v = v.(map[string]any)[step]
			case int:
				v = v.([]any)[step]
			}
		}
		return v
	}
	for _, c := range []struct {
		path []any
		want any
	}{
		{[]any{"version"}, "1.2.3"},
		{[]any{"engine", "preset"}, "default"},
		{[]any{"engine", "detector"}, "yunet"},
		{[]any{"engine", "embedder"}, "sface"},
		{[]any{"engine", "dim"}, 128.0},
		{[]any{"engine", "runtime"}, "1.22.0"},
		{[]any{"engine", "threads"}, 8.0},
		{[]any{"machine", "os"}, "linux"},
		{[]any{"machine", "arch"}, "amd64"},
		{[]any{"machine", "cpus"}, 16.0},
		{[]any{"machine", "go"}, "go1.26.0"},
		{[]any{"picture", "name"}, "built-in sample"},
		{[]any{"picture", "width"}, 1280.0},
		{[]any{"picture", "height"}, 720.0},
		{[]any{"picture", "bytes"}, 98765.0},
		{[]any{"duration_ms"}, 5000.0},
		{[]any{"stages", "pictures"}, 250.0},
		{[]any{"stages", "decode", "p50_ms"}, 3.0},
		{[]any{"stages", "detect", "p90_ms"}, 10.0},
		{[]any{"stages", "align", "p50_ms"}, 0.25},
		{[]any{"stages", "embed", "p99_ms"}, 21.0},
		{[]any{"stages", "total", "max_ms"}, 64.0},
		{[]any{"search", 0, "gallery"}, 1000.0},
		{[]any{"search", 0, "searches"}, 2000.0},
		{[]any{"search", 0, "kernel"}, "avx2"},
		{[]any{"search", 0, "p50_ms"}, 0.0045},
		{[]any{"search", 1, "p99_ms"}, 2.0},
		{[]any{"inflight", 1, "inflight"}, 2.0},
		{[]any{"inflight", 1, "gallery"}, 100000.0},
		{[]any{"inflight", 1, "pictures"}, 520.0},
		{[]any{"inflight", 1, "pictures_per_second"}, 104.25},
		{[]any{"inflight", 1, "p50_ms"}, 19.0},
		{[]any{"inflight", 2, "p99_ms"}, 140.0},
		{[]any{"recommendation", "target_ms"}, 100.0},
		{[]any{"recommendation", "max_inflight"}, 2.0},
		{[]any{"recommendation", "pictures_per_second"}, 104.25},
		{[]any{"recommendation", "p99_ms"}, 64.5},
	} {
		if v := at(c.path...); v != c.want {
			t.Errorf("%v is %v, want %v", c.path, v, c.want)
		}
	}
}

func TestMillis(t *testing.T) {
	for d, want := range map[time.Duration]string{
		0:                       "0",
		2 * ms:                  "2",
		1500 * time.Microsecond: "1.5",
		41583 * time.Nanosecond: "0.041583",
		time.Minute:             "60000",
		time.Nanosecond:         "0.000001",
	} {
		if got := millis(d).String(); got != want {
			t.Errorf("%s: %s ms, want %s", d, got, want)
		}
		if b, err := json.Marshal(millis(d)); err != nil || string(b) != want {
			t.Errorf("%s in JSON: %s %v, want %s", d, b, err, want)
		}
	}
}

func TestCount(t *testing.T) {
	for n, want := range map[int]string{0: "0 threads", 1: "1 thread", 2: "2 threads", 64: "64 threads"} {
		if got := count(n, "thread"); got != want {
			t.Errorf("%d: %q, want %q", n, got, want)
		}
	}
}
