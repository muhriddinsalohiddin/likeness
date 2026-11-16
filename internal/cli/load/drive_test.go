package load

import (
	"context"
	"io"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/muhriddinsalohiddin/likeness/internal/cli"
)

func TestTimetable(t *testing.T) {
	const ms = time.Millisecond
	for _, c := range []struct {
		name     string
		table    timetable
		due      []time.Duration
		measured int // how many of them, the last ones, are counted
	}{
		{"four a second", timetable{rate: 4, length: time.Second},
			[]time.Duration{0, 250 * ms, 500 * ms, 750 * ms}, 4},
		{"with a warm-up", timetable{rate: 2, warmup: time.Second, length: time.Second},
			[]time.Duration{0, 500 * ms, 1000 * ms, 1500 * ms}, 2},
		{"a warm-up between two requests", timetable{rate: 2, warmup: 600 * ms, length: 900 * ms},
			[]time.Duration{0, 500 * ms, 1000 * ms}, 1},
		{"one every two seconds", timetable{rate: 0.5, length: 5 * time.Second},
			[]time.Duration{0, 2 * time.Second, 4 * time.Second}, 3},
		{"a rate that does not divide the second", timetable{rate: 3, length: time.Second},
			[]time.Duration{0, 333333333, 666666666}, 3},
		{"a run shorter than the gap", timetable{rate: 1, length: 100 * ms},
			[]time.Duration{0}, 1},
	} {
		var due []time.Duration
		measured := 0
		for i := 0; ; i++ {
			at, counted, ok := c.table.slot(i)
			if !ok {
				break
			}
			if counted {
				measured++
			} else if measured > 0 {
				t.Errorf("%s: request %d is not counted, an earlier one is", c.name, i)
			}
			due = append(due, at)
		}
		if !slices.Equal(due, c.due) || measured != c.measured {
			t.Errorf("%s: due at %v with %d counted, want %v with %d", c.name, due, measured, c.due, c.measured)
		}
	}

	// rates and durations as they are typed: no request too many or too few
	for _, c := range []struct {
		rate   float64
		length time.Duration
		want   int
	}{
		{200, 300 * ms, 60}, {200, 500 * ms, 100}, {400, 250 * ms, 100}, {20, 250 * ms, 5},
		{7, 10 * time.Second, 70}, {0.1, 10 * time.Second, 1}, {100000, 10 * ms, 1000},
	} {
		table := timetable{rate: c.rate, length: c.length}
		n := 0
		for _, _, ok := table.slot(n); ok; _, _, ok = table.slot(n) {
			n++
		}
		if n != c.want {
			t.Errorf("%v a second for %v: %d requests, want %d", c.rate, c.length, n, c.want)
		}
	}
}

func TestGate(t *testing.T) {
	g := &gate{room: 2}
	for i := 1; i <= 4; i++ {
		if got, want := g.enter(), i <= 2; got != want {
			t.Fatalf("request %d of four at a gate with room for two: let in %v", i, got)
		}
	}
	g.leave()
	if !g.enter() {
		t.Error("no room after one left")
	}
	g.leave()
	g.leave()
	if g.inside.Load() != 0 {
		t.Errorf("%d inside at the end", g.inside.Load())
	}
}

func TestSleep(t *testing.T) {
	if !sleep(background, time.Millisecond) || !sleep(background, 0) || !sleep(background, -time.Second) {
		t.Error("an undisturbed sleep was reported as cut short")
	}
	ctx, cancel := context.WithCancel(background)
	cancel()
	start := time.Now()
	if sleep(ctx, time.Minute) || sleep(ctx, 0) {
		t.Error("a sleep under a context that is done was reported as undisturbed")
	}
	if time.Since(start) > 30*time.Second {
		t.Error("the sleep did not end with its context")
	}
}

func TestMeasured(t *testing.T) {
	o := &options{warmup: time.Second, duration: 10 * time.Second}
	start := time.Now()
	for stopped, want := range map[time.Duration]time.Duration{
		0:                      0,
		500 * time.Millisecond: 0, // still warming up
		4 * time.Second:        3 * time.Second,
		11 * time.Second:       10 * time.Second,
		12 * time.Second:       10 * time.Second, // the requests in flight are not part of it
	} {
		if got := o.measured(start, start.Add(stopped)); got != want {
			t.Errorf("stopped after %v: measured for %v, want %v", stopped, got, want)
		}
	}
}

func TestParse(t *testing.T) {
	files := pictures(t, "face:anna", "face:boris")
	env := cli.Env{Getenv: vars(keysVar, searchKey+",another-key-0123456789"), Stdout: io.Discard, Stderr: io.Discard}

	o, err := parse(append([]string{"--url", "http://localhost:8080"}, files...), env)
	if err != nil {
		t.Fatal(err)
	}
	if o.search.String() != "http://localhost:8080/v1/search" || o.key != searchKey || o.concurrency != 4 || o.rates != nil || o.stepped ||
		o.maxPending != 512 || o.duration != 10*time.Second || o.warmup != time.Second || o.timeout != 5*time.Second ||
		o.target != 100*time.Millisecond || o.json {
		t.Errorf("defaults: %+v", o)
	}
	if len(o.pictures) != 2 || string(o.pictures[0]) != "face:anna" || string(o.pictures[1]) != "face:boris" {
		t.Errorf("pictures: %q", o.pictures)
	}

	o, err = parse(append([]string{
		"-url", "https://faces.example/api", "-rate", "12.5", "-max-pending", "64", "-duration", "30s", "-warmup", "0",
		"-timeout", "2s", "-target", "250ms", "-json",
	}, files...), env)
	if err != nil {
		t.Fatal(err)
	}
	if o.search.String() != "https://faces.example/api/v1/search" || !slices.Equal(o.rates, []float64{12.5}) || o.stepped ||
		o.maxPending != 64 || o.duration != 30*time.Second || o.warmup != 0 || o.timeout != 2*time.Second ||
		o.target != 250*time.Millisecond || !o.json {
		t.Errorf("flags: %+v", o)
	}

	o, err = parse(append([]string{"--url", "http://localhost:8080", "--steps", "5,10, 20 ,0.5"}, files...), env)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(o.rates, []float64{5, 10, 20, 0.5}) || !o.stepped {
		t.Errorf("steps: %+v", o)
	}
}

func TestSearchURL(t *testing.T) {
	for address, want := range map[string]string{
		"http://localhost:8080":           "http://localhost:8080/v1/search",
		"http://localhost:8080/":          "http://localhost:8080/v1/search",
		"https://faces.example/api":       "https://faces.example/api/v1/search",
		"https://faces.example/a/b/":      "https://faces.example/a/b/v1/search",
		"http://[::1]:8080":               "http://[::1]:8080/v1/search",
		"http://localhost:8080?limit=3":   "http://localhost:8080/v1/search?limit=3",
		"http://user:secret@localhost/up": "http://user:secret@localhost/up/v1/search",
	} {
		if got, err := searchURL(address); err != nil || got.String() != want {
			t.Errorf("%q: %v %v, want %s", address, got, err, want)
		}
	}
	// as pasted, with a space around it
	if got, err := searchURL(" https://faces.example\n"); err != nil || got.String() != "https://faces.example/v1/search" {
		t.Errorf("an address with space around it: %v %v", got, err)
	}
	for _, address := range []string{
		"", "localhost", "localhost:8080", "//localhost:8080", "ftp://localhost", "http://", "http:///v1",
		"http://localhost:8080/#top", "http://local host", "http://user:secret@:80:90/",
	} {
		got, err := searchURL(address)
		if err == nil {
			t.Errorf("%q: accepted as %v", address, got)
			continue
		}
		if strings.Contains(err.Error(), "secret") {
			t.Errorf("%q: the address is repeated: %v", address, err)
		}
	}
}

func TestFirstKey(t *testing.T) {
	for list, want := range map[string]string{
		"":                 "",
		" ,\n, ":           "",
		"one":              "one",
		"one,two":          "one",
		" one , two":       "one",
		",,one":            "one",
		"one\ntwo\r\n":     "one",
		"\n  one\n\n two ": "one",
	} {
		if got, err := firstKey(list); err != nil || got != want {
			t.Errorf("%q: %q %v, want %q", list, got, err, want)
		}
	}
	for _, list := range []string{"one two,three", "kéy", "tab\there"} {
		if got, err := firstKey(list); err == nil || strings.Contains(err.Error(), list) {
			t.Errorf("%q: %q %v", list, got, err)
		}
	}
}

func TestParseSteps(t *testing.T) {
	for list, want := range map[string][]float64{
		"5":             {5},
		"5,10,20,40,80": {5, 10, 20, 40, 80},
		"5 , 10":        {5, 10},
		"80,5":          {80, 5},
		"0.5,1.5":       {0.5, 1.5},
		"1e2":           {100},
	} {
		if got, err := parseSteps(list); err != nil || !slices.Equal(got, want) {
			t.Errorf("%q: %v %v, want %v", list, got, err, want)
		}
	}
	for _, list := range []string{"", ",", "5,", "five", "5;10", "5,-10", "0", "5,Inf", "NaN", "1e9"} {
		if got, err := parseSteps(list); err == nil {
			t.Errorf("%q: accepted as %v", list, got)
		}
	}
}
