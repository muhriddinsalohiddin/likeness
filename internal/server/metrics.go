package server

import (
	"bufio"
	"bytes"
	"io"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// The metrics are written in the Prometheus text format by hand: a dozen
// series do not justify a client library and what it brings along.

// counters is a family of counters with up to two labels.
type counters struct {
	mu sync.Mutex
	n  map[[2]string]uint64
}

func (c *counters) add(a, b string) {
	c.mu.Lock()
	if c.n == nil {
		c.n = make(map[[2]string]uint64)
	}
	c.n[[2]string{a, b}]++
	c.mu.Unlock()
}

func (c *counters) get(a, b string) uint64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.n[[2]string{a, b}]
}

type sample struct {
	labels [2]string
	value  uint64
}

func (c *counters) samples() []sample {
	c.mu.Lock()
	out := make([]sample, 0, len(c.n))
	for labels, v := range c.n {
		out = append(out, sample{labels, v})
	}
	c.mu.Unlock()
	sort.Slice(out, func(i, j int) bool {
		if out[i].labels[0] != out[j].labels[0] {
			return out[i].labels[0] < out[j].labels[0]
		}
		return out[i].labels[1] < out[j].labels[1]
	})
	return out
}

// histograms is a family of histograms with one label and shared bounds.
type histograms struct {
	bounds []float64 // upper bounds in seconds, ascending
	mu     sync.Mutex
	h      map[string]*histogram
}

type histogram struct {
	counts []uint64 // one per bound, and a last one for everything above
	sum    float64
	n      uint64
}

func (hs *histograms) observe(label string, seconds float64) {
	i := sort.SearchFloat64s(hs.bounds, seconds)
	hs.mu.Lock()
	if hs.h == nil {
		hs.h = make(map[string]*histogram)
	}
	h := hs.h[label]
	if h == nil {
		h = &histogram{counts: make([]uint64, len(hs.bounds)+1)}
		hs.h[label] = h
	}
	h.counts[i]++
	h.sum += seconds
	h.n++
	hs.mu.Unlock()
}

type metrics struct {
	requests  counters // route, status code
	results   counters // route, result
	rejects   counters // route, reason a face was turned down
	refusals  counters // cause of a 429
	denials   counters // why a request was not authorised
	durations histograms
	stages    histograms
}

func newMetrics() *metrics {
	return &metrics{
		// A request is expected to take tens of milliseconds, a stage a few.
		durations: histograms{bounds: []float64{.005, .01, .025, .05, .075, .1, .15, .25, .5, 1, 2.5}},
		stages:    histograms{bounds: []float64{.0005, .001, .0025, .005, .01, .025, .05, .1, .25, 1}},
	}
}

// gauges are the values read at the moment of a scrape.
type gauges struct {
	running, queued, inside int
	subjects, templates     int
	build                   [][2]string // label name and value
}

// write renders the metrics. The page is put together in memory first: a
// scraper that reads slowly must not hold up the requests being counted.
func (m *metrics) write(out io.Writer, g gauges) error {
	var page bytes.Buffer
	w := bufio.NewWriter(&page)
	head := func(name, kind, help string) {
		w.WriteString("# HELP " + name + " " + help + "\n# TYPE " + name + " " + kind + "\n")
	}
	series := func(name string, labels [][2]string, value string) {
		w.WriteString(name)
		for i, l := range labels {
			if i == 0 {
				w.WriteByte('{')
			} else {
				w.WriteByte(',')
			}
			w.WriteString(l[0] + `="` + labelEscaper.Replace(l[1]) + `"`)
		}
		if len(labels) > 0 {
			w.WriteByte('}')
		}
		w.WriteString(" " + value + "\n")
	}
	gauge := func(name, help string, v int) {
		head(name, "gauge", help)
		series(name, nil, strconv.Itoa(v))
	}
	counter := func(name, help string, c *counters, labels ...string) {
		head(name, "counter", help)
		for _, s := range c.samples() {
			pairs := make([][2]string, len(labels))
			for i, l := range labels {
				pairs[i] = [2]string{l, s.labels[i]}
			}
			series(name, pairs, strconv.FormatUint(s.value, 10))
		}
	}
	histo := func(name, help, label string, hs *histograms) {
		head(name, "histogram", help)
		hs.mu.Lock()
		defer hs.mu.Unlock()
		names := make([]string, 0, len(hs.h))
		for l := range hs.h {
			names = append(names, l)
		}
		sort.Strings(names)
		for _, l := range names {
			h := hs.h[l]
			var below uint64
			for i, bound := range hs.bounds {
				below += h.counts[i]
				series(name+"_bucket", [][2]string{{label, l}, {"le", formatFloat(bound)}}, strconv.FormatUint(below, 10))
			}
			series(name+"_bucket", [][2]string{{label, l}, {"le", "+Inf"}}, strconv.FormatUint(h.n, 10))
			series(name+"_sum", [][2]string{{label, l}}, formatFloat(h.sum))
			series(name+"_count", [][2]string{{label, l}}, strconv.FormatUint(h.n, 10))
		}
	}

	head("likeness_build_info", "gauge", "The version and models this server runs; always 1.")
	series("likeness_build_info", g.build, "1")
	counter("likeness_requests_total", "Requests answered, by route and status code.", &m.requests, "route", "code")
	histo("likeness_request_duration_seconds", "Time from the arrival of a request to its answer.", "route", &m.durations)
	histo("likeness_stage_duration_seconds", "Time spent in each stage of handling a picture.", "stage", &m.stages)
	counter("likeness_results_total", "What picture requests came to, by route.", &m.results, "route", "result")
	counter("likeness_rejected_faces_total", "Pictures turned down, by route and reason.", &m.rejects, "route", "reason")
	counter("likeness_overloaded_total", "Requests refused with 429, by cause.", &m.refusals, "cause")
	counter("likeness_unauthorized_total", "Requests refused for lack of a valid key or of rights, by reason.", &m.denials, "reason")
	gauge("likeness_inflight_requests", "Pictures being analysed right now.", g.running)
	gauge("likeness_queued_requests", "Requests waiting for a free slot.", g.queued)
	gauge("likeness_pending_requests", "Picture requests anywhere between arriving and being answered.", g.inside)
	gauge("likeness_subjects", "People enrolled.", g.subjects)
	gauge("likeness_templates", "Face templates enrolled.", g.templates)
	if err := w.Flush(); err != nil {
		return err
	}
	_, err := out.Write(page.Bytes())
	return err
}

var labelEscaper = strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`)

func formatFloat(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }
