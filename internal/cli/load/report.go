package load

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"sync"
	"text/tabwriter"
	"time"
)

// stageNames are the parts of the server's timing_ms the report shows, in
// the order a request passes through them.
var stageNames = []string{"queue", "decode", "detect", "embed", "search", "total"}

// tally collects what became of the requests of one run.
type tally struct {
	mu sync.Mutex
	// accepted and refused hold the latencies of the 200s and the 429s,
	// in milliseconds.
	accepted, refused []float64
	other             map[int]int
	failed, timedOut  int
	skipped           int
	unreadable        int
	stages            map[string][]float64
	results           results
	// trouble is the first request that failed without an answer.
	trouble error
}

func newTally() *tally {
	return &tally{other: make(map[int]int), stages: make(map[string][]float64)}
}

func (t *tally) add(x exchange) {
	t.mu.Lock()
	defer t.mu.Unlock()
	switch {
	case x.err != nil && timedOut(x.err):
		t.timedOut++
	case x.err != nil:
		t.failed++
		if t.trouble == nil {
			t.trouble = x.err
		}
	case x.status == http.StatusOK:
		t.accepted = append(t.accepted, millis(x.took))
		t.read(x.answer)
	case x.status == http.StatusTooManyRequests:
		t.refused = append(t.refused, millis(x.took))
	default:
		t.other[x.status]++
	}
}

// read takes in what a 200 said.
func (t *tally) read(a *answer) {
	if a == nil {
		t.unreadable++
		return
	}
	for _, name := range stageNames {
		if took, ok := a.Timing[name]; ok {
			t.stages[name] = append(t.stages[name], took)
		}
	}
	switch {
	case a.Reject != "", len(a.Faces) > 0 && a.Faces[0].Reject != "":
		t.results.Rejected++
	case len(a.Faces) > 0 && a.Faces[0].Match:
		t.results.Match++
	default:
		t.results.NoMatch++
	}
}

// skip counts a request of an open loop that was due and not sent.
func (t *tally) skip() {
	t.mu.Lock()
	t.skipped++
	t.mu.Unlock()
}

// timedOut tells a request that ran out of time from one that failed.
func timedOut(err error) bool {
	var limit net.Error
	return errors.Is(err, context.DeadlineExceeded) || (errors.As(err, &limit) && limit.Timeout())
}

func millis(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

// rounded keeps the microseconds of a number of milliseconds and drops the
// noise below them.
func rounded(ms float64) float64 { return math.Round(ms*1000) / 1000 }

// sorted returns the values in rising order, leaving the slice it was
// given as it is.
func sorted(values []float64) []float64 {
	out := slices.Clone(values)
	slices.Sort(out)
	return out
}

// percentile returns the p-th percentile of values in rising order by
// nearest rank: the smallest value that at least p in a hundred of them do
// not exceed. It is one of the values, never something in between.
func percentile(rising []float64, p int) float64 {
	if len(rising) == 0 {
		return 0
	}
	rank := (p*len(rising) + 99) / 100
	return rising[min(max(rank, 1), len(rising))-1]
}

// latency is how long the accepted requests took, in milliseconds.
type latency struct {
	P50 float64 `json:"p50"`
	P90 float64 `json:"p90"`
	P99 float64 `json:"p99"`
	Max float64 `json:"max"`
}

// refusal is how long the server took to say no, in milliseconds.
type refusal struct {
	P50 float64 `json:"p50"`
	Max float64 `json:"max"`
}

// results is what the 200s said about the picture.
type results struct {
	Match   int `json:"match"`
	NoMatch int `json:"no_match"`
	// Rejected counts faces turned down as too poor to compare, and
	// pictures without a face.
	Rejected int `json:"rejected"`
}

// run is one closed loop, or an open loop at one rate.
type run struct {
	// Rate is what an open loop was asked for, in requests per second;
	// Concurrency is the number of workers of a closed loop.
	Rate        float64 `json:"rate,omitempty"`
	Concurrency int     `json:"concurrency,omitempty"`
	// MeasuredMS is less than the duration asked for if the run was cut
	// short.
	MeasuredMS float64     `json:"measured_ms"`
	Sent       int         `json:"sent"`
	OK         int         `json:"ok"`
	Refused    int         `json:"refused"`
	Other      map[int]int `json:"other_statuses"`
	Errors     int         `json:"transport_errors"`
	Timeouts   int         `json:"timeouts"`
	Skipped    int         `json:"skipped"`
	// Unreadable counts 200s whose body was not a search answer.
	Unreadable        int     `json:"unreadable,omitempty"`
	AcceptedPerSecond float64 `json:"accepted_per_second"`
	// RefusedShare is the part of the requests sent that got a 429, from 0
	// to 1.
	RefusedShare float64 `json:"refused_share"`
	// Latency and Refusal are absent when nothing was accepted or
	// refused.
	Latency *latency `json:"latency_ms,omitempty"`
	Refusal *refusal `json:"refusal_latency_ms,omitempty"`
	// Server holds the medians of the stages the server reported.
	Server       map[string]float64 `json:"server_ms"`
	Results      results            `json:"results"`
	WithinTarget bool               `json:"within_target"`
}

// summarise turns the tally of a run into its row of the report.
func summarise(t *tally, measured, target time.Duration) run {
	t.mu.Lock()
	defer t.mu.Unlock()
	r := run{
		MeasuredMS: rounded(millis(measured)),
		OK:         len(t.accepted), Refused: len(t.refused), Other: t.other,
		Errors: t.failed, Timeouts: t.timedOut, Skipped: t.skipped, Unreadable: t.unreadable,
		Server: make(map[string]float64), Results: t.results,
	}
	r.Sent = r.OK + r.Refused + r.Errors + r.Timeouts
	for _, n := range t.other {
		r.Sent += n
	}
	if measured > 0 {
		r.AcceptedPerSecond = rounded(float64(r.OK) / measured.Seconds())
	}
	if r.Sent > 0 {
		r.RefusedShare = math.Round(1e6*float64(r.Refused)/float64(r.Sent)) / 1e6
	}
	if took := sorted(t.accepted); len(took) > 0 {
		r.Latency = &latency{
			P50: rounded(percentile(took, 50)), P90: rounded(percentile(took, 90)),
			P99: rounded(percentile(took, 99)), Max: rounded(took[len(took)-1]),
		}
		r.WithinTarget = r.Latency.P99 <= millis(target)
	}
	if took := sorted(t.refused); len(took) > 0 {
		r.Refusal = &refusal{P50: rounded(percentile(took, 50)), Max: rounded(took[len(took)-1])}
	}
	for name, took := range t.stages {
		r.Server[name] = rounded(percentile(sorted(took), 50))
	}
	return r
}

// report is what the command found: the document --json prints.
type report struct {
	URL string `json:"url"`
	// Loop is "closed" or "open".
	Loop       string  `json:"loop"`
	Pictures   int     `json:"pictures"`
	DurationMS float64 `json:"duration_ms"`
	WarmupMS   float64 `json:"warmup_ms"`
	TimeoutMS  float64 `json:"timeout_ms"`
	TargetMS   float64 `json:"target_ms"`
	Runs       []run   `json:"runs"`
	// BestRate is the highest rate of --steps at which the 99th percentile
	// of accepted requests stayed within the target; absent if none did.
	BestRate *float64 `json:"highest_rate_within_target,omitempty"`
	// RefusingFrom is the lowest rate of --steps at which the server
	// refused requests; absent if it refused none.
	RefusingFrom *float64 `json:"lowest_rate_with_refusals,omitempty"`
	// Interrupted says that the command was stopped before it was through:
	// the last run is shorter than asked for, and later ones are missing.
	Interrupted bool   `json:"interrupted"`
	Verdict     string `json:"verdict"`

	duration, target time.Duration
}

func newReport(o *options) *report {
	r := &report{
		URL: o.search.Redacted(), Loop: "closed", Pictures: len(o.pictures),
		DurationMS: millis(o.duration), WarmupMS: millis(o.warmup),
		TimeoutMS: millis(o.timeout), TargetMS: millis(o.target),
		Runs: []run{}, duration: o.duration, target: o.target,
	}
	if len(o.rates) > 0 {
		r.Loop = "open"
	}
	return r
}

// add files the tally of a run, at the given rate if the loop was open.
func (r *report) add(o *options, rate float64, t *tally, measured time.Duration, note func(string, ...any)) {
	row := summarise(t, measured, o.target)
	if row.Rate = rate; rate == 0 {
		row.Concurrency = o.concurrency
	}
	r.Runs = append(r.Runs, row)
	if t.trouble != nil {
		note("%s without an answer, the first because: %v", count(row.Errors, "request"), t.trouble)
	}
}

// conclude writes the judgement: did the accepted requests stay within the
// target, at what rate, and what share was refused to get there.
// With stepped rates it names the highest one that held.
func (r *report) conclude(stepped bool) {
	target := r.target.String()
	if !stepped {
		r.Verdict = "nothing was measured"
		if len(r.Runs) > 0 {
			r.Verdict = r.Runs[0].judgement(target)
		}
		return
	}
	var best, refusing *run
	for i := range r.Runs {
		row := &r.Runs[i]
		if row.WithinTarget && (best == nil || row.Rate > best.Rate) {
			best = row
		}
		if row.Refused > 0 && (refusing == nil || row.Rate < refusing.Rate) {
			refusing = row
		}
	}
	r.Verdict = "at none of these rates did the 99th percentile of accepted requests stay within " + target
	if best != nil {
		r.BestRate = &best.Rate
		r.Verdict = fmt.Sprintf("the highest rate at which the 99th percentile of accepted requests stayed within %s "+
			"was %s requests per second: %s accepted per second, %s refused",
			target, number(best.Rate), tenths(best.AcceptedPerSecond), percent(best.RefusedShare))
	}
	// Where the refusing starts is where the server's capacity ends.
	if refusing == nil {
		r.Verdict += "; the server refused nothing at any of them"
		return
	}
	r.RefusingFrom = &refusing.Rate
	r.Verdict += fmt.Sprintf("; the server began refusing at %s requests per second", number(refusing.Rate))
}

func (r *run) judgement(target string) string {
	switch {
	case r.Sent == 0:
		return "nothing was measured"
	case r.Latency == nil:
		return fmt.Sprintf("no request was accepted: of %d sent, %s were refused", r.Sent, percent(r.RefusedShare))
	}
	verb := "stayed within"
	if !r.WithinTarget {
		verb = "went over"
	}
	return fmt.Sprintf("the 99th percentile of accepted requests, %s ms, %s the target of %s "+
		"at %s accepted requests per second; %s of the requests were refused",
		ms(r.Latency.P99), verb, target, tenths(r.AcceptedPerSecond), percent(r.RefusedShare))
}

// ms writes milliseconds with as many digits as are worth reading.
func ms(v float64) string {
	if v < 10 {
		return strconv.FormatFloat(v, 'f', 2, 64)
	}
	return strconv.FormatFloat(v, 'f', 1, 64)
}

func tenths(v float64) string { return strconv.FormatFloat(v, 'f', 1, 64) }

func percent(share float64) string { return tenths(100*share) + "%" }

func (r *report) print(w io.Writer, asJSON bool) error {
	if asJSON {
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		return enc.Encode(r)
	}
	return r.text(w)
}

// text prints three tables with a row for every run, and the judgement
// under them.
func (r *report) text(w io.Writer) error {
	load := "rate/s"
	if r.Loop == "closed" {
		load = "workers"
	}
	within := "within " + r.target.String()

	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	line := func(cells ...string) { say(tw, "%s\t\n", strings.Join(cells, "\t")) }

	say(tw, "requests\n")
	line(load, "sent", "200", "429", "other", "errors", "timeouts", "skipped", "accepted/s", "refused")
	for i := range r.Runs {
		row := &r.Runs[i]
		other := 0
		for _, n := range row.Other {
			other += n
		}
		line(row.load(), strconv.Itoa(row.Sent), strconv.Itoa(row.OK), strconv.Itoa(row.Refused), strconv.Itoa(other),
			strconv.Itoa(row.Errors), strconv.Itoa(row.Timeouts), strconv.Itoa(row.Skipped),
			tenths(row.AcceptedPerSecond), percent(row.RefusedShare))
	}

	say(tw, "\nlatency in ms, as the client saw it\n")
	line(load, "median", "90th", "99th", "max", "refusal median", "refusal max", within)
	for i := range r.Runs {
		row := &r.Runs[i]
		cells := []string{row.load(), "-", "-", "-", "-", "-", "-", "-"}
		if l := row.Latency; l != nil {
			cells[1], cells[2], cells[3], cells[4] = ms(l.P50), ms(l.P90), ms(l.P99), ms(l.Max)
			cells[7] = "no"
			if row.WithinTarget {
				cells[7] = "yes"
			}
		}
		if l := row.Refusal; l != nil {
			cells[5], cells[6] = ms(l.P50), ms(l.Max)
		}
		line(cells...)
	}

	say(tw, "\nthe server's side: medians in ms, and what the 200s said\n")
	line(append(append([]string{load}, stageNames...), "match", "no match", "turned down")...)
	for i := range r.Runs {
		row := &r.Runs[i]
		cells := make([]string, 0, len(stageNames)+4)
		cells = append(cells, row.load())
		for _, name := range stageNames {
			cell := "-"
			if took, ok := row.Server[name]; ok {
				cell = ms(took)
			}
			cells = append(cells, cell)
		}
		line(append(cells, strconv.Itoa(row.Results.Match), strconv.Itoa(row.Results.NoMatch), strconv.Itoa(row.Results.Rejected))...)
	}
	say(tw, "\n")
	if err := tw.Flush(); err != nil {
		return err
	}

	for i := range r.Runs {
		row := &r.Runs[i]
		at := ""
		if len(r.Runs) > 1 {
			at = " at " + row.load() + "/s"
		}
		if len(row.Other) > 0 {
			say(w, "other statuses%s: %s\n", at, row.others())
		}
		if row.Unreadable > 0 {
			say(w, "answers with status 200 that were not a search answer%s: %d\n", at, row.Unreadable)
		}
	}
	if r.Interrupted && len(r.Runs) > 0 {
		say(w, "interrupted: the last row covers %s of the %s asked for\n",
			seconds(r.Runs[len(r.Runs)-1].MeasuredMS), r.duration)
	}
	_, err := fmt.Fprintln(w, r.Verdict)
	return err
}

// load is the first cell of a row: what the run asked of the server.
func (r *run) load() string {
	if r.Rate > 0 {
		return number(r.Rate)
	}
	return strconv.Itoa(r.Concurrency)
}

// others lists the statuses that were neither 200 nor 429, by code.
func (r *run) others() string {
	codes := make([]int, 0, len(r.Other))
	for code := range r.Other {
		codes = append(codes, code)
	}
	slices.Sort(codes)
	parts := make([]string, len(codes))
	for i, code := range codes {
		parts[i] = fmt.Sprintf("%d x %d", r.Other[code], code)
	}
	return strings.Join(parts, ", ")
}

// seconds writes a number of milliseconds as a duration, to the hundredth
// of a second.
func seconds(ms float64) time.Duration {
	return time.Duration(ms * float64(time.Millisecond)).Round(10 * time.Millisecond)
}
