package load

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestPercentile(t *testing.T) {
	upTo := func(n int) []float64 {
		values := make([]float64, n)
		for i := range values {
			values[i] = float64(i + 1)
		}
		return values
	}
	for _, c := range []struct {
		name   string
		values []float64
		p      int
		want   float64
	}{
		{"nothing", nil, 50, 0},
		{"one value, median", []float64{7}, 50, 7},
		{"one value, 99th", []float64{7}, 99, 7},
		{"two values, median", []float64{1, 2}, 50, 1},
		{"two values, 90th", []float64{1, 2}, 90, 2},
		{"four values, median", []float64{1, 2, 3, 4}, 50, 2},
		{"five values, median", []float64{1, 2, 3, 4, 5}, 50, 3},
		{"ten values, median", upTo(10), 50, 5},
		{"ten values, 90th", upTo(10), 90, 9},
		{"ten values, 99th", upTo(10), 99, 10},
		{"a hundred values, median", upTo(100), 50, 50},
		{"a hundred values, 90th", upTo(100), 90, 90},
		{"a hundred values, 99th", upTo(100), 99, 99},
		{"two hundred values, 99th", upTo(200), 99, 198},
		{"a thousand and one values, 99th", upTo(1001), 99, 991},
		{"the lowest", upTo(10), 0, 1},
		{"the highest", upTo(10), 100, 10},
	} {
		if got := percentile(c.values, c.p); got != c.want {
			t.Errorf("%s: %v, want %v", c.name, got, c.want)
		}
	}

	// sorting leaves the measurements as they came
	took := []float64{30, 10, 20}
	if rising := sorted(took); !slices.Equal(rising, []float64{10, 20, 30}) || !slices.Equal(took, []float64{30, 10, 20}) {
		t.Errorf("sorted %v into %v", took, rising)
	}
}

// answered is a 200 that took the given number of milliseconds and said
// what the JSON says.
func answered(t *testing.T, took float64, body string) exchange {
	t.Helper()
	x := exchange{status: http.StatusOK, took: time.Duration(took * float64(time.Millisecond))}
	if body != "" {
		x.answer = &answer{}
		if err := json.Unmarshal([]byte(body), x.answer); err != nil {
			t.Fatal(err)
		}
	}
	return x
}

func status(code int, took float64) exchange {
	return exchange{status: code, took: time.Duration(took * float64(time.Millisecond))}
}

// tallied is a run with ten accepted requests of 10 to 100 ms, four
// refusals, three other answers, two timeouts and a broken connection.
func tallied(t *testing.T) *tally {
	t.Helper()
	tl := newTally()
	for i := 10; i >= 1; i-- {
		body := fmt.Sprintf(`{"faces":[{"match":true}],"timing_ms":{"queue":%d,"decode":2,"detect":3,"align":1,"embed":4,"search":0.004,"total":%d.5}}`, i-1, 10*i-1)
		switch i {
		case 2:
			body = `{"faces":[{"match":false,"matches":[]}],"timing_ms":{"total":19.5}}`
		case 3:
			body = `{"faces":[{"reject":"blurry","match":false}],"timing_ms":{"total":29.5}}`
		case 4:
			body = `{"faces":[],"reject":"no_face","timing_ms":{"total":39.5}}`
		case 5:
			body = "" // not a search answer
		}
		tl.add(answered(t, float64(10*i), body))
	}
	for _, took := range []float64{0.5, 0.25, 2, 1} {
		tl.add(status(http.StatusTooManyRequests, took))
	}
	tl.add(status(http.StatusInternalServerError, 1))
	tl.add(status(http.StatusServiceUnavailable, 1))
	tl.add(status(http.StatusInternalServerError, 1))
	tl.add(exchange{err: &url.Error{Op: "Post", URL: "http://example.test", Err: context.DeadlineExceeded}})
	tl.add(exchange{err: os.ErrDeadlineExceeded})
	tl.add(exchange{err: io.ErrUnexpectedEOF})
	tl.add(exchange{err: errors.New("connection reset")})
	tl.skip()
	tl.skip()
	return tl
}

func TestSummarise(t *testing.T) {
	tl := tallied(t)
	r := summarise(tl, 2*time.Second, 100*time.Millisecond)
	consistent(t, &r)

	if r.Sent != 21 || r.OK != 10 || r.Refused != 4 || r.Errors != 2 || r.Timeouts != 2 || r.Skipped != 2 || r.Unreadable != 1 {
		t.Errorf("counts: %+v", r)
	}
	if len(r.Other) != 2 || r.Other[500] != 2 || r.Other[503] != 1 {
		t.Errorf("other statuses: %v", r.Other)
	}
	if r.MeasuredMS != 2000 || r.AcceptedPerSecond != 5 || r.RefusedShare != 0.190476 {
		t.Errorf("rates: %+v", r)
	}
	if *r.Latency != (latency{P50: 50, P90: 90, P99: 100, Max: 100}) || *r.Refusal != (refusal{P50: 0.5, Max: 2}) {
		t.Errorf("latencies: %+v and %+v", r.Latency, r.Refusal)
	}
	if !r.WithinTarget {
		t.Error("a 99th percentile on the target is within it")
	}
	if r.Results != (results{Match: 6, NoMatch: 1, Rejected: 2}) {
		t.Errorf("results: %+v", r.Results)
	}
	// six answers have every stage, three only the total, one nothing
	want := map[string]float64{"queue": 6, "decode": 2, "detect": 3, "embed": 4, "search": 0.004, "total": 59.5}
	if len(r.Server) != len(want) {
		t.Errorf("the server's side: %v", r.Server)
	}
	for name, median := range want {
		if r.Server[name] != median {
			t.Errorf("median of %s: %v, want %v", name, r.Server[name], median)
		}
	}
	if tl.trouble == nil || tl.trouble.Error() != io.ErrUnexpectedEOF.Error() {
		t.Errorf("the first failure kept: %v", tl.trouble)
	}

	if r := summarise(tl, 2*time.Second, 99*time.Millisecond); r.WithinTarget {
		t.Error("a 99th percentile of 100 ms is within 99 ms")
	}

	// a run that was stopped before anything was measured
	empty := summarise(newTally(), 0, time.Second)
	consistent(t, &empty)
	if empty.Sent != 0 || empty.AcceptedPerSecond != 0 || empty.RefusedShare != 0 || empty.WithinTarget ||
		empty.Latency != nil || empty.Refusal != nil || empty.Other == nil || empty.Server == nil {
		t.Errorf("an empty run: %+v", empty)
	}
}

// stepped is the report of a run in two steps, the second cut short.
func stepped(t *testing.T) *report {
	t.Helper()
	search, err := url.Parse("http://user:secret@faces.example:8080/v1/search")
	if err != nil {
		t.Fatal(err)
	}
	o := &options{
		search: search, rates: []float64{20, 0.5}, stepped: true, pictures: make([][]byte, 3),
		duration: 2 * time.Second, warmup: time.Second, timeout: 5 * time.Second, target: 100 * time.Millisecond,
	}
	rep := newReport(o)
	quiet := func(string, ...any) {}
	light := newTally()
	for i := 1; i <= 4; i++ {
		light.add(answered(t, float64(i), `{"faces":[{"match":true}],"timing_ms":{"total":0.75}}`))
	}
	rep.add(o, 20, light, 2*time.Second, quiet)
	rep.add(o, 0.5, tallied(t), 1250*time.Millisecond, quiet)
	rep.Interrupted = true
	rep.conclude(o.stepped)
	return rep
}

func TestJSONAgreesWithText(t *testing.T) {
	rep := stepped(t)
	var document, tables bytes.Buffer
	if err := rep.print(&document, true); err != nil {
		t.Fatal(err)
	}
	if err := rep.print(&tables, false); err != nil {
		t.Fatal(err)
	}

	// the names are in snake case, and durations are milliseconds
	var doc struct {
		URL         string
		Loop        string
		Pictures    int
		DurationMS  float64 `json:"duration_ms"`
		WarmupMS    float64 `json:"warmup_ms"`
		TimeoutMS   float64 `json:"timeout_ms"`
		TargetMS    float64 `json:"target_ms"`
		Best        float64 `json:"highest_rate_within_target"`
		Refusing    float64 `json:"lowest_rate_with_refusals"`
		Interrupted bool
		Verdict     string
		Runs        []struct {
			Rate       float64
			MeasuredMS float64                               `json:"measured_ms"`
			Sent       int                                   `json:"sent"`
			OK         int                                   `json:"ok"`
			Refused    int                                   `json:"refused"`
			Other      map[string]int                        `json:"other_statuses"`
			Errors     int                                   `json:"transport_errors"`
			Timeouts   int                                   `json:"timeouts"`
			Skipped    int                                   `json:"skipped"`
			Unreadable int                                   `json:"unreadable"`
			PerSecond  float64                               `json:"accepted_per_second"`
			Share      float64                               `json:"refused_share"`
			Latency    *struct{ P50, P90, P99, Max float64 } `json:"latency_ms"`
			Refusal    *struct{ P50, Max float64 }           `json:"refusal_latency_ms"`
			Server     map[string]float64                    `json:"server_ms"`
			Results    struct {
				Match    int
				NoMatch  int `json:"no_match"`
				Rejected int
			}
			Within bool `json:"within_target"`
		}
	}
	dec := json.NewDecoder(&document)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&doc); err != nil {
		t.Fatalf("the document: %v", err)
	}
	if doc.URL != "http://user:xxxxx@faces.example:8080/v1/search" || doc.Loop != "open" || doc.Pictures != 3 ||
		doc.DurationMS != 2000 || doc.WarmupMS != 1000 || doc.TimeoutMS != 5000 || doc.TargetMS != 100 ||
		doc.Best != 20 || doc.Refusing != 0.5 || !doc.Interrupted || len(doc.Runs) != 2 {
		t.Fatalf("the document: %+v", doc)
	}
	if first := doc.Runs[0]; first.Rate != 20 || first.Refusal != nil || first.Latency == nil || len(first.Other) != 0 || first.Other == nil {
		t.Errorf("the first run: %+v", first)
	}
	if second := doc.Runs[1]; second.Rate != 0.5 || second.MeasuredMS != 1250 || second.Other["500"] != 2 || second.Unreadable != 1 {
		t.Errorf("the second run: %+v", second)
	}

	// every number of the document is in its place in the tables
	text := tables.String()
	var rows [][]string
	for _, line := range strings.Split(text, "\n") {
		if fields := strings.Fields(line); len(fields) > 0 && (fields[0] == "20" || fields[0] == "0.5") {
			rows = append(rows, fields)
		}
	}
	if len(rows) != 6 {
		t.Fatalf("three tables of two rows each:\n%s", text)
	}
	dash := func(have bool, v float64) string {
		if !have {
			return "-"
		}
		return ms(v)
	}
	yes := map[bool]string{true: "yes", false: "no"}
	for i, r := range doc.Runs {
		other := 0
		for _, n := range r.Other {
			other += n
		}
		l, refused := r.Latency, r.Refusal
		if refused == nil {
			refused = &struct{ P50, Max float64 }{}
		}
		stage := func(name string) string {
			took, ok := r.Server[name]
			return dash(ok, took)
		}
		for table, want := range [][]string{
			{
				number(r.Rate), strconv.Itoa(r.Sent), strconv.Itoa(r.OK), strconv.Itoa(r.Refused), strconv.Itoa(other),
				strconv.Itoa(r.Errors), strconv.Itoa(r.Timeouts), strconv.Itoa(r.Skipped), tenths(r.PerSecond), percent(r.Share),
			},
			{
				number(r.Rate), ms(l.P50), ms(l.P90), ms(l.P99), ms(l.Max),
				dash(r.Refusal != nil, refused.P50), dash(r.Refusal != nil, refused.Max), yes[r.Within],
			},
			{
				number(r.Rate), stage("queue"), stage("decode"), stage("detect"), stage("embed"), stage("search"), stage("total"),
				strconv.Itoa(r.Results.Match), strconv.Itoa(r.Results.NoMatch), strconv.Itoa(r.Results.Rejected),
			},
		} {
			if got := rows[2*table+i]; !slices.Equal(got, want) {
				t.Errorf("run %d, table %d: the row reads %v, the document says %v", i+1, table+1, got, want)
			}
		}
	}
	for _, want := range []string{
		"\n  rate/s  sent  200  429  other  errors  timeouts  skipped  accepted/s  refused\n",
		"\n      20     4    4    0      0       0         0        0         2.0     0.0%\n",
		"\n     0.5    21   10    4      3       2         2        2         8.0    19.0%\n",
		"  refusal median  refusal max  within 100ms\n",
		"  queue  decode  detect  embed  search  total  match  no match  turned down\n",
		"\nother statuses at 0.5/s: 2 x 500, 1 x 503\n",
		"\nanswers with status 200 that were not a search answer at 0.5/s: 1\n",
		"\ninterrupted: the last row covers 1.25s of the 2s asked for\n",
		"\n" + doc.Verdict + "\n",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("the tables lack %q:\n%s", want, text)
		}
	}
	if strings.Contains(text+document.String(), "secret") {
		t.Error("the password in the address is in the report")
	}
}

func TestVerdict(t *testing.T) {
	fast, slow := &latency{P50: 20, P90: 30, P99: 41.25, Max: 60}, &latency{P50: 200, P90: 300, P99: 412.5, Max: 600}
	held := run{Rate: 40, Sent: 400, OK: 380, Refused: 20, AcceptedPerSecond: 38, RefusedShare: 0.05, Latency: fast, WithinTarget: true}
	late := run{Rate: 80, Sent: 800, OK: 800, AcceptedPerSecond: 80, Latency: slow}
	shut := run{Rate: 160, Sent: 1600, Refused: 1200, RefusedShare: 0.75}
	low := run{Rate: 5, Sent: 50, OK: 50, AcceptedPerSecond: 5, Latency: fast, WithinTarget: true}

	const highest = "the highest rate at which the 99th percentile of accepted requests stayed within 100ms was "
	const none = "at none of these rates did the 99th percentile of accepted requests stay within 100ms"
	for name, c := range map[string]struct {
		stepped        bool
		runs           []run
		want           string
		best, refusing float64
	}{
		"within the target": {false, []run{held},
			"the 99th percentile of accepted requests, 41.2 ms, stayed within the target of 100ms at 38.0 accepted requests per second; 5.0% of the requests were refused", 0, 0},
		"over the target": {false, []run{late},
			"the 99th percentile of accepted requests, 412.5 ms, went over the target of 100ms at 80.0 accepted requests per second; 0.0% of the requests were refused", 0, 0},
		"nothing accepted": {false, []run{shut}, "no request was accepted: of 1600 sent, 75.0% were refused", 0, 0},
		"nothing sent":     {false, []run{{Concurrency: 4}}, "nothing was measured", 0, 0},
		"no run":           {false, nil, "nothing was measured", 0, 0},
		"the highest step that held": {true, []run{low, held, late, shut},
			highest + "40 requests per second: 38.0 accepted per second, 5.0% refused; the server began refusing at 40 requests per second", 40, 40},
		"steps in falling order": {true, []run{shut, late, held, low},
			highest + "40 requests per second: 38.0 accepted per second, 5.0% refused; the server began refusing at 40 requests per second", 40, 40},
		"steps without a refusal": {true, []run{low, late},
			highest + "5 requests per second: 5.0 accepted per second, 0.0% refused; the server refused nothing at any of them", 5, 0},
		"no step held": {true, []run{late, shut}, none + "; the server began refusing at 160 requests per second", 0, 160},
		"no step run":  {true, nil, none + "; the server refused nothing at any of them", 0, 0},
	} {
		rep := &report{Runs: c.runs, target: 100 * time.Millisecond}
		rep.conclude(c.stepped)
		if rep.Verdict != c.want {
			t.Errorf("%s:\n got %s\nwant %s", name, rep.Verdict, c.want)
		}
		for what, pair := range map[string]struct {
			got  *float64
			want float64
		}{"highest rate within the target": {rep.BestRate, c.best}, "lowest rate with refusals": {rep.RefusingFrom, c.refusing}} {
			if (pair.want == 0) != (pair.got == nil) || (pair.got != nil && *pair.got != pair.want) {
				t.Errorf("%s: the %s is %v, want %v", name, what, pair.got, pair.want)
			}
		}
	}
}

// failing is somewhere a report cannot be written to.
type failing struct{}

func (failing) Write([]byte) (int, error) { return 0, errors.New("the pipe is closed") }

func TestReportThatCannotBeWritten(t *testing.T) {
	rep := stepped(t)
	for _, asJSON := range []bool{false, true} {
		if err := rep.print(failing{}, asJSON); err == nil || !strings.Contains(err.Error(), "the pipe is closed") {
			t.Errorf("JSON %v: %v", asJSON, err)
		}
	}
}

func TestFormats(t *testing.T) {
	for v, want := range map[float64]string{0: "0.00", 0.004: "0.00", 0.316: "0.32", 9.994: "9.99", 10: "10.0", 41.25: "41.2", 1234.56: "1234.6"} {
		if got := ms(v); got != want {
			t.Errorf("ms(%v) = %s, want %s", v, got, want)
		}
	}
	for share, want := range map[float64]string{0: "0.0%", 0.0213: "2.1%", 0.5: "50.0%", 1: "100.0%"} {
		if got := percent(share); got != want {
			t.Errorf("percent(%v) = %s, want %s", share, got, want)
		}
	}
	for n, want := range map[int]string{0: "0 workers", 1: "1 worker", 4: "4 workers"} {
		if got := count(n, "worker"); got != want {
			t.Errorf("count(%d) = %s, want %s", n, got, want)
		}
	}
	for v, want := range map[float64]string{5: "5", 0.5: "0.5", 12.25: "12.25", 100000: "100000"} {
		if got := number(v); got != want {
			t.Errorf("number(%v) = %s, want %s", v, got, want)
		}
	}
	if got := rounded(1.23456789); got != 1.235 {
		t.Errorf("rounded to %v", got)
	}
}
