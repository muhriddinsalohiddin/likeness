package load

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/muhriddinsalohiddin/likeness/index"
	"github.com/muhriddinsalohiddin/likeness/internal/cli"
	"github.com/muhriddinsalohiddin/likeness/internal/enginetest"
	"github.com/muhriddinsalohiddin/likeness/internal/gallery"
	"github.com/muhriddinsalohiddin/likeness/internal/server"
	"github.com/muhriddinsalohiddin/likeness/store"
)

// The tests drive the real server over a stand-in engine whose "pictures"
// are lines of text (see enginetest): "slow:anna" takes Fake.Delay, the
// others no time at all.
const (
	adminKey   = "an-admin-key-for-tests"
	searchKey  = "a-search-key-for-tests"
	metricsKey = "a-metrics-key-for-tests"
)

var background = context.Background()

// target is a likeness server to send load to.
type target struct {
	url  string
	fake *enginetest.Fake
	// dialled counts the connections made to it.
	dialled atomic.Int64
}

// newTarget serves an empty gallery on the loopback interface.
func newTarget(t *testing.T, tune ...func(*server.Config)) *target {
	t.Helper()
	fake := &enginetest.Fake{}
	info := fake.Info()
	st, err := store.Open(filepath.Join(t.TempDir(), "faces.db"), store.Options{Dim: info.Dim, Fingerprint: info.Fingerprint})
	if err != nil {
		t.Fatal(err)
	}
	faces, err := gallery.Open(background, st, gallery.Options{Index: index.Options{OnHeap: true}})
	if err != nil {
		st.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() { faces.Close() })
	keys, err := server.NewKeys([]string{adminKey}, []string{searchKey}, []string{metricsKey})
	if err != nil {
		t.Fatal(err)
	}
	cfg := server.Config{Engine: fake, Gallery: faces, Keys: keys}
	for _, f := range tune {
		f(&cfg)
	}
	srv, err := server.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	tg := &target{fake: fake}
	ts := httptest.NewUnstartedServer(srv.Handler())
	ts.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateNew {
			tg.dialled.Add(1)
		}
	}
	ts.Start()
	t.Cleanup(ts.Close)
	tg.url = ts.URL
	return tg
}

// enrol adds a face to a subject through the API.
func (tg *target) enrol(t *testing.T, subject, picture string) {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, tg.url+"/v1/subjects/"+subject+"/faces", strings.NewReader(picture))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+adminKey)
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if body, _ := io.ReadAll(res.Body); res.StatusCode != http.StatusCreated {
		t.Fatalf("enrolling %s: %d %s", subject, res.StatusCode, body)
	}
}

// pictures writes each line into a file of its own and returns the names.
func pictures(t *testing.T, lines ...string) []string {
	t.Helper()
	dir := t.TempDir()
	names := make([]string, len(lines))
	for i, line := range lines {
		names[i] = filepath.Join(dir, fmt.Sprintf("picture-%d.txt", i))
		if err := os.WriteFile(names[i], []byte(line), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return names
}

// vars makes a getenv out of pairs of names and values.
func vars(pairs ...string) func(string) string {
	vars := make(map[string]string)
	for i := 0; i < len(pairs); i += 2 {
		vars[pairs[i]] = pairs[i+1]
	}
	return func(name string) string { return vars[name] }
}

var withSearchKey = vars(keysVar, searchKey)

// load runs the command and returns what it printed and what it noted.
func load(ctx context.Context, getenv func(string) string, args ...string) (stdout, stderr string, err error) {
	var out, notes bytes.Buffer
	err = Run(ctx, args, cli.Env{Getenv: getenv, Stdout: &out, Stderr: &notes})
	return out.String(), notes.String(), err
}

// measure runs the command with --json and insists on a report.
func measure(t *testing.T, getenv func(string) string, args ...string) *report {
	t.Helper()
	out, notes, err := load(background, getenv, append([]string{"--json"}, args...)...)
	if err != nil {
		t.Fatalf("load: %v\n%s", err, notes)
	}
	return decode(t, out)
}

func decode(t *testing.T, document string) *report {
	t.Helper()
	var rep report
	dec := json.NewDecoder(strings.NewReader(document))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&rep); err != nil {
		t.Fatalf("the report is not the JSON document expected: %v\n%s", err, document)
	}
	return &rep
}

// consistent checks what holds for every run, however it went.
func consistent(t *testing.T, r *run) {
	t.Helper()
	other := 0
	for _, n := range r.Other {
		other += n
	}
	if r.Sent != r.OK+r.Refused+other+r.Errors+r.Timeouts {
		t.Errorf("%d sent, but %d answered, %d refused, %d other, %d errors and %d timeouts",
			r.Sent, r.OK, r.Refused, other, r.Errors, r.Timeouts)
	}
	if got := r.Results.Match + r.Results.NoMatch + r.Results.Rejected + r.Unreadable; got != r.OK {
		t.Errorf("%d answers with status 200, %d of them accounted for: %+v", r.OK, got, r.Results)
	}
	if (r.Latency != nil) != (r.OK > 0) || (r.Refusal != nil) != (r.Refused > 0) {
		t.Errorf("latencies do not go with the counts: %+v", r)
	}
	if l := r.Latency; l != nil && (l.P50 <= 0 || l.P50 > l.P90 || l.P90 > l.P99 || l.P99 > l.Max) {
		t.Errorf("percentiles out of order: %+v", l)
	}
	if l := r.Refusal; l != nil && (l.P50 <= 0 || l.P50 > l.Max) {
		t.Errorf("refusal latencies out of order: %+v", l)
	}
}

func TestClosedLoop(t *testing.T) {
	tg := newTarget(t, func(c *server.Config) { c.MaxInflight = 4 })
	tg.fake.Delay = 2 * time.Millisecond
	tg.enrol(t, "anna", "face:anna")
	files := pictures(t, "slow:anna", "face:anna#2")

	args := append([]string{"--url", tg.url, "--concurrency", "3", "--duration", "300ms", "--warmup", "50ms"}, files...)
	rep := measure(t, withSearchKey, args...)
	if rep.Loop != "closed" || rep.Pictures != 2 || rep.DurationMS != 300 || rep.WarmupMS != 50 ||
		rep.TimeoutMS != 5000 || rep.TargetMS != 100 || rep.Interrupted || rep.BestRate != nil || rep.RefusingFrom != nil {
		t.Errorf("the report describes another run: %+v", rep)
	}
	if rep.URL != tg.url+"/v1/search" {
		t.Errorf("sent to %s", rep.URL)
	}
	if len(rep.Runs) != 1 {
		t.Fatalf("%d runs, want 1", len(rep.Runs))
	}
	r := &rep.Runs[0]
	consistent(t, r)
	// three workers on four slots: nothing is refused, nothing goes wrong
	if r.Concurrency != 3 || r.Rate != 0 || r.Sent < 3 || r.OK != r.Sent || r.Skipped != 0 || r.MeasuredMS != 300 {
		t.Fatalf("the run: %+v", r)
	}
	if r.Results.Match != r.OK {
		t.Errorf("%d of %d answers named anna", r.Results.Match, r.OK)
	}
	if want := float64(r.OK) / 0.3; r.AcceptedPerSecond < want-0.01 || r.AcceptedPerSecond > want+0.01 {
		t.Errorf("%v accepted per second, want %v", r.AcceptedPerSecond, want)
	}
	if r.Latency.Max >= rep.TimeoutMS {
		t.Errorf("a request that took %v ms was counted as answered in time", r.Latency.Max)
	}
	// the stand-in engine always reports the same stage timings
	if s := r.Server; s["decode"] != 2 || s["detect"] != 3 || s["embed"] != 4 || s["total"] <= 0 || s["total"] > r.Latency.P50 {
		t.Errorf("the server's side: %v next to a median of %v ms", s, r.Latency.P50)
	}
	for _, name := range stageNames {
		if _, ok := r.Server[name]; !ok {
			t.Errorf("no median for the %s stage", name)
		}
	}
	if !strings.HasPrefix(rep.Verdict, "the 99th percentile of accepted requests, ") || !strings.Contains(rep.Verdict, "0.0% of the requests were refused") {
		t.Errorf("verdict: %s", rep.Verdict)
	}
	// the enrolment, the first request, the warm-up and what was counted
	if calls := tg.fake.Calls(); calls < r.Sent+2 {
		t.Errorf("the engine saw %d pictures, the report counts %d", calls, r.Sent)
	}
	if tg.fake.Peak() > 3 {
		t.Errorf("%d pictures at once from 3 workers", tg.fake.Peak())
	}
	// one connection for the enrolment, and one for each worker, kept
	if tg.dialled.Load() > 4 {
		t.Errorf("%d connections were made for %d requests", tg.dialled.Load(), tg.fake.Calls())
	}

	// the same in text
	out, notes, err := load(background, withSearchKey, args...)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"requests\n", "workers", "accepted/s", "latency in ms, as the client saw it\n", "refusal median", "within 100ms",
		"the server's side", "turned down", "\nthe 99th percentile of accepted requests, ",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("the tables lack %q:\n%s", want, out)
		}
	}
	for _, want := range []string{"likeness load: searching at " + tg.url + "/v1/search with 2 pictures", "3 workers for 300ms, after 50ms of warm-up"} {
		if !strings.Contains(notes, want) {
			t.Errorf("the notes lack %q:\n%s", want, notes)
		}
	}
}

func TestOverload(t *testing.T) {
	const service = 20 * time.Millisecond
	tg := newTarget(t, func(c *server.Config) { c.MaxInflight, c.Queue = 1, 1 })
	tg.fake.Delay = service
	files := pictures(t, "slow:anna")

	// four times what one slot can carry
	rep := measure(t, withSearchKey, append([]string{"--url", tg.url, "--rate", "200", "--duration", "500ms", "--warmup", "100ms"}, files...)...)
	if rep.Loop != "open" || len(rep.Runs) != 1 {
		t.Fatalf("the report: %+v", rep)
	}
	r := &rep.Runs[0]
	consistent(t, r)
	if r.Rate != 200 || r.Concurrency != 0 || r.Sent != 100 || r.Skipped != 0 {
		t.Fatalf("200 requests per second for half a second: %+v", r)
	}
	if r.OK == 0 || r.Refused == 0 || r.Errors != 0 || r.Timeouts != 0 || len(r.Other) != 0 {
		t.Fatalf("the server should answer some and refuse the rest: %+v", r)
	}
	// one picture at a time, each taking the service time: what was
	// accepted cannot be much more than fits into the half second
	if most := 2 * int(500*time.Millisecond/service); r.OK > most {
		t.Errorf("%d requests accepted, more than the %d one slot can carry", r.OK, most)
	}
	if r.RefusedShare <= 0 || r.RefusedShare >= 1 {
		t.Errorf("refused share %v", r.RefusedShare)
	}
	// accepted requests take the service time, and not much longer for
	// all the others at the door; a refusal takes less than either
	if r.Latency.P50 < millis(service) || r.Latency.P50 > 15*millis(service) {
		t.Errorf("accepted requests took a median of %v ms for a service time of %v", r.Latency.P50, service)
	}
	if r.Refusal.P50 >= r.Latency.P50 {
		t.Errorf("refusing took a median of %v ms, serving %v ms", r.Refusal.P50, r.Latency.P50)
	}
	if tg.fake.Peak() != 1 {
		t.Errorf("%d pictures were analysed at once", tg.fake.Peak())
	}
	if !strings.Contains(rep.Verdict, percent(r.RefusedShare)+" of the requests were refused") {
		t.Errorf("verdict: %s", rep.Verdict)
	}
}

func TestSteps(t *testing.T) {
	tg := newTarget(t, func(c *server.Config) { c.MaxInflight, c.Queue = 1, 1 })
	tg.fake.Delay = 10 * time.Millisecond
	files := pictures(t, "slow:anna")
	args := func(target string) []string {
		return append([]string{
			"--url", tg.url, "--steps", "20, 400", "--duration", "250ms", "--warmup", "50ms", "--timeout", "2s", "--target", target,
		}, files...)
	}

	// a target no answered request can miss: both rates hold
	rep := measure(t, withSearchKey, args("3s")...)
	if rep.Loop != "open" || len(rep.Runs) != 2 {
		t.Fatalf("the report: %+v", rep)
	}
	slow, fast := &rep.Runs[0], &rep.Runs[1]
	consistent(t, slow)
	consistent(t, fast)
	if slow.Rate != 20 || slow.Sent != 5 || fast.Rate != 400 || fast.Sent != 100 {
		t.Errorf("the steps: %+v and %+v", slow, fast)
	}
	if slow.OK == 0 || fast.OK == 0 || fast.Refused == 0 || !slow.WithinTarget || !fast.WithinTarget {
		t.Errorf("the steps: %+v and %+v", slow, fast)
	}
	if rep.BestRate == nil || *rep.BestRate != 400 || rep.RefusingFrom == nil ||
		!strings.HasPrefix(rep.Verdict, "the highest rate at which the 99th percentile of accepted requests stayed within 3s was 400 requests per second") ||
		!strings.HasSuffix(rep.Verdict, fmt.Sprintf("the server began refusing at %s requests per second", number(*rep.RefusingFrom))) {
		t.Errorf("verdict: %s (%v, %v)", rep.Verdict, rep.BestRate, rep.RefusingFrom)
	}

	// and one no request can meet: a table with a row for each rate
	out, notes, err := load(background, withSearchKey, args("1ns")...)
	if err != nil {
		t.Fatal(err)
	}
	rows := map[string]int{}
	for _, line := range strings.Split(out, "\n") {
		if fields := strings.Fields(line); len(fields) > 1 {
			rows[fields[0]]++
		}
	}
	if rows["rate/s"] != 3 || rows["20"] != 3 || rows["400"] != 3 {
		t.Errorf("three tables with a row for each rate:\n%s", out)
	}
	if !strings.Contains(out, "\nat none of these rates did the 99th percentile of accepted requests stay within 1ns; the server began refusing at ") {
		t.Errorf("verdict:\n%s", out)
	}
	for _, want := range []string{"20 requests per second for 250ms, after 50ms of warm-up", "400 requests per second for 250ms"} {
		if !strings.Contains(notes, want) {
			t.Errorf("the notes lack %q:\n%s", want, notes)
		}
	}
}

func TestSkippedSlots(t *testing.T) {
	// a server that answers one request in 50 ms and lets the rest queue
	tg := newTarget(t, func(c *server.Config) { c.MaxInflight, c.Queue, c.QueueTimeout = 1, 64, time.Minute })
	tg.fake.Delay = 50 * time.Millisecond
	files := pictures(t, "slow:anna")

	rep := measure(t, withSearchKey, append([]string{
		"--url", tg.url, "--rate", "200", "--max-pending", "2", "--duration", "300ms", "--warmup", "0",
	}, files...)...)
	r := &rep.Runs[0]
	consistent(t, r)
	if r.Sent == 0 || r.Skipped == 0 || r.Sent+r.Skipped != 60 {
		t.Errorf("60 requests were due: %d sent and %d skipped", r.Sent, r.Skipped)
	}
	if r.OK != r.Sent {
		t.Errorf("%d of %d answered", r.OK, r.Sent)
	}
	if tg.fake.Calls() > r.Sent+1 {
		t.Errorf("the engine saw %d pictures, the report counts %d sent", tg.fake.Calls(), r.Sent)
	}
}

func TestTimeouts(t *testing.T) {
	tg := newTarget(t)
	tg.fake.Delay = time.Minute
	files := pictures(t, "slow:anna")

	rep := measure(t, withSearchKey, append([]string{
		"--url", tg.url, "--concurrency", "2", "--duration", "150ms", "--warmup", "0", "--timeout", "30ms",
	}, files...)...)
	r := &rep.Runs[0]
	consistent(t, r)
	if r.Sent == 0 || r.Timeouts != r.Sent || r.Errors != 0 || r.AcceptedPerSecond != 0 || r.WithinTarget {
		t.Errorf("every request should run out of time: %+v", r)
	}
	if want := fmt.Sprintf("no request was accepted: of %d sent, 0.0%% were refused", r.Sent); rep.Verdict != want {
		t.Errorf("verdict %q, want %q", rep.Verdict, want)
	}
}

func TestResultsAndOtherStatuses(t *testing.T) {
	tg := newTarget(t)
	tg.enrol(t, "anna", "face:anna")
	files := pictures(t, "face:anna", "face:boris", "reject:blurry", "reject:no_face", "error:corrupt")

	rep := measure(t, withSearchKey, append([]string{"--url", tg.url, "--concurrency", "1", "--duration", "300ms", "--warmup", "0"}, files...)...)
	r := &rep.Runs[0]
	consistent(t, r)
	if r.Results.Match == 0 || r.Results.NoMatch == 0 || r.Results.Rejected == 0 || r.Unreadable != 0 {
		t.Errorf("a match, a stranger and two poor pictures in rotation: %+v", r.Results)
	}
	if r.Results.Rejected < r.Results.Match || r.Results.Rejected > 2*r.Results.Match+2 {
		t.Errorf("two of five pictures are turned down, one matches: %+v", r.Results)
	}
	if len(r.Other) != 1 || r.Other[http.StatusUnprocessableEntity] == 0 {
		t.Errorf("the damaged picture is a 422: %v", r.Other)
	}
}

func TestFirstRequestStopsTheRun(t *testing.T) {
	tg := newTarget(t)
	files := pictures(t, "face:anna")
	const wrongKey = "not-a-key-of-this-server"

	for name, c := range map[string]struct {
		getenv func(string) string
		url    string
		says   string
	}{
		"a wrong key":   {vars(keysVar, wrongKey), tg.url, "does not know the key: the first key in LIKENESS_SEARCH_KEYS"},
		"no key":        {vars(), tg.url, "wants an API key: set LIKENESS_SEARCH_KEYS"},
		"a metrics key": {vars(keysVar, metricsKey), tg.url, "may not search"},
		"a wrong path":  {withSearchKey, tg.url + "/nowhere", "/nowhere/v1/search answered 404"},
	} {
		start := time.Now()
		out, notes, err := load(background, c.getenv, append([]string{"--url", c.url, "--duration", "1m"}, files...)...)
		if err == nil || !strings.Contains(err.Error(), c.says) {
			t.Errorf("%s: %v, want an error that says %q", name, err, c.says)
		}
		if out != "" {
			t.Errorf("%s: a report was printed:\n%s", name, out)
		}
		if time.Since(start) > 30*time.Second {
			t.Errorf("%s: the run went on", name)
		}
		for _, key := range []string{wrongKey, metricsKey, searchKey} {
			if strings.Contains(fmt.Sprint(err, out, notes), key) {
				t.Errorf("%s: the key is in the output: %v\n%s", name, err, notes)
			}
		}
	}
	if tg.fake.Calls() != 0 {
		t.Errorf("%d pictures reached the engine", tg.fake.Calls())
	}
}

func TestUnreachableServer(t *testing.T) {
	// a port nothing listens on: one that was just given back
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	address := ln.Addr().String()
	ln.Close()

	out, _, err := load(background, withSearchKey, append([]string{"--url", "http://" + address, "--duration", "1m"}, pictures(t, "face:anna")...)...)
	if err == nil || !strings.Contains(err.Error(), "cannot connect to "+address) || !strings.Contains(err.Error(), "is the server running") {
		t.Errorf("nothing listens there: %v", err)
	}
	if out != "" {
		t.Errorf("a report was printed:\n%s", out)
	}
}

func TestInterrupted(t *testing.T) {
	for name, loop := range map[string][]string{
		"closed loop": {"--concurrency", "2"},
		"open loop":   {"--rate", "100"},
		"steps":       {"--steps", "100,200"},
	} {
		t.Run(name, func(t *testing.T) {
			tg := newTarget(t)
			tg.fake.Delay = 5 * time.Millisecond
			ctx, cancel := context.WithCancel(background)
			defer cancel()
			go func() {
				// the first request and five of the run
				for deadline := time.Now().Add(20 * time.Second); tg.fake.Calls() < 6 && time.Now().Before(deadline); {
					time.Sleep(time.Millisecond)
				}
				cancel()
			}()

			args := append([]string{"--url", tg.url, "--duration", "1m", "--warmup", "0", "--json"}, loop...)
			start := time.Now()
			out, notes, err := load(ctx, withSearchKey, append(args, pictures(t, "slow:anna")...)...)
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("an interrupted run returns %v\n%s", err, notes)
			}
			if time.Since(start) > 30*time.Second {
				t.Errorf("stopping took %v", time.Since(start))
			}
			rep := decode(t, out)
			if !rep.Interrupted || len(rep.Runs) != 1 {
				t.Fatalf("the report: %+v", rep)
			}
			r := &rep.Runs[0]
			consistent(t, r)
			// what was in flight was given the time to finish
			if r.OK < 5 || r.MeasuredMS <= 0 || r.MeasuredMS >= rep.DurationMS {
				t.Errorf("what was measured until then: %+v", r)
			}
			if !strings.Contains(notes, "interrupted") {
				t.Errorf("the notes do not mention it:\n%s", notes)
			}
		})
	}

	// stopped before it began: nothing is sent, nothing printed
	tg := newTarget(t)
	ctx, cancel := context.WithCancel(background)
	cancel()
	out, _, err := load(ctx, withSearchKey, append([]string{"--url", tg.url}, pictures(t, "face:anna")...)...)
	if !errors.Is(err, context.Canceled) || out != "" || tg.fake.Calls() != 0 {
		t.Errorf("stopped before the start: %v, %d requests, printed %q", err, tg.fake.Calls(), out)
	}
}

func TestUsage(t *testing.T) {
	file := pictures(t, "face:anna")[0]
	const server = "http://127.0.0.1:8080"
	for name, c := range map[string]struct {
		args []string
		says string
	}{
		"no server":            {[]string{file}, "--url is required"},
		"not an address":       {[]string{"--url", "localhost:8080", file}, "--url has to be the address of a server"},
		"another scheme":       {[]string{"--url", "ftp://localhost", file}, "--url has to be the address of a server"},
		"both kinds of loop":   {[]string{"--url", server, "--rate", "5", "--concurrency", "2", file}, "exclude each other"},
		"a rate and steps":     {[]string{"--url", server, "--rate", "5", "--steps", "5,10", file}, "exclude each other"},
		"no workers":           {[]string{"--url", server, "--concurrency", "0", file}, "--concurrency is at least 1"},
		"no rate":              {[]string{"--url", server, "--rate", "0", file}, "--rate is a number of requests per second"},
		"a negative rate":      {[]string{"--url", server, "--rate", "-3", file}, "--rate is a number of requests per second"},
		"a rate that is none":  {[]string{"--url", server, "--rate", "NaN", file}, "--rate is a number of requests per second"},
		"a rate beyond reason": {[]string{"--url", server, "--rate", "1e9", file}, "at most 100000"},
		"a rate in words":      {[]string{"--url", server, "--rate", "fast", file}, "invalid value"},
		"a step in words":      {[]string{"--url", server, "--steps", "5,ten,20", file}, `"ten" is not one`},
		"a step left out":      {[]string{"--url", server, "--steps", "5,,20", file}, `"" is not one`},
		"a step of nothing":    {[]string{"--url", server, "--steps", "0", file}, `"0" is not one`},
		"nothing pending":      {[]string{"--url", server, "--rate", "5", "--max-pending", "0", file}, "--max-pending is at least 1"},
		"no duration":          {[]string{"--url", server, "--duration", "0", file}, "have to be above 0"},
		"no timeout":           {[]string{"--url", server, "--timeout", "-1s", file}, "have to be above 0"},
		"no target":            {[]string{"--url", server, "--target", "0s", file}, "have to be above 0"},
		"a negative warm-up":   {[]string{"--url", server, "--warmup", "-1s", file}, "--warmup cannot be negative"},
		"a duration in words":  {[]string{"--url", server, "--duration", "soon", file}, "invalid value"},
		"an unknown flag":      {[]string{"--url", server, "--fast", file}, "flag provided but not defined"},
		"no picture":           {[]string{"--url", server}, "name at least one picture"},
		"a flag at the end":    {[]string{"--url", server, file, "--json"}, "--json comes after a picture; flags go before the pictures"},
		"the server last":      {[]string{file, "--url", server}, "--url comes after a picture; flags go before the pictures"},
	} {
		out, notes, err := load(background, withSearchKey, c.args...)
		if !errors.Is(err, cli.ErrUsage) {
			t.Errorf("%s: %v", name, err)
		}
		if !strings.Contains(notes, c.says) || !strings.Contains(notes, "likeness load") {
			t.Errorf("%s: explained with %q, want %q", name, notes, c.says)
		}
		if out != "" {
			t.Errorf("%s: printed %q", name, out)
		}
	}

	out, notes, err := load(background, withSearchKey, "-h")
	if !errors.Is(err, flag.ErrHelp) || out != "" {
		t.Errorf("-h: %v, printed %q", err, out)
	}
	for _, want := range []string{
		"usage: likeness load --url URL [flags] PICTURE...", "-concurrency workers", "-rate requests", "-steps rates",
		"-max-pending requests", "-duration", "-warmup", "-timeout", "-target", "-json", "(default 10s)",
		"LIKENESS_SEARCH_KEYS", "_FILE",
	} {
		if !strings.Contains(notes, want) {
			t.Errorf("the help lacks %q", want)
		}
	}
}

func TestPictureFiles(t *testing.T) {
	tg := newTarget(t)
	dir := t.TempDir()
	good := pictures(t, "face:anna")[0]

	_, _, err := load(background, withSearchKey, "--url", tg.url, good, filepath.Join(dir, "missing.jpg"))
	if !errors.Is(err, os.ErrNotExist) || !strings.Contains(err.Error(), "missing.jpg") {
		t.Errorf("a picture that is not there: %v", err)
	}
	empty := filepath.Join(dir, "empty.jpg")
	if err := os.WriteFile(empty, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := load(background, withSearchKey, "--url", tg.url, empty); err == nil || !strings.Contains(err.Error(), "empty.jpg is empty") {
		t.Errorf("an empty picture: %v", err)
	}
	if _, _, err := load(background, withSearchKey, "--url", tg.url, dir); err == nil {
		t.Error("a directory was taken for a picture")
	}
	if tg.fake.Calls() != 0 {
		t.Errorf("%d requests were sent all the same", tg.fake.Calls())
	}
}

func TestKey(t *testing.T) {
	tg := newTarget(t)
	args := append([]string{"--url", tg.url, "--concurrency", "1", "--duration", "50ms", "--warmup", "0"}, pictures(t, "face:anna")...)
	dir := t.TempDir()
	file := filepath.Join(dir, "search-keys")
	if err := os.WriteFile(file, []byte(searchKey+"\nanother-key-0123456789\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	// the first of the list is the one presented, wherever the list is
	for name, getenv := range map[string]func(string) string{
		"one key":        withSearchKey,
		"a list":         vars(keysVar, " "+searchKey+" , "+metricsKey),
		"an admin key":   vars(keysVar, adminKey),
		"from a file":    vars(keysVar+"_FILE", file),
		"an empty entry": vars(keysVar, ","+searchKey),
	} {
		for _, format := range []string{"--json", "--json=false"} {
			out, notes, err := load(background, getenv, append([]string{format}, args...)...)
			if err != nil {
				t.Errorf("%s: %v", name, err)
			}
			if !strings.Contains(out, "requests") && !strings.Contains(out, `"runs"`) {
				t.Errorf("%s: no report:\n%s", name, out)
			}
			for _, key := range []string{searchKey, adminKey, metricsKey, "another-key-0123456789"} {
				if strings.Contains(out+notes, key) {
					t.Errorf("%s: a key is in the output:\n%s\n%s", name, out, notes)
				}
			}
		}
	}

	// a list that starts with the wrong key does not fall back on the next
	if _, _, err := load(background, vars(keysVar, metricsKey+","+searchKey), args...); err == nil || !strings.Contains(err.Error(), "may not search") {
		t.Errorf("a metrics key first: %v", err)
	}
	// mistakes in the environment, told without repeating the key
	if _, _, err := load(background, vars(keysVar, searchKey, keysVar+"_FILE", file), args...); err == nil || !strings.Contains(err.Error(), "both set") {
		t.Errorf("a key given twice: %v", err)
	}
	if _, _, err := load(background, vars(keysVar+"_FILE", filepath.Join(dir, "missing")), args...); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("a missing key file: %v", err)
	}
	_, notes, err := load(background, vars(keysVar, "a key with spaces in it"), args...)
	if err == nil || !strings.Contains(err.Error(), "visible ASCII") || strings.Contains(fmt.Sprint(err, notes), "with spaces") {
		t.Errorf("a key that cannot be sent: %v\n%s", err, notes)
	}
}

func TestServerWithoutKeys(t *testing.T) {
	tg := newTarget(t, func(c *server.Config) { c.Keys, c.NoAuth = server.Keys{}, true })
	rep := measure(t, vars(), append([]string{"--url", tg.url, "--concurrency", "1", "--duration", "50ms", "--warmup", "0"}, pictures(t, "face:anna")...)...)
	if r := &rep.Runs[0]; r.OK == 0 || r.OK != r.Sent {
		t.Errorf("without keys on either side: %+v", r)
	}
}

// seen is what a stand-in for the server was sent.
type seen struct {
	mu       sync.Mutex
	paths    map[string]int
	bodies   map[string]int
	types    map[string]int
	auth     map[string]int
	requests int
}

// TestRequests looks at the requests themselves, and at answers the real
// server does not give.
func TestRequests(t *testing.T) {
	got := seen{paths: map[string]int{}, bodies: map[string]int{}, types: map[string]int{}, auth: map[string]int{}}
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		got.mu.Lock()
		got.requests++
		got.paths[r.Method+" "+r.URL.Path]++
		got.bodies[string(body)]++
		got.types[r.Header.Get("Content-Type")]++
		got.auth[r.Header.Get("Authorization")]++
		got.mu.Unlock()
		switch string(body) {
		case "sparse":
			io.WriteString(w, `{"faces":[{"match":true}],"timing_ms":{"detect":0.5,"total":1.5}}`)
		case "html":
			io.WriteString(w, "<html>not this server</html>")
		case "broken":
			http.Error(w, "broken", http.StatusInternalServerError)
		case "moved":
			http.Redirect(w, r, "/elsewhere", http.StatusFound)
		}
	}))
	defer ts.Close()
	files := pictures(t, "sparse", "html", "broken", "moved")
	args := append([]string{"--url", ts.URL + "/api/", "--concurrency", "1", "--duration", "200ms", "--warmup", "0"}, files...)

	rep := measure(t, withSearchKey, args...)
	r := &rep.Runs[0]
	consistent(t, r)
	if rep.URL != ts.URL+"/api/v1/search" {
		t.Errorf("sent to %s", rep.URL)
	}
	if r.Results.Match == 0 || r.Unreadable == 0 || r.OK != r.Results.Match+r.Unreadable {
		t.Errorf("a sparse answer and one that is no answer: %+v", r)
	}
	if len(r.Other) != 2 || r.Other[http.StatusInternalServerError] == 0 || r.Other[http.StatusFound] == 0 {
		t.Errorf("other statuses: %v", r.Other)
	}
	// stages the answers do not mention are left out, not taken for zero
	if len(r.Server) != 2 || r.Server["detect"] != 0.5 || r.Server["total"] != 1.5 {
		t.Errorf("the server's side: %v", r.Server)
	}

	got.mu.Lock()
	if len(got.paths) != 1 || got.paths["POST /api/v1/search"] != got.requests {
		t.Errorf("requests went to %v", got.paths)
	}
	if len(got.types) != 1 || got.types["application/octet-stream"] == 0 {
		t.Errorf("content types %v", got.types)
	}
	if len(got.auth) != 1 || got.auth["Bearer "+searchKey] == 0 {
		t.Errorf("a request without the key, or with something else")
	}
	for _, body := range []string{"sparse", "html", "broken", "moved"} {
		if n := got.bodies[body]; n == 0 || n < got.requests/4-1 || n > got.requests/4+1 {
			t.Errorf("picture %q was sent %d times out of %d", body, n, got.requests)
		}
	}
	got.auth = map[string]int{}
	got.mu.Unlock()

	// in text, with no key to send
	out, _, err := load(background, vars(), args...)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{" x 302, ", " x 500\n", "answers with status 200 that were not a search answer: "} {
		if !strings.Contains(out, want) {
			t.Errorf("the report lacks %q:\n%s", want, out)
		}
	}
	got.mu.Lock()
	defer got.mu.Unlock()
	if len(got.auth) != 1 || got.auth[""] == 0 {
		t.Errorf("an Authorization header was sent without a key")
	}
}

// TestTransportErrors has the server hang up instead of answering, before
// the answer or in the middle of it.
func TestTransportErrors(t *testing.T) {
	for name, hangUp := range map[string]func(w http.ResponseWriter){
		"no answer": func(http.ResponseWriter) { panic(http.ErrAbortHandler) },
		"half an answer": func(w http.ResponseWriter) {
			w.Header().Set("Content-Length", "4096")
			io.WriteString(w, `{"faces":[`)
			w.(http.Flusher).Flush()
			panic(http.ErrAbortHandler)
		},
	} {
		var mu sync.Mutex
		calls := 0
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			mu.Lock()
			calls++
			first := calls == 1
			mu.Unlock()
			if !first { // the request ahead of the run is answered
				hangUp(w)
			}
		}))
		out, notes, err := load(background, vars(), append([]string{
			"--url", ts.URL, "--concurrency", "1", "--duration", "100ms", "--warmup", "0", "--json",
		}, pictures(t, "face:anna")...)...)
		ts.Close()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		r := &decode(t, out).Runs[0]
		consistent(t, r)
		if r.Sent == 0 || r.Errors != r.Sent {
			t.Errorf("%s: every request is cut off: %+v", name, r)
		}
		if !strings.Contains(notes, "without an answer, the first because: ") {
			t.Errorf("%s: the notes do not say why:\n%s", name, notes)
		}
	}
}

func TestReportGoesNowhere(t *testing.T) {
	tg := newTarget(t)
	args := append([]string{"--url", tg.url, "--concurrency", "1", "--duration", "20ms", "--warmup", "0"}, pictures(t, "face:anna")...)
	err := Run(background, args, cli.Env{Getenv: withSearchKey, Stdout: failing{}, Stderr: io.Discard})
	if err == nil || !strings.Contains(err.Error(), "the pipe is closed") {
		t.Errorf("a report that cannot be printed: %v", err)
	}
}
