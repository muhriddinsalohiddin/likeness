// Package load is the `likeness load` command: it drives a running server
// with search requests and reports how the server holds up.
//
// It answers what a benchmark inside the process cannot: what a client
// experiences over HTTP, and whether a server with more work than it can
// carry refuses the excess at once, so that what it accepts stays fast.
package load

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/muhriddinsalohiddin/likeness/internal/cli"
)

const usage = `usage: likeness load --url URL [flags] PICTURE...

Sends the pictures in rotation to POST /v1/search of a running server and
reports what a client sees: how many requests were answered and how many
refused with 429, how long both took, and where the server says the time
went.

By default the load is a closed loop: --concurrency workers each send a
request, wait for the answer and send the next. A closed loop slows down
with the server, so it cannot overload it. With --rate, or with --steps for
several rates in turn, requests start on a fixed schedule whether or not
earlier ones have finished: the way to see what the server does with more
work than it can carry.

flags:
`

const environment = `
environment:
  LIKENESS_SEARCH_KEYS   the API key to send: the first one of the list. If it
                         is not set no key is sent, which suits a server run
                         with --insecure-no-auth.

The key can be read from a file instead, named by LIKENESS_SEARCH_KEYS_FILE.
`

// keysVar names the variable the API key is taken from: the one the server
// reads its search keys from, so that one environment serves both.
const keysVar = "LIKENESS_SEARCH_KEYS"

// maxRate keeps the schedule of an open loop within what a timer can tell
// apart.
const maxRate = 100000

type options struct {
	// search is the address of the search endpoint.
	search *url.URL
	// key is the API key to present; empty for none.
	key string
	// concurrency is the number of workers of a closed loop.
	concurrency int
	// rates are the requests per second of an open loop, one run for each;
	// none means a closed loop.
	rates []float64
	// stepped says that the rates came from --steps: the report then names
	// the highest one the server carried.
	stepped    bool
	maxPending int
	duration   time.Duration
	warmup     time.Duration
	timeout    time.Duration
	target     time.Duration
	json       bool
	pictures   [][]byte
}

// parse reads the command line, the environment and the pictures.
func parse(args []string, env cli.Env) (*options, error) {
	o := &options{}
	var (
		address, steps string
		rate           float64
	)
	fs := cli.Flags("likeness load", env, usage, environment)
	fs.StringVar(&address, "url", "", "`address` of the server, as in http://localhost:8080; a path in it is kept (required)")
	fs.IntVar(&o.concurrency, "concurrency", 4, "closed loop: `workers` that each send a request, wait for the answer and send the next")
	fs.Float64Var(&rate, "rate", 0, "open loop: `requests` started per second, whether or not earlier ones have finished")
	fs.StringVar(&steps, "steps", "", "open loop at each of these `rates` in turn, as in 5,10,20,40,80: one row for each")
	fs.IntVar(&o.maxPending, "max-pending", 512, "most `requests` of an open loop in flight; a request due beyond that is skipped and counted")
	fs.DurationVar(&o.duration, "duration", 10*time.Second, "how long to measure; with --steps, each rate")
	fs.DurationVar(&o.warmup, "warmup", time.Second, "requests started this long before the measuring are sent but not counted; with --steps, before each rate")
	fs.DurationVar(&o.timeout, "timeout", 5*time.Second, "longest wait for an answer")
	fs.DurationVar(&o.target, "target", 100*time.Millisecond, "latency the accepted requests should stay within")
	fs.BoolVar(&o.json, "json", false, "print one JSON document instead of the tables")
	if err := cli.Parse(fs, args); err != nil {
		return nil, err
	}
	given := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) { given[f.Name] = true })
	loops := 0
	for _, name := range []string{"concurrency", "rate", "steps"} {
		if given[name] {
			loops++
		}
	}

	stray := ""
	for _, name := range fs.Args() {
		if strings.HasPrefix(name, "-") {
			stray = name
			break
		}
	}

	problem := ""
	switch {
	case stray != "":
		problem = stray + " comes after a picture; flags go before the pictures"
	case address == "":
		problem = "--url is required: the server to send to, as in http://localhost:8080"
	case loops > 1:
		problem = "--concurrency, --rate and --steps exclude each other: a loop is closed or open, at one rate or at several"
	case o.concurrency < 1:
		problem = "--concurrency is at least 1"
	case given["rate"] && !validRate(rate):
		problem = fmt.Sprintf("--rate is a number of requests per second, above 0 and at most %d", maxRate)
	case o.maxPending < 1:
		problem = "--max-pending is at least 1"
	case o.duration <= 0 || o.timeout <= 0 || o.target <= 0:
		problem = "--duration, --timeout and --target have to be above 0"
	case o.warmup < 0:
		problem = "--warmup cannot be negative"
	case fs.NArg() == 0:
		problem = "name at least one picture to send"
	}
	if problem != "" {
		return nil, cli.Usage(fs, "%s", problem)
	}
	var err error
	if o.search, err = searchURL(address); err != nil {
		return nil, cli.Usage(fs, "%v", err)
	}
	switch {
	case given["rate"]:
		o.rates = []float64{rate}
	case given["steps"]:
		if o.rates, err = parseSteps(steps); err != nil {
			return nil, cli.Usage(fs, "%v", err)
		}
		o.stepped = true
	}

	list, err := cli.Secret(env.Getenv, keysVar)
	if err != nil {
		return nil, err
	}
	if o.key, err = firstKey(list); err != nil {
		return nil, err
	}

	for _, name := range fs.Args() {
		picture, err := os.ReadFile(name)
		if err != nil {
			return nil, err
		}
		if len(picture) == 0 {
			return nil, fmt.Errorf("%s is empty; a picture is a JPEG, PNG or WebP file", name)
		}
		o.pictures = append(o.pictures, picture)
	}
	return o, nil
}

func validRate(rate float64) bool { return rate > 0 && rate <= maxRate }

// parseSteps reads the rates of --steps.
func parseSteps(list string) ([]float64, error) {
	fields := strings.Split(list, ",")
	rates := make([]float64, 0, len(fields))
	for _, field := range fields {
		rate, err := strconv.ParseFloat(strings.TrimSpace(field), 64)
		if err != nil || !validRate(rate) {
			return nil, fmt.Errorf("--steps is a list of rates such as 5,10,20,40,80, each above 0 and at most %d; %q is not one",
				maxRate, strings.TrimSpace(field))
		}
		rates = append(rates, rate)
	}
	return rates, nil
}

// searchURL turns the address of a server into that of its search
// endpoint. The address is not repeated in the error: it may hold a
// password.
func searchURL(address string) (*url.URL, error) {
	u, err := url.Parse(strings.TrimSpace(address))
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" || u.Fragment != "" {
		return nil, errors.New("--url has to be the address of a server, as in http://localhost:8080 or https://faces.example.com/api")
	}
	return u.JoinPath("v1", "search"), nil
}

// firstKey picks the key to present out of a list separated by commas or
// line breaks. The key is never repeated in an error.
func firstKey(list string) (string, error) {
	for _, key := range strings.FieldsFunc(list, func(r rune) bool { return r == ',' || r == '\n' || r == '\r' }) {
		if key = strings.TrimSpace(key); key == "" {
			continue
		}
		for i := 0; i < len(key); i++ {
			if c := key[i]; c <= ' ' || c > '~' {
				return "", fmt.Errorf("the first key in %s contains a space or a character outside visible ASCII", keysVar)
			}
		}
		return key, nil
	}
	return "", nil
}

// Run drives the server until the measuring is over or ctx is done, and
// prints the report. It is a cli.Command.
func Run(ctx context.Context, args []string, env cli.Env) error {
	o, err := parse(args, env)
	if err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	note := func(format string, args ...any) { say(env.Stderr, "likeness load: "+format+"\n", args...) }
	l := newLoader(o)
	defer l.client.CloseIdleConnections()

	note("searching at %s with %s", o.search.Redacted(), count(len(o.pictures), "picture"))
	if err := l.probe(); err != nil {
		return err
	}

	rep := newReport(o)
	if len(o.rates) == 0 {
		note("%s for %s%s", count(o.concurrency, "worker"), o.duration, o.afterWarmup())
		t, measured := l.closed(ctx)
		rep.add(o, 0, t, measured, note)
	}
	for _, rate := range o.rates {
		if ctx.Err() != nil {
			break
		}
		note("%s requests per second for %s%s", number(rate), o.duration, o.afterWarmup())
		t, measured := l.open(ctx, rate)
		rep.add(o, rate, t, measured, note)
	}
	if ctx.Err() != nil {
		rep.Interrupted = true
		note("interrupted; the report covers what was measured until then")
	}
	rep.conclude(o.stepped)

	if err := rep.print(env.Stdout, o.json); err != nil {
		return err
	}
	return ctx.Err()
}

func (o *options) afterWarmup() string {
	if o.warmup == 0 {
		return ""
	}
	return fmt.Sprintf(", after %s of warm-up", o.warmup)
}

// count writes a number of things: "1 picture", "4 workers".
func count(n int, thing string) string {
	if n == 1 {
		return "1 " + thing
	}
	return strconv.Itoa(n) + " " + thing + "s"
}

// number writes a rate or a count without digits it does not need.
func number(v float64) string { return strconv.FormatFloat(v, 'f', -1, 64) }

func say(w io.Writer, format string, args ...any) { _, _ = fmt.Fprintf(w, format, args...) }
