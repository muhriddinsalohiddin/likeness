package server

import (
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
)

// params are the options of one request: its query string and, for a form
// upload, the text fields of the form.
//
// The readers below remember the first thing that is wrong and finish
// reports it, so a handler reads all its options and checks once. An option
// nobody read is an error as well: a misspelt "replace" that is silently
// ignored would add a template where the caller meant to swap one.
type params struct {
	values map[string]string
	err    *apiError
}

func gather(r *http.Request, fields map[string]string) (*params, *apiError) {
	query, err := url.ParseQuery(r.URL.RawQuery)
	if err != nil {
		return nil, refuse(http.StatusBadRequest, "bad_request", "the query string is malformed")
	}
	p := &params{values: make(map[string]string, len(query)+len(fields))}
	for name, given := range query {
		if len(given) != 1 {
			return nil, invalid(name, "is given more than once")
		}
		p.values[name] = given[0]
	}
	for name, value := range fields {
		if _, twice := p.values[name]; twice {
			return nil, invalid(name, "is given both in the query string and in the form")
		}
		p.values[name] = value
	}
	return p, nil
}

func invalid(name, problem string) *apiError {
	return refuse(http.StatusBadRequest, "invalid_parameter", "%s %s", name, problem)
}

func (p *params) fail(name, problem string) {
	if p.err == nil {
		p.err = invalid(name, problem)
	}
}

// take removes and returns an option.
func (p *params) take(name string) (string, bool) {
	v, ok := p.values[name]
	delete(p.values, name)
	return v, ok
}

// flag reads a yes-or-no option; absent means no.
func (p *params) flag(name string) bool {
	v, ok := p.take(name)
	if !ok {
		return false
	}
	switch strings.ToLower(v) {
	case "true", "1":
		return true
	case "false", "0":
		return false
	}
	p.fail(name, "must be true or false")
	return false
}

// number reads a whole number between lo and hi; absent means def.
func (p *params) number(name string, def, lo, hi int) int {
	v, ok := p.take(name)
	if !ok {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < lo || n > hi {
		p.fail(name, "must be a number from "+strconv.Itoa(lo)+" to "+strconv.Itoa(hi))
		return def
	}
	return n
}

// fraction reads a similarity: above 0, at most 1.
func (p *params) fraction(name string) (float32, bool) {
	v, ok := p.take(name)
	if !ok {
		return 0, false
	}
	f, err := strconv.ParseFloat(v, 32)
	if err != nil || !(f > 0 && f <= 1) {
		p.fail(name, "must be a number above 0 and at most 1")
		return 0, false
	}
	return float32(f), true
}

// oneOf reads an option with a fixed set of values, the first of which is
// the default.
func (p *params) oneOf(name string, allowed ...string) string {
	v, ok := p.take(name)
	if !ok {
		return allowed[0]
	}
	for _, a := range allowed {
		if v == a {
			return v
		}
	}
	p.fail(name, "must be one of: "+strings.Join(allowed, ", "))
	return allowed[0]
}

// finish reports the first problem found, or an option nothing asked for.
func (p *params) finish() *apiError {
	if p.err != nil {
		return p.err
	}
	if len(p.values) == 0 {
		return nil
	}
	unknown := make([]string, 0, len(p.values))
	for name := range p.values {
		unknown = append(unknown, name)
	}
	sort.Strings(unknown)
	return invalid(strconv.Quote(unknown[0]), "is not an option of this request")
}
