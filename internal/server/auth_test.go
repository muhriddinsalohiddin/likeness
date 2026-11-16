package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
)

func TestKeysDecideWhoMayDoWhat(t *testing.T) {
	r := newRig(t)
	r.enrol("anna", "face:anna")
	r.enrol("boris", "face:boris")
	before := r.fake.Calls()

	const stranger = "a-key-nobody-issued-0000"
	for _, c := range []struct {
		method, target, key string
		want                int
	}{
		{"POST", "/v1/search", "", 401},
		{"POST", "/v1/search", stranger, 401},
		{"POST", "/v1/search", metricsKey, 403},
		{"POST", "/v1/verify?subject=anna", "", 401},
		{"POST", "/v1/verify?subject=anna", metricsKey, 403},
		{"GET", "/v1/info", "", 401},
		{"GET", "/v1/info", metricsKey, 403},
		{"POST", "/v1/subjects/carol/faces", "", 401},
		{"POST", "/v1/subjects/carol/faces", searchKey, 403},
		{"POST", "/v1/subjects/carol/faces", metricsKey, 403},
		{"GET", "/v1/subjects", searchKey, 403},
		{"GET", "/v1/subjects/anna", searchKey, 403},
		{"DELETE", "/v1/subjects/anna", searchKey, 403},
		{"DELETE", "/v1/subjects/anna/faces/1", searchKey, 403},
		{"GET", "/metrics", "", 401},
		{"GET", "/metrics", stranger, 401},
		{"GET", "/metrics", searchKey, 403},
	} {
		a := r.send(c.method, c.target, c.key, "image/jpeg", []byte("face:carol"))
		code := "unauthorized"
		if c.want == http.StatusForbidden {
			code = "forbidden"
		}
		a.wantError(c.want, code)
		if challenge := a.header.Get("WWW-Authenticate"); (challenge != "") != (c.want == http.StatusUnauthorized) {
			t.Errorf("%s %s: WWW-Authenticate %q with status %d", c.method, c.target, challenge, c.want)
		}
	}
	if r.fake.Calls() != before || r.gallery.Info().Subjects != 2 {
		t.Error("a refused request did something")
	}

	for _, c := range []struct{ method, target, key string }{
		{"POST", "/v1/search", searchKey},
		{"POST", "/v1/search", adminKey},
		{"POST", "/v1/verify?subject=anna", searchKey},
		{"GET", "/v1/info", searchKey},
		{"GET", "/v1/info", adminKey},
		{"GET", "/v1/subjects", adminKey},
		{"GET", "/v1/subjects/anna", adminKey},
		{"GET", "/metrics", metricsKey},
		{"GET", "/metrics", adminKey},
		{"GET", "/healthz", ""},
		{"GET", "/readyz", ""},
		{"DELETE", "/v1/subjects/boris", adminKey},
	} {
		if a := r.send(c.method, c.target, c.key, "image/jpeg", []byte("face:anna")); a.status != http.StatusOK {
			t.Errorf("%s %s: status %d: %s", c.method, c.target, a.status, a.body)
		}
	}
}

func TestAuthorizationHeader(t *testing.T) {
	r := newRig(t)
	for _, c := range []struct {
		header string
		want   int
	}{
		{"Bearer " + searchKey, 200},
		{"bearer " + searchKey, 200},
		{"BEARER   " + searchKey + " ", 200},
		{searchKey, 401},
		{"Bearer", 401},
		{"Bearer ", 401},
		{"Basic " + searchKey, 401},
		{"Bearer " + searchKey + "x", 401},
		{"Bearer " + searchKey[1:], 401},
		{"Token " + searchKey, 401},
	} {
		req := httptest.NewRequest(http.MethodGet, "/v1/info", nil)
		req.Header.Set("Authorization", c.header)
		if a := r.do(req); a.status != c.want {
			t.Errorf("Authorization %q: status %d, want %d", c.header, a.status, c.want)
		}
	}
}

func TestNewKeys(t *testing.T) {
	long := strings.Repeat("k", 16)
	for name, lists := range map[string][3][]string{
		"too short":        {{"short"}, nil, nil},
		"too short search": {{long}, {"fifteen-chars-xx"[:15]}, nil},
		"too long":         {{strings.Repeat("k", 513)}, nil, nil},
		"with a space":     {{"sixteen chars or more"}, nil, nil},
		"with a newline":   {{long + "\n"}, nil, nil},
		"not ASCII":        {{"калит-калит-калит-калит"}, nil, nil},
		"twice":            {{long, long}, nil, nil},
		"in two lists":     {{long}, {long}, nil},
		"metrics and all":  {{long}, nil, {long}},
	} {
		if _, err := NewKeys(lists[0], lists[1], lists[2]); err == nil {
			t.Errorf("%s: accepted", name)
		} else if strings.Contains(err.Error(), long) {
			t.Errorf("%s: the error repeats the key: %v", name, err)
		}
	}

	keys, err := NewKeys([]string{"admin-key-number-one", "admin-key-number-two"}, []string{"search-key-number-one"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if keys.Len() != 3 {
		t.Errorf("%d keys", keys.Len())
	}
	if grants, label, ok := keys.find("admin-key-number-two"); !ok || label != "admin-2" || grants != ScopeAdmin|ScopeSearch|ScopeMetrics {
		t.Errorf("admin key: %v %q %v", grants, label, ok)
	}
	if grants, label, ok := keys.find("search-key-number-one"); !ok || label != "search-1" || grants != ScopeSearch {
		t.Errorf("search key: %v %q %v", grants, label, ok)
	}
	if _, _, ok := keys.find("search-key-number-on"); ok {
		t.Error("a prefix of a key passed for the key")
	}
	if _, _, ok := (Keys{}).find(""); ok {
		t.Error("the empty set knows a key")
	}
}

func TestWithoutAuthentication(t *testing.T) {
	r := newRig(t, func(c *Config) { c.Keys, c.NoAuth = Keys{}, true })
	r.picture("/v1/subjects/anna/faces", "", "face:anna").want(http.StatusCreated)
	r.picture("/v1/search", "", "face:anna").want(http.StatusOK)
	r.get("/v1/subjects", "").want(http.StatusOK)
	if a := r.get("/metrics", ""); a.status != http.StatusOK {
		t.Errorf("metrics: %d", a.status)
	}
	// a key nobody asked for does no harm
	r.picture("/v1/search", "whatever-this-may-be", "face:anna").want(http.StatusOK)
}

func TestWrongMethodsAndPaths(t *testing.T) {
	r := newRig(t)
	for _, c := range []struct{ method, target, allow string }{
		{"GET", "/v1/search", "POST"},
		{"PUT", "/v1/verify", "POST"},
		{"PUT", "/v1/subjects/anna", "DELETE, GET"},
		{"POST", "/v1/subjects/anna", "DELETE, GET"},
		{"POST", "/v1/subjects", "GET"},
		{"GET", "/v1/subjects/anna/faces", "POST"},
		{"GET", "/v1/subjects/anna/faces/1", "DELETE"},
		{"POST", "/v1/info", "GET"},
		{"POST", "/healthz", "GET"},
		{"DELETE", "/metrics", "GET"},
	} {
		a := r.send(c.method, c.target, adminKey, "", nil)
		a.wantError(http.StatusMethodNotAllowed, "method_not_allowed")
		if got := a.header.Get("Allow"); got != c.allow {
			t.Errorf("%s %s: Allow %q, want %q", c.method, c.target, got, c.allow)
		}
	}
	for _, target := range []string{"/", "/v1", "/v1/", "/v2/search", "/v1/subjects/", "/v1/subjects/anna/faces/1/2", "/favicon.ico"} {
		r.get(target, adminKey).wantError(http.StatusNotFound, "not_found")
	}
	// HEAD is GET without the body
	if a := r.send(http.MethodHead, "/healthz", "", "", nil); a.status != http.StatusOK {
		t.Errorf("HEAD /healthz: %d", a.status)
	}
}

func TestRequestIDs(t *testing.T) {
	r := newRig(t)
	made := regexp.MustCompile(`^[0-9a-f]{16}$`)

	first := r.get("/healthz", "").header.Get("X-Request-Id")
	second := r.get("/healthz", "").header.Get("X-Request-Id")
	if !made.MatchString(first) || !made.MatchString(second) || first == second {
		t.Errorf("ids %q and %q", first, second)
	}

	ask := func(id string) answer {
		req := httptest.NewRequest(http.MethodGet, "/v1/nowhere", nil)
		req.Header.Set("X-Request-Id", id)
		return r.do(req)
	}
	// the caller's own id is kept, in the header and in the error
	a := ask("login-7f3a.22_B")
	var e errorView
	if err := json.Unmarshal(a.body, &e); err != nil {
		t.Fatal(err)
	}
	if a.header.Get("X-Request-Id") != "login-7f3a.22_B" || e.RequestID != "login-7f3a.22_B" {
		t.Errorf("the caller's id was not kept: %v %s", a.header, a.body)
	}
	// unless it could make a mess of a log
	for _, id := range []string{`a"b`, "a b", "a\\b", "{id}", strings.Repeat("a", 65), "ünïcode"} {
		if got := ask(id).header.Get("X-Request-Id"); !made.MatchString(got) {
			t.Errorf("id %q was passed on as %q", id, got)
		}
	}
}
