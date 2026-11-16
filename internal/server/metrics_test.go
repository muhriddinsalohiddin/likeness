package server

import (
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

func TestMetrics(t *testing.T) {
	r := newRig(t)
	r.enrol("anna", "face:anna")
	r.search("/v1/search", "face:anna")
	r.search("/v1/search", "face:anna#2")
	r.search("/v1/search", "face:carol")
	r.search("/v1/search", "reject:blurry")
	r.picture("/v1/verify?subject=anna", searchKey, "face:anna").want(http.StatusOK)
	r.picture("/v1/verify?subject=anna", searchKey, "face:boris").want(http.StatusOK)
	r.picture("/v1/subjects/anna/faces", adminKey, "reject:too_dark").want(http.StatusUnprocessableEntity)
	r.picture("/v1/subjects/annette/faces", adminKey, "face:anna").want(http.StatusConflict)
	r.picture("/v1/search", "", "face:anna").want(http.StatusUnauthorized)
	r.picture("/v1/search", "nobody-gave-out-this-key", "face:anna").want(http.StatusUnauthorized)
	r.get("/v1/subjects", searchKey).want(http.StatusForbidden)
	r.get("/v1/nowhere", "").want(http.StatusNotFound)
	r.get("/healthz", "").want(http.StatusOK)

	a := r.get("/metrics", metricsKey)
	if a.status != http.StatusOK || !strings.HasPrefix(a.header.Get("Content-Type"), "text/plain; version=0.0.4") {
		t.Fatalf("status %d, Content-Type %q", a.status, a.header.Get("Content-Type"))
	}
	text := string(a.body)
	for _, line := range []string{
		`likeness_requests_total{route="search",code="200"} 4`,
		`likeness_requests_total{route="search",code="401"} 2`,
		`likeness_requests_total{route="verify",code="200"} 2`,
		`likeness_requests_total{route="enrol",code="201"} 1`,
		`likeness_requests_total{route="enrol",code="422"} 1`,
		`likeness_requests_total{route="enrol",code="409"} 1`,
		`likeness_requests_total{route="subjects",code="403"} 1`,
		`likeness_requests_total{route="other",code="404"} 1`,
		`likeness_results_total{route="search",result="match"} 2`,
		`likeness_results_total{route="search",result="no_match"} 1`,
		`likeness_results_total{route="search",result="rejected"} 1`,
		`likeness_results_total{route="verify",result="verified"} 1`,
		`likeness_results_total{route="verify",result="not_verified"} 1`,
		`likeness_results_total{route="enrol",result="enrolled"} 1`,
		`likeness_results_total{route="enrol",result="rejected"} 1`,
		`likeness_results_total{route="enrol",result="duplicate"} 1`,
		`likeness_rejected_faces_total{route="search",reason="blurry"} 1`,
		`likeness_rejected_faces_total{route="enrol",reason="too_dark"} 1`,
		`likeness_unauthorized_total{reason="no_key"} 1`,
		`likeness_unauthorized_total{reason="unknown_key"} 1`,
		`likeness_unauthorized_total{reason="no_right"} 1`,
		`likeness_request_duration_seconds_count{route="search"} 6`,
		`likeness_request_duration_seconds_bucket{route="search",le="+Inf"} 6`,
		// nine pictures reached the engine, which takes 2 ms to decode one
		`likeness_stage_duration_seconds_count{stage="decode"} 9`,
		`likeness_stage_duration_seconds_bucket{stage="decode",le="0.001"} 0`,
		`likeness_stage_duration_seconds_bucket{stage="decode",le="0.0025"} 9`,
		`likeness_stage_duration_seconds_count{stage="store"} 2`,
		`likeness_inflight_requests 0`,
		`likeness_queued_requests 0`,
		`likeness_pending_requests 0`,
		`likeness_subjects 1`,
		`likeness_templates 1`,
		"# TYPE likeness_requests_total counter",
		"# TYPE likeness_request_duration_seconds histogram",
		"# TYPE likeness_subjects gauge",
	} {
		if !strings.Contains(text, line+"\n") {
			t.Errorf("missing %q", line)
		}
	}
	// and 3 ms to find the faces in it
	if m := regexp.MustCompile(`likeness_stage_duration_seconds_sum\{stage="detect"\} (\S+)\n`).FindStringSubmatch(text); m == nil {
		t.Error("no sum for the detect stage")
	} else if sum, _ := strconv.ParseFloat(m[1], 64); !near(sum, 0.027, 1e-9) {
		t.Errorf("detect took %v s in all", sum)
	}
	if !regexp.MustCompile(`likeness_build_info\{version="[^"]+",preset="fake",detector="fake-detector",embedder="fake-embedder",kernel="[^"]+",go="go[^"]+"\} 1\n`).MatchString(text) {
		t.Errorf("build info:\n%s", text)
	}
	// health checks and scrapes are not counted, and nothing names a person
	for _, absent := range []string{`route="healthz"`, `route="metrics"`, "anna"} {
		if strings.Contains(text, absent) {
			t.Errorf("%q should not be in the metrics", absent)
		}
	}

	// every line is a comment or a sample, and buckets only ever grow
	sample := regexp.MustCompile(`^[a-z_]+(\{([a-z_]+="[^"]*",?)+\})? (\d+(\.\d+)?(e-?\d+)?)$`)
	var last float64
	for _, line := range strings.Split(strings.TrimSuffix(text, "\n"), "\n") {
		if strings.HasPrefix(line, "# ") {
			continue
		}
		m := sample.FindStringSubmatch(line)
		if m == nil {
			t.Errorf("malformed line %q", line)
			continue
		}
		if strings.Contains(line, "_bucket{") {
			v, _ := strconv.ParseFloat(m[3], 64)
			if !strings.Contains(line, `le="0.005"`) && !strings.Contains(line, `le="0.0005"`) && v < last {
				t.Errorf("a bucket smaller than the one before it: %q", line)
			}
			last = v
		}
	}

	// asking twice gives the same answer: reading the metrics changes nothing
	if again := r.get("/metrics", metricsKey); string(again.body) != text {
		t.Error("two scrapes in a row differ")
	}
}

func TestLabelEscaping(t *testing.T) {
	m := newMetrics()
	var out strings.Builder
	if err := m.write(&out, gauges{build: [][2]string{{"version", "1.0\"\n\\beta"}}}); err != nil {
		t.Fatal(err)
	}
	if want := `likeness_build_info{version="1.0\"\n\\beta"} 1` + "\n"; !strings.Contains(out.String(), want) {
		t.Errorf("label value not escaped:\n%s", out.String())
	}
}

func TestHistogramBuckets(t *testing.T) {
	hs := histograms{bounds: []float64{0.01, 0.1, 1}}
	for _, v := range []float64{0.005, 0.01, 0.011, 0.5, 2, 3} {
		hs.observe("x", v)
	}
	h := hs.h["x"]
	// a value on a bound belongs to that bound's bucket
	if got := h.counts; got[0] != 2 || got[1] != 1 || got[2] != 1 || got[3] != 2 || h.n != 6 {
		t.Errorf("buckets %v, count %d", got, h.n)
	}
}
