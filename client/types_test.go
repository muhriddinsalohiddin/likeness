package client

import (
	"encoding/json"
	"os"
	"reflect"
	"regexp"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/muhriddinsalohiddin/likeness"
)

func TestTiming(t *testing.T) {
	var got Timing
	doc := `{"read": 0.08, "queue": 0, "decode": 2.02, "detect": 12.21, "align": 0.28, "embed": 8.07, "search": 0.004, "total": 22.68}`
	if err := json.Unmarshal([]byte(doc), &got); err != nil {
		t.Fatal(err)
	}
	want := Timing{
		Read: 80 * time.Microsecond, Decode: 2020 * time.Microsecond, Detect: 12210 * time.Microsecond,
		Align: 280 * time.Microsecond, Embed: 8070 * time.Microsecond, Search: 4 * time.Microsecond,
		Total: 22680 * time.Microsecond,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("read %+v, want %+v", got, want)
	}

	// what is decoded replaces what was there
	if err := json.Unmarshal([]byte(`{"compare": 0.002, "warm_up": 1500, "total": 1}`), &got); err != nil {
		t.Fatal(err)
	}
	want = Timing{Compare: 2 * time.Microsecond, Total: time.Millisecond, Other: map[string]time.Duration{"warm_up": 1500 * time.Millisecond}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("read %+v, want %+v", got, want)
	}

	// it is written the way it is read
	for _, tm := range []Timing{
		{},
		{Read: 80 * time.Microsecond, Queue: 3 * time.Millisecond, Decode: time.Millisecond, Detect: 2, Align: 3, Embed: 4, Store: 10360 * time.Microsecond, Total: time.Second},
		want,
	} {
		written, err := json.Marshal(tm)
		if err != nil {
			t.Fatal(err)
		}
		var back Timing
		if err := json.Unmarshal(written, &back); err != nil || !reflect.DeepEqual(back, tm) {
			t.Errorf("%+v was written as %s and read back as %+v (%v)", tm, written, back, err)
		}
	}
	written, err := json.Marshal(struct {
		Timing Timing `json:"timing_ms"`
	}{Timing{Read: 80 * time.Microsecond, Compare: 2 * time.Microsecond, Total: 26940 * time.Microsecond}})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(written), `{"timing_ms":{"align":0,"compare":0.002,"decode":0,"detect":0,"embed":0,"queue":0,"read":0.08,"total":26.94}}`; got != want {
		t.Errorf("written as %s, want %s", got, want)
	}

	for _, doc := range []string{`[1, 2]`, `{"read": "fast"}`, `"12ms"`} {
		if err := json.Unmarshal([]byte(doc), &got); err == nil {
			t.Errorf("%s was read as %+v", doc, got)
		}
	}
}

func TestLimits(t *testing.T) {
	doc := `{"max_body": 10485760, "max_inflight": 2, "queue": 16, "queue_timeout_ms": 250.5, "max_templates": 10, "search_limit": 50, "candidates": true, "max_faces": 16}`
	var got Limits
	if err := json.Unmarshal([]byte(doc), &got); err != nil {
		t.Fatal(err)
	}
	want := Limits{MaxBody: 10 << 20, MaxInflight: 2, Queue: 16, QueueTimeout: 250500 * time.Microsecond, MaxTemplates: 10, SearchLimit: 50, Candidates: true}
	if got != want {
		t.Errorf("read %+v, want %+v", got, want)
	}
	written, err := json.Marshal(Info{Limits: want})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(written), `"limits":{"max_body":10485760,"max_inflight":2,"queue":16,"max_templates":10,"search_limit":50,"candidates":true,"queue_timeout_ms":250.5}`) {
		t.Errorf("written as %s", written)
	}
	var back Info
	if err := json.Unmarshal(written, &back); err != nil || back.Limits != want {
		t.Errorf("read back as %+v (%v)", back.Limits, err)
	}
	for _, doc := range []string{`{"queue_timeout_ms": "100ms"}`, `{"queue": 1.5}`, `[]`} {
		if err := json.Unmarshal([]byte(doc), &got); err == nil {
			t.Errorf("%s was read as %+v", doc, got)
		}
	}
}

// A result that a caller logs or passes on as JSON reads like the API's.
func TestResultsAsJSON(t *testing.T) {
	r := newRig(t)
	r.enrol("anna", "face:anna")
	r.enrol("boris", "face:boris")
	res, err := r.searcher().Search(ctx, []byte("face:anna#2"), SearchOptions{})
	if err != nil {
		t.Fatal(err)
	}
	written, err := json.Marshal(res)
	if err != nil {
		t.Fatal(err)
	}
	for _, part := range []string{
		`"faces":[{"box":{"left":100,`, `"match":true,"matches":[{"subject":"anna","score":0.88`, `"margin":`,
		`"faces_found":1`, `"timing_ms":{"align":1,`, `"request_id":"` + res.RequestID + `"`,
	} {
		if !strings.Contains(string(written), part) {
			t.Errorf("no %s in %s", part, written)
		}
	}
	back := new(SearchResult)
	if err := json.Unmarshal(written, back); err != nil || !reflect.DeepEqual(back, res) {
		t.Errorf("read back as %+v (%v), want %+v", back, err, res)
	}
}

func TestBest(t *testing.T) {
	var none *SearchResult
	if none.Best() != nil || (&SearchResult{}).Best() != nil {
		t.Error("a result without faces has a best match")
	}
	res := &SearchResult{Faces: []SearchFace{
		{Match: false, Matches: []Match{{Subject: "candidate", Score: 0.65}}},
		{Match: true, Matches: []Match{{Subject: "boris", Score: 0.75}, {Subject: "anna", Score: 0.71}}},
		{Face: Face{Reject: ReasonBlurry}},
		{Match: true, Matches: []Match{{Subject: "carol", Score: 0.92}}},
		{Match: true}, // a server that contradicts itself
	}}
	if m := res.Best(); m == nil || m.Subject != "carol" || m != &res.Faces[3].Matches[0] {
		t.Errorf("the best of several faces: %+v", m)
	}
}

// codes and reasons are every constant of the package.
var (
	codes = []Code{
		CodeBadRequest, CodeInvalidParameter, CodeMissingImage, CodeMissingSubject, CodeInvalidSubject,
		CodeUnauthorised, CodeForbidden, CodeCandidatesDisabled, CodeReadOnly, CodeSubjectNotFound,
		CodeTemplateNotFound, CodeNotFound, CodeMethodNotAllowed, CodeDuplicateFace, CodeSubjectMismatch,
		CodeTooManyTemplates, CodeBodyTooLarge, CodeImageTooLarge, CodeUnsupportedImage, CodeCorruptImage,
		CodeImageTooSmall, CodeFaceRejected, CodeOverloaded, CodeInternal, CodeShuttingDown,
	}
	reasons = []Reason{
		ReasonNoFace, ReasonMultipleFaces, ReasonLowScore, ReasonOutOfFrame, ReasonFaceTooSmall,
		ReasonBadPose, ReasonTooDark, ReasonTooBright, ReasonBlurry,
	}
)

func TestReasonsAreTheEngines(t *testing.T) {
	theirs := []likeness.Reason{
		likeness.NoFace, likeness.MultipleFaces, likeness.LowScore, likeness.OutOfFrame, likeness.FaceTooSmall,
		likeness.BadPose, likeness.TooDark, likeness.TooBright, likeness.Blurry,
	}
	if len(theirs) != len(reasons) {
		t.Fatalf("%d reasons here, %d in the engine", len(reasons), len(theirs))
	}
	for i, reason := range reasons {
		if string(reason) != string(theirs[i]) {
			t.Errorf("reason %q is %q in the engine", reason, theirs[i])
		}
	}
}

// The codes and reasons in docs/api.md and the constants here have to be
// the same set, or one of the two has fallen behind.
func TestConstantsMatchTheDocumentation(t *testing.T) {
	doc, err := os.ReadFile("../docs/api.md")
	if err != nil {
		t.Skipf("no documentation to compare with: %v", err)
	}
	_, errorsOn, found := strings.Cut(string(doc), "\n## Errors\n")
	errorTable, reasonsOn, more := strings.Cut(errorsOn, "\n## Why a face is turned down\n")
	reasonTable, _, most := strings.Cut(reasonsOn, "and `quality` holds")
	if !found || !more || !most {
		t.Fatal("docs/api.md no longer has its Errors and Why a face is turned down sections where this test looks for them")
	}

	// | 404 | `subject_not_found`, `template_not_found` | |
	var documented []string
	row := regexp.MustCompile(`(?m)^\| \d{3} \| ([^|]+) \|`)
	name := regexp.MustCompile("`([a-z_]+)`")
	for _, cells := range row.FindAllStringSubmatch(errorTable, -1) {
		for _, code := range name.FindAllStringSubmatch(cells[1], -1) {
			documented = append(documented, code[1])
		}
	}
	compare(t, "error code", documented, codes)

	// | `no_face` | no face was found |
	documented = nil
	row = regexp.MustCompile(`(?m)^\| ([^|]+) \|`)
	for _, cells := range row.FindAllStringSubmatch(reasonTable, -1) {
		for _, reason := range name.FindAllStringSubmatch(cells[1], -1) {
			documented = append(documented, reason[1])
		}
	}
	compare(t, "reject reason", documented, reasons)
}

func compare[T ~string](t *testing.T, kind string, documented []string, constants []T) {
	t.Helper()
	have := make([]string, len(constants))
	for i, c := range constants {
		have[i] = string(c)
	}
	for _, name := range documented {
		if !slices.Contains(have, name) {
			t.Errorf("the %s %q is documented and has no constant", kind, name)
		}
	}
	for _, name := range have {
		if !slices.Contains(documented, name) {
			t.Errorf("the %s %q has a constant and is not documented", kind, name)
		}
	}
	if len(documented) < 5 {
		t.Errorf("found only %d %ss in docs/api.md: %v", len(documented), kind, documented)
	}
}
