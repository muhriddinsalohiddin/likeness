package eval

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/muhriddinsalohiddin/likeness"
	measure "github.com/muhriddinsalohiddin/likeness/internal/eval"
)

func TestPercent(t *testing.T) {
	for rate, want := range map[float64]string{
		0: "0%", 1: "100%", 0.5: "50.0%", 0.123456: "12.3%", 0.1: "10.0%", 0.0999: "9.99%", 0.0123456: "1.23%", 0.01: "1.00%",
		0.0050251: "0.50%", 0.00123456: "0.12%", 0.00012: "0.012%", 0.0000123: "0.0012%", 1.0050335853501442e-05: "0.0010%",
		1e-8: "0.0000010%", 0.99999: "100.0%",
	} {
		if got := percent(rate); got != want {
			t.Errorf("a rate of %v is written %s, want %s", rate, got, want)
		}
	}
	for rate, want := range map[float64]string{
		1e-2: "1%", 1e-3: "0.1%", 1e-4: "0.01%", 1e-5: "0.001%", 1e-6: "0.0001%", 0.05: "5%", 0.25: "25%", 1: "100%",
	} {
		if got := exactPercent(rate); got != want {
			t.Errorf("a target of %v is written %s, want %s", rate, got, want)
		}
	}
	for similarity, want := range map[float64]string{0.363: "0.363", 0.5: "0.500", -0.0114: "-0.011", 1: "1.000", 0.50071: "0.501"} {
		if got := fixed(similarity); got != want {
			t.Errorf("a similarity of %v is written %s, want %s", similarity, got, want)
		}
	}
	// the engine's thresholds are float32: 0.363 is not 0.36300000548362732
	for threshold, want := range map[float32]float64{0.363: 0.363, 0.6: 0.6, 0.7: 0.7, 0.25: 0.25, 0.28: 0.28} {
		if got := similarity(threshold); got != want {
			t.Errorf("the threshold %v became %v", threshold, got)
		}
	}
}

func TestWords(t *testing.T) {
	if probes(1) != "1 probe" || probes(0) != "0 probes" || probes(48) != "48 probes" {
		t.Errorf("counting probes: %s, %s, %s", probes(1), probes(0), probes(48))
	}
	if people(1) != "1 person" || people(0) != "0 people" || people(16) != "16 people" {
		t.Errorf("counting people: %s, %s, %s", people(1), people(0), people(16))
	}

	text := "For a gallery of 2 people, use --search-threshold 0.297. About 1% of searches will then name someone."
	for width, want := range map[int]string{
		200: text,
		60:  "For a gallery of 2 people, use --search-threshold 0.297.|About 1% of searches will then name someone.",
		30:  "For a gallery of 2 people, use|--search-threshold 0.297.|About 1% of searches will then|name someone.",
		// a word longer than the line stays whole
		10: "For a|gallery of|2 people,|use|--search-threshold|0.297.|About 1%|of|searches|will then|name|someone.",
	} {
		if got := strings.Join(wrapped(text, width), "|"); got != want {
			t.Errorf("at %d characters:\n%s\nwant:\n%s", width, got, want)
		}
	}
	if got := wrapped("", 10); len(got) != 1 || got[0] != "" {
		t.Errorf("an empty paragraph: %q", got)
	}

	var buf bytes.Buffer
	table(&buf, []string{"gallery", "threshold"}, []string{"1000", "0.412"}, []string{"100000", "-", "not enough pairs"})
	want := "  gallery  threshold\n" +
		"     1000      0.412\n" +
		"   100000          -  not enough pairs\n"
	if buf.String() != want {
		t.Errorf("a table:\n%s\nwant:\n%s", buf.String(), want)
	}
}

// measured makes distributions by hand: impostor pairs spread evenly from
// 0 up to 0.5, and genuine ones of which a tenth score 0.3 and the rest
// 0.9.
func measured(genuine, impostor int) *measure.Scores {
	s := &measure.Scores{}
	for n := range genuine {
		if n%10 == 0 {
			s.Genuine.Add(0.3)
		} else {
			s.Genuine.Add(0.9)
		}
	}
	for n := range impostor {
		s.Impostor.Add(0.5 * float32(n) / float32(impostor))
	}
	return s
}

func TestAdvice(t *testing.T) {
	info := likeness.Info{Preset: "default", VerifyThreshold: 0.363, SearchThreshold: 0.6}
	nobody, err := measure.Identify(context.Background(), nil, 5, 0)
	if err != nil {
		t.Fatal(err)
	}
	compiled := func(impostor int, galleries ...int) (*report, string) {
		o := &options{dir: "faces", policy: "search", holdout: 5, galleries: galleries}
		r := compile(o, info, &counts{}, measured(1000, impostor), nobody)
		return r, string(r.text())
	}

	// Plenty of pairs: the usual FMR, and the largest gallery within
	// reach. A gallery of 100 needs 99 500 impostor pairs, one of 1000
	// ten times as many.
	r, text := compiled(200_000, 1000, 10, 100)
	if a := r.Advice.Verify; a == nil || a.FMR != 1e-4 || *a.Threshold != 0.5 || *a.FNMR != 0.1 {
		t.Errorf("the verify advice: %+v", a)
	}
	if a := r.Advice.Search; a == nil || a.Gallery != 100 || a.FPIR != 0.01 || *a.Threshold != 0.5 || *a.FNMR != 0.1 {
		t.Errorf("the search advice: %+v", a)
	}
	for _, want := range []string{
		"For a gallery of 100 people, use --search-threshold 0.500.",
		"about 10.0% of searches for somebody",
		"For verification, use --verify-threshold 0.500: an FMR of 0.01% and an FNMR of 10.0%.",
		"The preset's verify threshold, 0.363: FMR 27.4%, FNMR 10.0%.",
		// nobody was searched for
		"a gallery of 0 people",
		"0 probes of enrolled people, 0 probes of the 0 people kept out of the gallery (--holdout 5)",
		"      0.600     -     -  the preset's search threshold\n",
	} {
		if !strings.Contains(strings.Join(strings.Fields(text), " "), strings.Join(strings.Fields(want), " ")) {
			t.Errorf("the report lacks %q:\n%s", want, text)
		}
	}
	if strings.Contains(text, "Rank-1") || strings.Contains(text, "lowest these pictures") {
		t.Errorf("the report speaks of what was not measured:\n%s", text)
	}

	// More pairs still do not make the advice stricter than the usual
	// FMR, though they would support it.
	r, _ = compiled(1_200_000, 1000)
	if a, stricter := r.Advice.Verify, r.Verification.Targets[3]; a == nil || a.FMR != 1e-4 || !stricter.Supported {
		t.Errorf("the verify advice: %+v, with an FMR of %v supported: %v", a, stricter.FMR, stricter.Supported)
	}
	if a := r.Advice.Search; a == nil || a.Gallery != 1000 {
		t.Errorf("the search advice: %+v", a)
	}

	// Fewer pairs: a laxer FMR, said to be one; and no gallery within
	// reach, of which the smallest is named.
	r, text = compiled(20_000, 100_000, 5000)
	if a := r.Advice.Verify; a == nil || a.FMR != 1e-3 {
		t.Errorf("the verify advice: %+v", a)
	}
	if r.Advice.Search != nil {
		t.Errorf("the search advice: %+v", r.Advice.Search)
	}
	for _, want := range []string{
		"These pictures are too few to say which --search-threshold a gallery of 5000 people needs: that takes 4974959 pairs of different people, and there are 20000. Evaluate more people.",
		"an FMR of 0.1%, the lowest these pictures can measure, and an FNMR of 10.0%.",
	} {
		if !strings.Contains(strings.Join(strings.Fields(text), " "), want) {
			t.Errorf("the report lacks %q:\n%s", want, text)
		}
	}

	// Hardly any pairs: no advice at all.
	r, text = compiled(500, 1000)
	if r.Advice.Verify != nil || r.Advice.Search != nil {
		t.Errorf("advice out of 500 pairs: %+v", r.Advice)
	}
	for _, want := range []string{
		"too few to say which --search-threshold a gallery of 1000 people needs",
		"These pictures are too few to advise a --verify-threshold: an FMR of 1% takes 1000 pairs of different people, and there are 500.",
		"These figures hold for pictures like the ones evaluated.",
	} {
		if !strings.Contains(strings.Join(strings.Fields(text), " "), want) {
			t.Errorf("the report lacks %q:\n%s", want, text)
		}
	}
}
