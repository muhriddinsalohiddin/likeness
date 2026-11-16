package eval

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/muhriddinsalohiddin/likeness"
	measure "github.com/muhriddinsalohiddin/likeness/internal/eval"
)

var (
	// fmrTargets are the false match rates a verify threshold is worked
	// out for.
	fmrTargets = []float64{1e-2, 1e-3, 1e-4, 1e-5, 1e-6}
	// fpirTargets are the shares of searches for a stranger that may name
	// somebody, for which a search threshold is worked out.
	fpirTargets = []float64{0.01, 0.001}
)

// The closing advice is given for these targets.
const (
	advisedFMR  = 1e-4
	advisedFPIR = 0.01
)

// distribution sums up the scores of one kind of pair.
type distribution struct {
	Pairs  uint64  `json:"pairs"`
	Mean   float64 `json:"mean"`
	StdDev float64 `json:"std_dev"`
}

// operating is a verify threshold and the error rates it has on the
// pictures. Rates are fractions here and everywhere in the JSON form.
type operating struct {
	Threshold float64 `json:"threshold"`
	FMR       float64 `json:"fmr"`
	FNMR      float64 `json:"fnmr"`
}

// setting is the threshold that keeps the false match rate of one
// comparison within a target, and the share of genuine pairs it turns
// away. Both are absent when the pictures make too few impostor pairs to
// measure a rate that low.
type setting struct {
	FMR         float64  `json:"fmr_target"`
	Supported   bool     `json:"supported"`
	PairsNeeded uint64   `json:"impostor_pairs_needed"`
	Threshold   *float64 `json:"threshold,omitempty"`
	FNMR        *float64 `json:"fnmr,omitempty"`
}

type verification struct {
	Genuine      distribution `json:"genuine"`
	Impostor     distribution `json:"impostor"`
	EER          float64      `json:"eer"`
	EERThreshold float64      `json:"eer_threshold"`
	// Current is how the preset's own verify threshold does.
	Current operating `json:"current"`
	Targets []setting `json:"fmr_targets"`
}

// searched is how the searches of the trial went at one threshold. FNIR is
// absent when no probe had its person enrolled, FPIR when every probe had.
type searched struct {
	Threshold float64  `json:"threshold"`
	FNIR      *float64 `json:"fnir,omitempty"`
	FPIR      *float64 `json:"fpir,omitempty"`
}

type identification struct {
	Holdout  int      `json:"holdout"`
	Gallery  int      `json:"gallery"`
	LeftOut  int      `json:"left_out"`
	Mated    int      `json:"mated_probes"`
	NonMated int      `json:"non_mated_probes"`
	Rank1    *float64 `json:"rank1_hit_rate,omitempty"`
	// Current is how the preset's own search threshold does, Thresholds
	// how the ones worked out for the gallery sizes do, lowest first.
	Current    searched   `json:"current"`
	Thresholds []searched `json:"thresholds"`
}

// gallerySetting is the search threshold for a gallery of a given size:
// the one at which a search for a stranger names somebody no more often
// than the FPIR target.
type gallerySetting struct {
	Gallery int     `json:"gallery_size"`
	FPIR    float64 `json:"fpir_target"`
	setting
}

// advice is what to run the server with. A part is absent when the
// pictures cannot support any figure for it.
type advice struct {
	Search *gallerySetting `json:"search,omitempty"`
	Verify *setting        `json:"verify,omitempty"`
}

// report is everything one run measured. The field names in its JSON form
// are what scripts read; they stay as they are.
type report struct {
	Version        string           `json:"version"`
	Preset         string           `json:"preset"`
	Detector       string           `json:"detector"`
	Embedder       string           `json:"embedder"`
	Fingerprint    string           `json:"fingerprint"`
	Dataset        string           `json:"dataset"`
	Policy         string           `json:"policy"`
	People         int              `json:"people"`
	Pictures       int              `json:"pictures"`
	UsablePeople   int              `json:"usable_people"`
	Usable         int              `json:"usable_pictures"`
	TurnedDown     map[string]int   `json:"turned_down"`
	Unreadable     map[string]int   `json:"unreadable"`
	Verification   verification     `json:"verification"`
	Identification identification   `json:"identification"`
	Galleries      []gallerySetting `json:"search_thresholds"`
	Advice         advice           `json:"advice"`
}

// similarity rounds a threshold of the engine, a float32, to what it was
// written as.
func similarity(score float32) float64 { return math.Round(float64(score)*1e4) / 1e4 }

// settingFor works out the threshold for a false match rate.
func settingFor(s *measure.Scores, fmr float64) setting {
	set := setting{FMR: fmr, PairsNeeded: measure.PairsNeeded(fmr)}
	if p := s.AtFMR(fmr); p.Supported {
		set.Supported, set.Threshold, set.FNMR = true, &p.Threshold, &p.FNMR
	}
	return set
}

// compile puts the measurements together.
func compile(o *options, info likeness.Info, c *counts, s *measure.Scores, id *measure.Identification) *report {
	r := &report{
		Version: likeness.Version,
		Preset:  info.Preset, Detector: info.Detector, Embedder: info.Embedder, Fingerprint: info.Fingerprint,
		Dataset: o.dir, Policy: o.policy,
		People: c.people, Pictures: c.pictures, UsablePeople: c.usablePeople, Usable: c.usable,
		TurnedDown: c.turnedDown, Unreadable: c.unreadable,
	}

	v := &r.Verification
	v.Genuine = distribution{s.Genuine.Count(), s.Genuine.Mean(), s.Genuine.StdDev()}
	v.Impostor = distribution{s.Impostor.Count(), s.Impostor.Mean(), s.Impostor.StdDev()}
	v.EER, v.EERThreshold, _ = s.EER()
	verify := similarity(info.VerifyThreshold)
	v.Current = operating{Threshold: verify, FMR: s.FMR(verify), FNMR: s.FNMR(verify)}
	for _, fmr := range fmrTargets {
		v.Targets = append(v.Targets, settingFor(s, fmr))
	}

	var thresholds []float64
	for _, gallery := range o.galleries {
		for _, fpir := range fpirTargets {
			row := gallerySetting{Gallery: gallery, FPIR: fpir, setting: settingFor(s, measure.SearchFMR(fpir, gallery))}
			r.Galleries = append(r.Galleries, row)
			if row.Supported {
				thresholds = append(thresholds, *row.Threshold)
			}
		}
	}

	at := func(threshold float64) searched {
		out := searched{Threshold: threshold}
		if id.Mated > 0 {
			fnir := id.FNIR(threshold)
			out.FNIR = &fnir
		}
		if id.NonMated > 0 {
			fpir := id.FPIR(threshold)
			out.FPIR = &fpir
		}
		return out
	}
	i := &r.Identification
	*i = identification{
		Holdout: o.holdout, Gallery: id.Gallery, LeftOut: id.LeftOut, Mated: id.Mated, NonMated: id.NonMated,
		Current: at(similarity(info.SearchThreshold)), Thresholds: []searched{},
	}
	if id.Mated > 0 {
		rank1 := id.Rank1()
		i.Rank1 = &rank1
	}
	slices.Sort(thresholds)
	for _, t := range slices.Compact(thresholds) {
		i.Thresholds = append(i.Thresholds, at(t))
	}

	// The search threshold is advised for the largest gallery the
	// pictures can speak for, the verify threshold for the usual target
	// or, failing that, the strictest one they can.
	for n, row := range r.Galleries {
		if row.FPIR == advisedFPIR && row.Supported && (r.Advice.Search == nil || row.Gallery > r.Advice.Search.Gallery) {
			r.Advice.Search = &r.Galleries[n]
		}
	}
	for n, target := range v.Targets {
		if target.Supported && target.FMR >= advisedFMR {
			r.Advice.Verify = &v.Targets[n]
		}
	}
	return r
}

// writeJSON writes the report to a file as one JSON document.
func (r *report) writeJSON(file string) error {
	doc, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(file, append(doc, '\n'), 0o644)
}

// text is the report for a person to read.
func (r *report) text() []byte {
	buf := &bytes.Buffer{}
	fmt.Fprintf(buf, "likeness %s\n", r.Version)
	head := tabwriter.NewWriter(buf, 0, 0, 2, ' ', 0)
	line := func(name, format string, args ...any) {
		_, _ = fmt.Fprintf(head, "  %s\t%s\n", name, fmt.Sprintf(format, args...))
	}
	line("models", "%s: %s and %s", r.Preset, r.Detector, r.Embedder)
	line("fingerprint", "%s", r.Fingerprint)
	line("dataset", "%s", r.Dataset)
	line("pictures", "%d, of %d people", r.Pictures, r.People)
	line("usable", "%d, of %d people, under the %s policy", r.Usable, r.UsablePeople, r.Policy)
	if len(r.TurnedDown) > 0 {
		line("turned down", "%s", tallied(r.TurnedDown))
	}
	if len(r.Unreadable) > 0 {
		line("unreadable", "%s", tallied(r.Unreadable))
	}
	_ = head.Flush()

	v := r.Verification
	fmt.Fprintf(buf, "\nVerification: every picture against every other one\n")
	spread := func(name string, d distribution) []string {
		return []string{name, strconv.FormatUint(d.Pairs, 10), fixed(d.Mean), fixed(d.StdDev)}
	}
	table(buf, []string{"pairs", "count", "mean", "deviation"},
		spread("same person", v.Genuine), spread("different people", v.Impostor))
	fmt.Fprintf(buf, "  Equal error rate: %s, at the threshold %s.\n", percent(v.EER), fixed(v.EERThreshold))
	fmt.Fprintf(buf, "  The preset's verify threshold, %s: FMR %s, FNMR %s.\n",
		fixed(v.Current.Threshold), percent(v.Current.FMR), percent(v.Current.FNMR))
	fmt.Fprintf(buf, "  FMR is the share of different people taken for the same person, FNMR the share of\n"+
		"  pictures of the same person not taken for them.\n\n")
	rows := make([][]string, len(v.Targets))
	for n, t := range v.Targets {
		rows[n] = append([]string{exactPercent(t.FMR)}, t.cells(v.Impostor.Pairs)...)
	}
	table(buf, []string{"FMR target", "threshold", "FNMR"}, rows...)

	i := r.Identification
	fmt.Fprintf(buf, "\nIdentification: searches in a gallery of %s, the first picture of each\n", people(i.Gallery))
	fmt.Fprintf(buf, "  %s of enrolled people", probes(i.Mated))
	if i.Holdout > 0 {
		fmt.Fprintf(buf, ", %s of the %s kept out of the gallery (--holdout %d)",
			probes(i.NonMated), people(i.LeftOut), i.Holdout)
	}
	fmt.Fprintf(buf, "\n")
	if i.Rank1 != nil {
		fmt.Fprintf(buf, "  Rank-1 hit rate: %s of the enrolled are the most similar person in the gallery,\n"+
			"  whatever the score.\n", percent(*i.Rank1))
	}
	fmt.Fprintf(buf, "  FNIR is the share of searches for an enrolled person that do not name them, FPIR the\n"+
		"  share of searches for somebody not enrolled that name someone. The first row is the\n"+
		"  preset's threshold, the others are those of the next table.\n\n")
	rows = make([][]string, 0, 1+len(i.Thresholds))
	rows = append(rows, append(i.Current.cells(), "the preset's search threshold"))
	for _, s := range i.Thresholds {
		rows = append(rows, s.cells())
	}
	table(buf, []string{"threshold", "FNIR", "FPIR"}, rows...)

	fmt.Fprintf(buf, "\nSearch threshold by gallery size, worked out from the pairs of different people\n")
	rows = make([][]string, len(r.Galleries))
	for n, g := range r.Galleries {
		rows[n] = append([]string{strconv.Itoa(g.Gallery), exactPercent(g.FPIR), percent(g.FMR)}, g.cells(v.Impostor.Pairs)...)
	}
	table(buf, []string{"gallery", "FPIR target", "FMR needed", "threshold", "FNMR"}, rows...)
	fmt.Fprintf(buf, "  FNMR here estimates the share of searches for an enrolled person that will miss them.\n")

	fmt.Fprintf(buf, "\nWhat to run the server with\n")
	for _, paragraph := range r.Advice.paragraphs(r) {
		for _, line := range wrapped(paragraph, textWidth) {
			fmt.Fprintf(buf, "  %s\n", line)
		}
	}
	return buf.Bytes()
}

// cells are the threshold and the FNMR of a setting as a table shows them,
// or a note where the pictures do not support them.
func (s setting) cells(impostorPairs uint64) []string {
	if !s.Supported {
		return []string{"-", "-", fmt.Sprintf("not enough pairs: takes %d, there are %d", s.PairsNeeded, impostorPairs)}
	}
	return []string{fixed(*s.Threshold), percent(*s.FNMR)}
}

func (s searched) cells() []string {
	cells := []string{fixed(s.Threshold), "-", "-"}
	if s.FNIR != nil {
		cells[1] = percent(*s.FNIR)
	}
	if s.FPIR != nil {
		cells[2] = percent(*s.FPIR)
	}
	return cells
}

// paragraphs say in plain words which thresholds to use.
func (a advice) paragraphs(r *report) []string {
	var out []string
	pairs := r.Verification.Impostor.Pairs
	switch s := a.Search; {
	case s != nil:
		out = append(out, fmt.Sprintf("For a gallery of %s, use --search-threshold %s. About %s of searches for somebody "+
			"not enrolled will then name someone, and about %s of searches for somebody enrolled will miss them. "+
			"Other gallery sizes are in the table above.",
			people(s.Gallery), fixed(*s.Threshold), exactPercent(s.FPIR), percent(*s.FNMR)))
	case len(r.Galleries) > 0:
		smallest := r.Galleries[0]
		for _, g := range r.Galleries {
			if g.PairsNeeded < smallest.PairsNeeded {
				smallest = g
			}
		}
		out = append(out, fmt.Sprintf("These pictures are too few to say which --search-threshold a gallery of %s needs: "+
			"that takes %d pairs of different people, and there are %d. Evaluate more people.",
			people(smallest.Gallery), smallest.PairsNeeded, pairs))
	}
	switch s := a.Verify; {
	case s != nil:
		lowest := ""
		if s.FMR > advisedFMR {
			lowest = ", the lowest these pictures can measure,"
		}
		out = append(out, fmt.Sprintf("For verification, use --verify-threshold %s: an FMR of %s%s and an FNMR of %s.",
			fixed(*s.Threshold), exactPercent(s.FMR), lowest, percent(*s.FNMR)))
	default:
		out = append(out, fmt.Sprintf("These pictures are too few to advise a --verify-threshold: an FMR of %s takes "+
			"%d pairs of different people, and there are %d.",
			exactPercent(fmrTargets[0]), measure.PairsNeeded(fmrTargets[0]), pairs))
	}
	return append(out, "These figures hold for pictures like the ones evaluated. "+
		"Other cameras, other light and other people give other figures.")
}

// textWidth is where the lines of a paragraph are broken.
const textWidth = 88

// wrapped breaks a paragraph into lines of at most width characters.
func wrapped(paragraph string, width int) []string {
	var lines []string
	line := ""
	for _, word := range strings.Fields(paragraph) {
		if line != "" && len(line)+1+len(word) > width {
			lines, line = append(lines, line), ""
		}
		if line != "" {
			line += " "
		}
		line += word
	}
	return append(lines, line)
}

// fixed writes a threshold, or another similarity, with three decimals.
func fixed(similarity float64) string { return strconv.FormatFloat(similarity, 'f', 3, 64) }

// percent writes a rate as a percentage with enough digits to mean
// something: 12.3%, 1.23%, 0.012%.
func percent(rate float64) string {
	v := rate * 100
	decimals := 1
	switch {
	case v == 0:
		return "0%"
	case rate >= 1:
		return "100%"
	case v >= 10:
	case v >= 1:
		decimals = 2
	default:
		// two digits after the zeros
		decimals = 1 - int(math.Floor(math.Log10(v)))
	}
	return strconv.FormatFloat(v, 'f', decimals, 64) + "%"
}

// exactPercent writes a round rate, a target, without padding: 1%, 0.01%.
func exactPercent(rate float64) string {
	return strconv.FormatFloat(rate*100, 'g', 3, 64) + "%"
}

// probes and people write a number with what it counts.
func probes(n int) string {
	if n == 1 {
		return "1 probe"
	}
	return strconv.Itoa(n) + " probes"
}

func people(n int) string {
	if n == 1 {
		return "1 person"
	}
	return strconv.Itoa(n) + " people"
}

// table writes lines of cells in right-aligned columns, indented. A cell
// beyond the last column is a note and follows the line as it is.
func table(w io.Writer, header []string, rows ...[]string) {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	for _, row := range append([][]string{header}, rows...) {
		note := ""
		if len(row) > len(header) {
			row, note = row[:len(header)], "  "+row[len(header)]
		}
		_, _ = fmt.Fprintln(tw, strings.Join(row, "\t")+"\t"+note)
	}
	_ = tw.Flush()
}
