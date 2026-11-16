package bench

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"
)

// millis is a duration that is written in milliseconds: in JSON as a
// number, with as many decimals as it needs.
type millis time.Duration

func (m millis) String() string {
	return strconv.FormatFloat(float64(m)/float64(time.Millisecond), 'f', -1, 64)
}

func (m millis) MarshalJSON() ([]byte, error) { return []byte(m.String()), nil }

// fixed writes the duration in the given unit with a fixed number of
// decimals, for a column of a table.
func (m millis) fixed(unit time.Duration, decimals int) string {
	return strconv.FormatFloat(float64(m)/float64(unit), 'f', decimals, 64)
}

// spread summarises how long one thing took over many repetitions.
type spread struct {
	P50 millis `json:"p50_ms"`
	P90 millis `json:"p90_ms"`
	P99 millis `json:"p99_ms"`
	Max millis `json:"max_ms"`
}

// stageTimes is the first measurement: one picture at a time. The stages
// are the engine's own account of a call, Total the time the call took as
// its caller saw it.
type stageTimes struct {
	Pictures int    `json:"pictures"`
	Decode   spread `json:"decode"`
	Detect   spread `json:"detect"`
	Align    spread `json:"align"`
	Embed    spread `json:"embed"`
	Total    spread `json:"total"`
}

// searchTimes is the second measurement, for one gallery.
type searchTimes struct {
	Gallery  int    `json:"gallery"`
	Searches int    `json:"searches"`
	Kernel   string `json:"kernel"`
	P50      millis `json:"p50_ms"`
	P99      millis `json:"p99_ms"`
}

// levelTimes is the third measurement, for one number of pictures at once.
// The times are those of a whole picture: analysis and search.
type levelTimes struct {
	Inflight  int     `json:"inflight"`
	Gallery   int     `json:"gallery"`
	Pictures  int     `json:"pictures"`
	PerSecond float64 `json:"pictures_per_second"`
	P50       millis  `json:"p50_ms"`
	P99       millis  `json:"p99_ms"`
}

// recommendation is the level to run the server at. MaxInflight is zero
// when no level kept the target.
type recommendation struct {
	Target      millis  `json:"target_ms"`
	MaxInflight int     `json:"max_inflight"`
	PerSecond   float64 `json:"pictures_per_second"`
	P99         millis  `json:"p99_ms"`
}

type engineInfo struct {
	Preset   string `json:"preset"`
	Detector string `json:"detector"`
	Embedder string `json:"embedder"`
	Dim      int    `json:"dim"`
	Runtime  string `json:"runtime"`
	Threads  int    `json:"threads"`
}

type machineInfo struct {
	OS   string `json:"os"`
	Arch string `json:"arch"`
	CPUs int    `json:"cpus"`
	Go   string `json:"go"`
}

type pictureInfo struct {
	Name   string `json:"name"`
	Width  int    `json:"width"`
	Height int    `json:"height"`
	Bytes  int    `json:"bytes"`
}

// report is everything one run measured. The field names in its JSON form
// are what scripts read; they stay as they are.
type report struct {
	Version        string         `json:"version"`
	Engine         engineInfo     `json:"engine"`
	Machine        machineInfo    `json:"machine"`
	Picture        pictureInfo    `json:"picture"`
	Duration       millis         `json:"duration_ms"`
	Stages         stageTimes     `json:"stages"`
	Search         []searchTimes  `json:"search"`
	Inflight       []levelTimes   `json:"inflight"`
	Recommendation recommendation `json:"recommendation"`
}

// write prints the report, as tables or as one JSON document. It goes out
// in one piece, so a failure does not leave half a report behind.
func (r *report) write(w io.Writer, asJSON bool) error {
	var buf bytes.Buffer
	if asJSON {
		enc := json.NewEncoder(&buf)
		enc.SetIndent("", "  ")
		if err := enc.Encode(r); err != nil {
			return err
		}
	} else {
		r.text(&buf)
	}
	_, err := w.Write(buf.Bytes())
	return err
}

// text writes the report for a person to read.
func (r *report) text(buf *bytes.Buffer) {
	e, m, p := r.Engine, r.Machine, r.Picture
	period := time.Duration(r.Duration)
	fmt.Fprintf(buf, "likeness %s\n", r.Version)
	fmt.Fprintf(buf, "  models   %s: %s and %s, vectors of %d values\n", e.Preset, e.Detector, e.Embedder, e.Dim)
	fmt.Fprintf(buf, "  runtime  ONNX Runtime %s, %s\n", e.Runtime, count(e.Threads, "thread"))
	fmt.Fprintf(buf, "  machine  %s/%s, %s, %s\n", m.OS, m.Arch, count(m.CPUs, "processor"), m.Go)
	fmt.Fprintf(buf, "  picture  %s, %dx%d, %d bytes\n", p.Name, p.Width, p.Height, p.Bytes)

	fmt.Fprintf(buf, "\nOne picture at a time: %s in %s\n", count(r.Stages.Pictures, "picture"), period)
	stage := func(name string, s spread) []string {
		return []string{name, inMillis(s.P50), inMillis(s.P90), inMillis(s.P99), inMillis(s.Max)}
	}
	table(buf, []string{"stage", "p50 ms", "p90 ms", "p99 ms", "max ms"},
		stage("decode", r.Stages.Decode), stage("detect", r.Stages.Detect),
		stage("align", r.Stages.Align), stage("embed", r.Stages.Embed),
		stage("total", r.Stages.Total))

	fmt.Fprintf(buf, "\nSearch: the face of the picture among random ones, one search at a time\n")
	searches := make([][]string, len(r.Search))
	for i, s := range r.Search {
		searches[i] = []string{
			strconv.Itoa(s.Gallery), strconv.Itoa(s.Searches),
			s.P50.fixed(time.Microsecond, 1), s.P99.fixed(time.Microsecond, 1), s.Kernel,
		}
	}
	table(buf, []string{"gallery", "searches", "p50 µs", "p99 µs", "kernel"}, searches...)

	fmt.Fprintf(buf, "\nSeveral at once: each picture analysed and searched for, %s a level\n", period)
	levels := make([][]string, len(r.Inflight))
	for i, l := range r.Inflight {
		levels[i] = []string{
			strconv.Itoa(l.Inflight), strconv.Itoa(l.Gallery), strconv.Itoa(l.Pictures),
			strconv.FormatFloat(l.PerSecond, 'f', 1, 64), inMillis(l.P50), inMillis(l.P99),
		}
	}
	table(buf, []string{"at once", "gallery", "pictures", "pictures/s", "p50 ms", "p99 ms"}, levels...)

	rec := r.Recommendation
	if rec.MaxInflight == 0 {
		fmt.Fprintf(buf, "\nRecommendation: none. No level keeps the 99th percentile within %s ms on this machine", rec.Target)
		if best, ok := quickest(r.Inflight); ok {
			fmt.Fprintf(buf, ";\n  the lowest is %s ms, with %d at once", inMillis(best.P99), best.Inflight)
		}
		fmt.Fprintf(buf, ".\n")
		return
	}
	fmt.Fprintf(buf, "\nRecommendation: run the server with --max-inflight %d\n", rec.MaxInflight)
	fmt.Fprintf(buf, "  %d at once: %.1f pictures per second, 99th percentile %s ms, within the %s ms asked for.\n",
		rec.MaxInflight, rec.PerSecond, inMillis(rec.P99), rec.Target)
}

// quickest returns the level with the lowest 99th percentile.
func quickest(levels []levelTimes) (best levelTimes, ok bool) {
	for i, l := range levels {
		if i == 0 || l.P99 < best.P99 {
			best, ok = l, true
		}
	}
	return best, ok
}

func inMillis(m millis) string { return m.fixed(time.Millisecond, 2) }

// count writes a number with what it counts: 1 thread, 8 threads.
func count(n int, thing string) string {
	if n == 1 {
		return "1 " + thing
	}
	return strconv.Itoa(n) + " " + thing + "s"
}

// table writes lines of cells in right-aligned columns, indented.
func table(w io.Writer, header []string, rows ...[]string) {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	for _, row := range append([][]string{header}, rows...) {
		_, _ = fmt.Fprintln(tw, strings.Join(row, "\t")+"\t")
	}
	_ = tw.Flush()
}
