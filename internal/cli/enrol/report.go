package enrol

import (
	"encoding/csv"
	"os"
	"strconv"
)

// reportHeader names the columns of the report.
var reportHeader = []string{"subject", "path", "outcome", "reason", "score", "other_subject", "template"}

// report is the CSV file that says what became of each picture. A nil
// report takes its lines and keeps nothing.
type report struct {
	file *os.File
	csv  *csv.Writer
}

// createReport starts the report at path; no path, no report.
func createReport(path string) (*report, error) {
	if path == "" {
		return nil, nil
	}
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	r := &report{file: f, csv: csv.NewWriter(f)}
	r.write(reportHeader)
	return r, nil
}

// write adds a line. The writer keeps its first error for close.
func (r *report) write(line []string) { _ = r.csv.Write(line) }

func (r *report) add(p *picture, v verdict) {
	if r == nil {
		return
	}
	line := []string{p.subject, p.path, string(v.outcome), v.reason, "", v.other, ""}
	if v.outcome == duplicate || v.outcome == mismatch {
		line[4] = strconv.FormatFloat(float64(v.score), 'f', 4, 32)
	}
	if v.template != 0 {
		line[6] = strconv.FormatUint(v.template, 10)
	}
	r.write(line)
}

// close writes out what is pending and reports whether all of the report
// reached the file.
func (r *report) close() error {
	if r == nil {
		return nil
	}
	r.csv.Flush()
	err := r.csv.Error()
	if failed := r.file.Close(); err == nil {
		err = failed
	}
	return err
}
