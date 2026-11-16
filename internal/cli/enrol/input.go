package enrol

import (
	"bufio"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/muhriddinsalohiddin/likeness/store"
)

// picture is one entry of the input: a file that is to become a template
// of a subject.
type picture struct {
	subject, path string
	// refused and reason are set when the entry cannot be used as it
	// stands; the file is then not looked at.
	refused outcome
	reason  string
}

// gather lists the pictures in the order they will be enrolled in, and
// what is wrong with the input, each remark with its place. Subject ids
// are checked here, before any time is spent on a picture.
func (o *options) gather() ([]picture, []string, error) {
	if o.list != "" {
		return fromList(o.list)
	}
	return fromDir(o.dir)
}

// fromDir reads a directory with one sub-directory per subject.
func fromDir(dir string) ([]picture, []string, error) {
	subjects, err := os.ReadDir(dir)
	if err != nil {
		return nil, nil, err
	}
	var (
		pictures []picture
		notes    []string
	)
	for _, s := range subjects {
		if hidden(s.Name()) || !isDir(dir, s) {
			continue
		}
		sub := filepath.Join(dir, s.Name())
		files, err := os.ReadDir(sub)
		if err != nil {
			return nil, nil, err
		}
		p := picture{subject: s.Name()}
		if err := store.ValidSubject(p.subject); err != nil {
			p.refused, p.reason = invalidSubject, whyInvalid(err)
			notes = append(notes, fmt.Sprintf("%s: the name cannot be a subject id: %s", sub, p.reason))
		}
		found := false
		for _, f := range files {
			if f.IsDir() || hidden(f.Name()) || !pictureName(f.Name()) {
				continue
			}
			p.path = filepath.Join(sub, f.Name())
			pictures = append(pictures, p)
			found = true
		}
		if !found && p.refused == "" {
			notes = append(notes, fmt.Sprintf("%s: no pictures", sub))
		}
	}
	return pictures, notes, nil
}

func hidden(name string) bool { return strings.HasPrefix(name, ".") }

// isDir follows a link, so that a subject's pictures may live elsewhere.
func isDir(parent string, e fs.DirEntry) bool {
	if e.Type()&fs.ModeSymlink == 0 {
		return e.IsDir()
	}
	st, err := os.Stat(filepath.Join(parent, e.Name()))
	return err == nil && st.IsDir()
}

func pictureName(name string) bool {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".jpg", ".jpeg", ".png", ".webp":
		return true
	}
	return false
}

// whyInvalid is what ValidSubject has against an id, without the preamble.
func whyInvalid(err error) string {
	return strings.TrimPrefix(err.Error(), store.ErrSubject.Error()+": ")
}

// fromList reads a CSV file of lines "subject,path". A line that cannot be
// used is kept as a refused entry, so that it is counted and reported with
// the rest.
func fromList(list string) ([]picture, []string, error) {
	f, err := os.Open(list)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	text := bufio.NewReader(f)
	// spreadsheets put a byte order mark in front, which would otherwise
	// become part of the first subject id
	if lead, err := text.Peek(3); err == nil && string(lead) == "\xef\xbb\xbf" {
		_, _ = text.Discard(3)
	}
	r := csv.NewReader(text)
	r.FieldsPerRecord = -1 // counted here, to carry on after a bad line
	r.TrimLeadingSpace = true

	var (
		pictures []picture
		notes    []string
	)
	base := filepath.Dir(list)
	for first := true; ; first = false {
		rec, err := r.Read()
		if errors.Is(err, io.EOF) {
			return pictures, notes, nil
		}
		var broken *csv.ParseError
		if errors.As(err, &broken) {
			p := picture{refused: invalidLine, reason: broken.Err.Error()}
			pictures = append(pictures, p)
			notes = append(notes, fmt.Sprintf("%s:%d: %s", list, broken.StartLine, p.reason))
			continue
		}
		if err != nil {
			return nil, nil, fmt.Errorf("%s: %w", list, err)
		}
		for i := range rec {
			rec[i] = strings.TrimSpace(rec[i])
		}
		if first && len(rec) == 2 && strings.EqualFold(rec[0], "subject") && strings.EqualFold(rec[1], "path") {
			continue // a header
		}

		p := picture{subject: rec[0]}
		if len(rec) > 1 {
			p.path = rec[1]
		}
		switch {
		case len(rec) != 2:
			p.refused, p.reason = invalidLine, "not of the form subject,path"
		case p.path == "":
			p.refused, p.reason = invalidLine, "no path"
		default:
			if err := store.ValidSubject(p.subject); err != nil {
				p.refused, p.reason = invalidSubject, whyInvalid(err)
			}
		}
		line, _ := r.FieldPos(0)
		switch {
		case p.refused == invalidSubject:
			notes = append(notes, fmt.Sprintf("%s:%d: %q cannot be a subject id: %s", list, line, p.subject, p.reason))
		case p.refused != "":
			notes = append(notes, fmt.Sprintf("%s:%d: %s", list, line, p.reason))
		case !filepath.IsAbs(p.path):
			p.path = filepath.Join(base, p.path)
		}
		pictures = append(pictures, p)
	}
}

// remark prints what is wrong with the input, a screenful at most.
func remark(w io.Writer, notes []string) {
	const most = 10
	for i, note := range notes {
		if i == most {
			say(w, "likeness enrol: and %d more of the kind\n", len(notes)-most)
			return
		}
		say(w, "likeness enrol: %s\n", note)
	}
}
