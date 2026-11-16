package enrol

import (
	"encoding/csv"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestEnrolList(t *testing.T) {
	engines := useFake(t)
	dir := tree(t,
		"photos/a1.jpg", "face:anna",
		"photos/a2.jpg", "face:anna#2",
		"b.png", "face:boris",
	)
	elsewhere := filepath.Join(tree(t, "c.dat", "face:carla"), "c.dat")
	list := filepath.Join(dir, "staff.csv")
	lines := []string{
		"\xef\xbb\xbfSubject, Path", // as a spreadsheet writes it
		"anna,photos/a1.jpg",
		"boris, b.png",
		"",
		`anna,"photos/a2.jpg"`,
		"carla," + elsewhere,
		"dora,photos/missing.jpg",
		"emil",
		"emil,one.jpg,two.jpg",
		"bad id,photos/a1.jpg",
		"fred,",
		`gina,pho"tos/a1.jpg`,
		"hana,photos/a1.jpg",
		`ivan,"photos/a1.jpg`,
	}
	if err := os.WriteFile(list, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	work := t.TempDir()
	path, report := filepath.Join(work, "faces.db"), filepath.Join(work, "report.csv")

	r := run(t, environment(), "--store", path, "--report", report, "--allow-duplicate", "--list", list)
	if r.err != nil {
		t.Fatalf("%v\n%s", r.err, r.stderr)
	}
	r.summary(t, "pictures read", "12 of 12", "enrolled", "5", "unreadable", "1", "invalid subject", "1", "invalid line", "5",
		"subjects in store", "0 before, 4 after")
	held(t, path, map[string]int{"anna": 2, "boris": 1, "carla": 1, "hana": 1})
	if n := engines.last().Calls(); n != 5 {
		t.Errorf("%d pictures were analysed, want 5: a line that cannot be used costs nothing", n)
	}

	want := [][]string{
		{"anna", filepath.Join(dir, "photos", "a1.jpg"), "enrolled", "", "", "", "1"},
		{"boris", filepath.Join(dir, "b.png"), "enrolled", "", "", "", "2"},
		{"anna", filepath.Join(dir, "photos", "a2.jpg"), "enrolled", "", "", "", "3"},
		{"carla", elsewhere, "enrolled", "", "", "", "4"},
		{"dora", filepath.Join(dir, "photos", "missing.jpg"), "unreadable", "no such file or directory", "", "", ""},
		{"emil", "", "invalid line", "not of the form subject,path", "", "", ""},
		{"emil", "one.jpg", "invalid line", "not of the form subject,path", "", "", ""},
		{"bad id", "photos/a1.jpg", "invalid subject", "contains ' '", "", "", ""},
		{"fred", "", "invalid line", "no path", "", "", ""},
		{"", "", "invalid line", csv.ErrBareQuote.Error(), "", "", ""},
		{"hana", filepath.Join(dir, "photos", "a1.jpg"), "enrolled", "", "", "", "5"},
		{"", "", "invalid line", csv.ErrQuote.Error(), "", "", ""},
	}
	got := readReport(t, report)[1:]
	if len(got) != len(want) {
		t.Fatalf("the report has %d lines, want %d: %q", len(got), len(want), got)
	}
	for i := range got {
		if !slices.Equal(got[i], want[i]) {
			t.Errorf("the report says %q, want %q", got[i], want[i])
		}
	}
	// each line that cannot be used is named with its place
	for _, note := range []string{
		list + ":8: not of the form subject,path\n",
		list + ":9: not of the form subject,path\n",
		list + `:10: "bad id" cannot be a subject id: contains ' '` + "\n",
		list + ":11: no path\n",
		list + ":12: " + csv.ErrBareQuote.Error() + "\n",
		list + ":14: " + csv.ErrQuote.Error() + "\n",
	} {
		if !strings.Contains(r.stderr, "likeness enrol: "+note) {
			t.Errorf("stderr lacks %q:\n%s", note, r.stderr)
		}
	}
	if n := strings.Count(r.stderr, "\n"); n != 6 {
		t.Errorf("stderr has %d lines, want one for each of the 6 bad ones:\n%s", n, r.stderr)
	}
}

func TestEnrolListPlaces(t *testing.T) {
	useFake(t)
	dir := tree(t, "photos/a.jpg", "face:anna", "staff.csv", "anna,photos/a.jpg\nsubject,path\n")
	work := t.TempDir()
	path, report := filepath.Join(work, "faces.db"), filepath.Join(work, "report.csv")

	// a list named from where it lies: paths stay as they were written, and
	// only a first line is a header
	t.Chdir(dir)
	r := run(t, environment(), "--store", path, "--report", report, "--list", "staff.csv")
	if r.err != nil {
		t.Fatal(r.err)
	}
	want := [][]string{
		{"anna", filepath.Join("photos", "a.jpg"), "enrolled", "", "", "", "1"},
		{"subject", "path", "unreadable", "no such file or directory", "", "", ""},
	}
	if got := readReport(t, report)[1:]; len(got) != 2 || !slices.Equal(got[0], want[0]) || !slices.Equal(got[1], want[1]) {
		t.Errorf("the report says %q, want %q", got, want)
	}

	for name, text := range map[string]string{"an empty list": "", "a header alone": "subject,path\n"} {
		if err := os.WriteFile("empty.csv", []byte(text), 0o644); err != nil {
			t.Fatal(err)
		}
		r := run(t, environment(), "--store", path, "--list", "empty.csv")
		if r.err == nil || !strings.Contains(r.err.Error(), "empty.csv: no lines of the form subject,path") {
			t.Errorf("%s: %v", name, r.err)
		}
	}
	if r := run(t, environment(), "--store", path, "--list", "missing.csv"); !errors.Is(r.err, os.ErrNotExist) {
		t.Errorf("a list that is not there: %v", r.err)
	}
	if r := run(t, environment(), "--store", path, "--list", "photos"); r.err == nil {
		t.Error("a directory was read as a list")
	}
}

func TestFromDir(t *testing.T) {
	dir := tree(t,
		"b/2.JPEG", "",
		"b/10.jpg", "",
		"b/1.WebP", "",
		"b/1.gif", "",
		"b/jpg", "",
		"b/.2.jpg", "",
		"b/deeper/3.jpg", "",
		"a/x.png", "",
		".git/x.png", "",
		"c.jpg", "",
	)
	pictures := tree(t, "1.jpg", "")
	if err := os.Symlink(pictures, filepath.Join(dir, "linked")); err != nil {
		t.Skipf("no links here: %v", err)
	}
	if err := os.Symlink(filepath.Join(pictures, "1.jpg"), filepath.Join(dir, "file.jpg")); err != nil {
		t.Fatal(err)
	}

	got, notes, err := fromDir(dir)
	if err != nil || len(notes) != 0 {
		t.Fatalf("%v %q", err, notes)
	}
	// subjects in the order of their names, and so the pictures of each
	want := []picture{
		{subject: "a", path: filepath.Join(dir, "a", "x.png")},
		{subject: "b", path: filepath.Join(dir, "b", "1.WebP")},
		{subject: "b", path: filepath.Join(dir, "b", "10.jpg")},
		{subject: "b", path: filepath.Join(dir, "b", "2.JPEG")},
		{subject: "linked", path: filepath.Join(dir, "linked", "1.jpg")},
	}
	if !slices.Equal(got, want) {
		t.Errorf("the pictures found are\n%+v, want\n%+v", got, want)
	}
}
