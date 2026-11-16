package eval

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// listing writes a dataset out as "person: picture picture; person: ...".
func listing(set *dataset) string {
	people := make([]string, 0, len(set.people))
	for _, p := range set.people {
		names := make([]string, 0, len(p.pictures))
		for _, pic := range p.pictures {
			names = append(names, strings.TrimPrefix(pic.name, p.name+"/"))
		}
		people = append(people, p.name+": "+strings.Join(names, " "))
	}
	return strings.Join(people, "; ")
}

func TestScan(t *testing.T) {
	dir := t.TempDir()
	for _, file := range []string{
		// in the order of their names, whatever order they are made in
		"boris/b.png", "boris/a.jpg", "boris/C.JPG", "boris/d.Jpeg", "boris/e.webp",
		"anna/2.jpg", "anna/10.jpg", "anna/1.jpg",
		// deeper directories belong to the person at the top
		"clara/2019/summer/beach.jpg", "clara/2019/a.jpg", "clara/portrait.jpg", "clara/2018.jpg",
		// by the whole name: a picture comes before a directory that its
		// name begins with
		"clara/2019.jpg",
		// not pictures
		"anna/notes.txt", "anna/jpg", "anna/picture.jpg.bak", "boris/Thumbs.db", "boris/f.gif", "boris/g.bmp",
		// hidden
		"anna/.hidden.jpg", "boris/.cache/thumb.jpg", ".git/objects/pack.jpg", ".DS_Store",
		// not in a person's directory
		"stray.jpg", "README.md",
		// people with nothing to show
		"dmitri/notes.txt", "elena/.hidden.jpg",
	} {
		write(t, filepath.Join(dir, filepath.FromSlash(file)), "face:x")
	}
	if err := os.Mkdir(filepath.Join(dir, "fyodor"), 0o755); err != nil {
		t.Fatal(err)
	}
	// a directory that is named like a picture is a directory
	write(t, filepath.Join(dir, "anna", "album.jpg", "3.jpg"), "face:x")

	set, err := scan(dir, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := "anna: 1.jpg 10.jpg 2.jpg album.jpg/3.jpg; " +
		"boris: C.JPG a.jpg b.png d.Jpeg e.webp; " +
		"clara: 2018.jpg 2019.jpg 2019/a.jpg 2019/summer/beach.jpg portrait.jpg"
	if got := listing(set); got != want {
		t.Errorf("the dataset:\n %s\nwant:\n %s", got, want)
	}
	if set.dir != dir {
		t.Errorf("the dataset is at %s, want %s", set.dir, dir)
	}

	pictures := set.pictures()
	if len(pictures) != 14 {
		t.Fatalf("%d pictures, want 14", len(pictures))
	}
	deep := pictures[12]
	at, err := os.Stat(filepath.Join(dir, "clara", "2019", "summer", "beach.jpg"))
	if err != nil {
		t.Fatal(err)
	}
	if deep.name != "clara/2019/summer/beach.jpg" || deep.path != filepath.Join(dir, "clara", "2019", "summer", "beach.jpg") ||
		deep.size != int64(len("face:x")) || deep.modTime != at.ModTime().UnixNano() {
		t.Errorf("a picture in a deeper directory: %+v", deep)
	}

	for _, c := range []struct {
		maxPeople, maxPerPerson int
		want                    string
	}{
		{2, 0, "anna: 1.jpg 10.jpg 2.jpg album.jpg/3.jpg; boris: C.JPG a.jpg b.png d.Jpeg e.webp"},
		{0, 2, "anna: 1.jpg 10.jpg; boris: C.JPG a.jpg; clara: 2018.jpg 2019.jpg"},
		{2, 1, "anna: 1.jpg; boris: C.JPG"},
		{3, 4, "anna: 1.jpg 10.jpg 2.jpg album.jpg/3.jpg; boris: C.JPG a.jpg b.png d.Jpeg; clara: 2018.jpg 2019.jpg 2019/a.jpg 2019/summer/beach.jpg"},
		{9, 9, want},
	} {
		set, err := scan(dir, c.maxPeople, c.maxPerPerson)
		if err != nil {
			t.Fatal(err)
		}
		if got := listing(set); got != c.want {
			t.Errorf("the first %d people and %d pictures:\n %s\nwant:\n %s", c.maxPeople, c.maxPerPerson, got, c.want)
		}
	}
}

func TestScanFollowsLinks(t *testing.T) {
	elsewhere := t.TempDir()
	write(t, filepath.Join(elsewhere, "anna", "1.jpg"), "face:anna#1")
	write(t, filepath.Join(elsewhere, "anna", "2.jpg"), "face:anna#2")
	write(t, filepath.Join(elsewhere, "loose.png"), "face:boris#2 from elsewhere")

	dir := t.TempDir()
	write(t, filepath.Join(dir, "boris", "1.jpg"), "face:boris#1")
	for link, target := range map[string]string{
		// a person kept elsewhere, and a picture kept elsewhere
		filepath.Join(dir, "anna"):           filepath.Join(elsewhere, "anna"),
		filepath.Join(dir, "boris", "2.png"): filepath.Join(elsewhere, "loose.png"),
		// links that lead nowhere, round in a circle, or to a directory
		// below a person: none of them is followed
		filepath.Join(dir, "clara"):             filepath.Join(elsewhere, "nobody"),
		filepath.Join(dir, "boris", "gone.jpg"): filepath.Join(elsewhere, "gone.jpg"),
		filepath.Join(dir, "boris", "round"):    dir,
		filepath.Join(dir, "boris", "dir.jpg"):  filepath.Join(elsewhere, "anna"),
	} {
		if err := os.Symlink(target, link); err != nil {
			t.Skipf("no symbolic links here: %v", err)
		}
	}

	set, err := scan(dir, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := listing(set), "anna: 1.jpg 2.jpg; boris: 1.jpg 2.png"; got != want {
		t.Errorf("the dataset:\n %s\nwant:\n %s", got, want)
	}
	// the size is that of the picture, not of the link
	if linked := set.people[1].pictures[1]; linked.size != int64(len("face:boris#2 from elsewhere")) {
		t.Errorf("a linked picture of %d bytes", linked.size)
	}
}

func TestScanFails(t *testing.T) {
	if os.Getuid() == 0 {
		t.Skip("root reads what nobody may read")
	}
	dir := t.TempDir()
	write(t, filepath.Join(dir, "anna", "1.jpg"), "face:anna")
	write(t, filepath.Join(dir, "boris", "private", "1.jpg"), "face:boris")
	locked := filepath.Join(dir, "boris", "private")
	if err := os.Chmod(locked, 0); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(locked, 0o755) })

	// a person whose pictures cannot be listed is not quietly left out
	if _, err := scan(dir, 0, 0); !errors.Is(err, fs.ErrPermission) || !strings.Contains(err.Error(), "private") {
		t.Errorf("a directory that may not be read: %v", err)
	}
}

func TestGather(t *testing.T) {
	vector := func(x float32) outcome { return outcome{Vector: []float32{x}} }
	set := &dataset{dir: "faces", people: []person{
		{name: "anna", pictures: make([]picture, 3)},
		{name: "boris", pictures: make([]picture, 2)},
		{name: "clara", pictures: make([]picture, 4)},
		{name: "dmitri", pictures: make([]picture, 1)},
	}}
	people, c := set.gather([]outcome{
		vector(1), {Reject: "blurry"}, vector(2),
		{Reject: "blurry"}, {Failure: "corrupt_image"},
		{Reject: "no_face"}, vector(3), {Reject: "blurry"}, {Failure: "corrupt_image"},
		vector(4),
	})
	if c.dir != "faces" || c.people != 4 || c.pictures != 10 || c.usable != 4 || c.usablePeople != 3 || c.repeated != 1 {
		t.Errorf("counts: %+v", c)
	}
	if tallied(c.turnedDown) != "blurry 3, no_face 1" || tallied(c.unreadable) != "corrupt_image 2" {
		t.Errorf("turned down: %v; unreadable: %v", c.turnedDown, c.unreadable)
	}
	// boris has no usable picture and is not among the people measured
	if len(people) != 3 || len(people[0]) != 2 || people[0][1][0] != 2 || len(people[1]) != 1 || people[1][0][0] != 3 || people[2][0][0] != 4 {
		t.Errorf("the people: %v", people)
	}
	if err := c.enough("search"); err != nil {
		t.Errorf("three people, one of them with two pictures, are not enough: %v", err)
	}

	if got := tallied(map[string]int{"b": 2, "a": 2, "c": 7, "d": 1}); got != "c 7, a 2, b 2, d 1" {
		t.Errorf("tallied: %s", got)
	}
	if got := tallied(nil); got != "" {
		t.Errorf("nothing tallied: %q", got)
	}
}

func TestIsPicture(t *testing.T) {
	for name, want := range map[string]bool{
		"a.jpg": true, "a.JPG": true, "a.jpeg": true, "a.JpEg": true, "a.png": true, "a.PNG": true, "a.webp": true, "a.WEBP": true,
		"a.b.jpg": true, "a b.jpg": true,
		"a.gif": false, "a.bmp": false, "a.tiff": false, "a.jpg.txt": false, "jpg": false, "a": false, "a.": false, "": false,
	} {
		if got := isPicture(name); got != want {
			t.Errorf("%q is a picture: %v, want %v", name, got, want)
		}
	}
}
