package eval

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	measure "github.com/muhriddinsalohiddin/likeness/internal/eval"
)

// picture is one file of the dataset.
type picture struct {
	// name is the path below the dataset directory, written with slashes:
	// what the cache and the messages know the picture by.
	name string
	path string
	// size and modTime, in nanoseconds, tell whether the file is still
	// the one a cached vector was made from.
	size    int64
	modTime int64
}

type person struct {
	name     string
	pictures []picture
}

// dataset is the people found in a directory, in the order of their names,
// each with their pictures in the order of theirs.
type dataset struct {
	dir    string
	people []person
}

func hidden(name string) bool { return strings.HasPrefix(name, ".") }

// isPicture goes by the name: the formats the engine reads.
func isPicture(name string) bool {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".jpg", ".jpeg", ".png", ".webp":
		return true
	}
	return false
}

// resolve says what a directory entry is, following a symbolic link.
func resolve(dir string, e fs.DirEntry) (fs.FileInfo, error) {
	if e.Type()&fs.ModeSymlink != 0 {
		return os.Stat(filepath.Join(dir, e.Name()))
	}
	return e.Info()
}

// scan lists the pictures below dir: every directory in it is a person and
// every picture below that, however deep, is theirs. maxPeople and
// maxPerPerson, if not 0, cut the list short.
func scan(dir string, maxPeople, maxPerPerson int) (*dataset, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	set := &dataset{dir: dir}
	for _, e := range entries {
		if maxPeople > 0 && len(set.people) == maxPeople {
			break
		}
		if hidden(e.Name()) {
			continue
		}
		// A person may be a link to a directory kept elsewhere.
		if at, err := resolve(dir, e); err != nil || !at.IsDir() {
			continue
		}
		p := person{name: e.Name()}
		if err := p.collect(filepath.Join(dir, e.Name()), e.Name()); err != nil {
			return nil, err
		}
		sort.Slice(p.pictures, func(a, b int) bool { return p.pictures[a].name < p.pictures[b].name })
		if maxPerPerson > 0 && len(p.pictures) > maxPerPerson {
			p.pictures = p.pictures[:maxPerPerson]
		}
		if len(p.pictures) > 0 {
			set.people = append(set.people, p)
		}
	}

	layout := "it has to hold one directory per person with that person's pictures in it, " +
		filepath.Join(dir, "<person>", "<picture>.jpg")
	switch len(set.people) {
	case 0:
		return nil, fmt.Errorf("no pictures in %s: %s", dir, layout)
	case 1:
		return nil, fmt.Errorf("%s has pictures of one person only, %s, and it takes two to tell people apart: %s",
			dir, set.people[0].name, layout)
	}
	return set, nil
}

// collect adds the pictures in dir and below it. name is dir as seen from
// the dataset directory. Links to pictures are followed, links to further
// directories are not: they could lead round in a circle.
func (p *person) collect(dir, name string) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if hidden(e.Name()) {
			continue
		}
		path, below := filepath.Join(dir, e.Name()), name+"/"+e.Name()
		if e.IsDir() {
			if err := p.collect(path, below); err != nil {
				return err
			}
			continue
		}
		if !isPicture(e.Name()) {
			continue
		}
		at, err := resolve(dir, e)
		if err != nil || !at.Mode().IsRegular() {
			continue
		}
		p.pictures = append(p.pictures, picture{name: below, path: path, size: at.Size(), modTime: at.ModTime().UnixNano()})
	}
	return nil
}

// pictures lists every picture of the dataset, person after person.
func (d *dataset) pictures() []picture {
	var all []picture
	for _, p := range d.people {
		all = append(all, p.pictures...)
	}
	return all
}

// counts says what the dataset came to once its pictures were analysed.
type counts struct {
	dir              string
	people, pictures int
	// usable is the number of pictures that gave a vector, usablePeople
	// the number of people with at least one of them, and repeated the
	// number of people with two or more.
	usable, usablePeople, repeated int
	// turnedDown counts the pictures without a usable face by reason,
	// unreadable the ones that could not be analysed by kind of error.
	turnedDown, unreadable map[string]int
}

// gather sorts the outcomes, which are in the order of pictures(), back
// into people. People without a usable picture are left out.
func (d *dataset) gather(outcomes []outcome) ([]measure.Person, *counts) {
	c := &counts{dir: d.dir, people: len(d.people), turnedDown: map[string]int{}, unreadable: map[string]int{}}
	var people []measure.Person
	for _, p := range d.people {
		var vectors measure.Person
		for _, o := range outcomes[c.pictures : c.pictures+len(p.pictures)] {
			switch {
			case o.Failure != "":
				c.unreadable[o.Failure]++
			case o.Reject != "":
				c.turnedDown[o.Reject]++
			default:
				vectors = append(vectors, o.Vector)
			}
		}
		c.pictures += len(p.pictures)
		c.usable += len(vectors)
		if len(vectors) > 0 {
			c.usablePeople++
			people = append(people, vectors)
		}
		if len(vectors) > 1 {
			c.repeated++
		}
	}
	return people, c
}

// enough reports, as an error that says what to do, a set of usable
// pictures that cannot be evaluated: there has to be somebody to tell from
// somebody else, and two pictures of somebody to recognise them by.
func (c *counts) enough(policy string) error {
	lost := ""
	if n := c.pictures - c.usable; n > 0 {
		lost = fmt.Sprintf(" (%s)", strings.Join(c.losses(), "; "))
		if policy != "none" {
			lost += "; --policy none takes every face the detector finds"
		}
	}
	switch {
	case c.usablePeople < 2:
		return fmt.Errorf("%d of the %d pictures in %s can be used, of %d of the %d people, and it takes two people to tell apart%s",
			c.usable, c.pictures, c.dir, c.usablePeople, c.people, lost)
	case c.repeated == 0:
		return fmt.Errorf("nobody in %s has two usable pictures, so there is no pair of the same person to measure: "+
			"put several pictures of at least some people in their directories%s", c.dir, lost)
	}
	return nil
}

// losses describes the pictures that could not be used.
func (c *counts) losses() []string {
	var parts []string
	if len(c.turnedDown) > 0 {
		parts = append(parts, "turned down: "+tallied(c.turnedDown))
	}
	if len(c.unreadable) > 0 {
		parts = append(parts, "unreadable: "+tallied(c.unreadable))
	}
	return parts
}

// tallied writes counts out as "blurry 3, no_face 1", the largest first.
func tallied(counts map[string]int) string {
	names := make([]string, 0, len(counts))
	for name := range counts {
		names = append(names, name)
	}
	sort.Slice(names, func(a, b int) bool {
		if counts[names[a]] != counts[names[b]] {
			return counts[names[a]] > counts[names[b]]
		}
		return names[a] < names[b]
	})
	for i, name := range names {
		names[i] = fmt.Sprintf("%s %d", name, counts[name])
	}
	return strings.Join(names, ", ")
}
