package eval

import (
	"bufio"
	"encoding/gob"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
)

// cacheMark is the first line of a cache file. It tells a cache from a
// file named by mistake, which must not be written over.
const cacheMark = "likeness eval cache 1\n"

// basis is what a cached outcome depends on besides the picture itself.
type basis struct {
	// Fingerprint identifies the embedder's weights.
	Fingerprint string
	Detector    string
	// Policy is the name of the quality policy with its limits spelt
	// out, so that a release that moves a limit does not reuse what an
	// earlier one decided.
	Policy string
}

// outcome is what analysing one picture came to: the vector of its best
// face, the reason it has no usable face, or the kind of error that kept
// it from being analysed.
type outcome struct {
	Vector  []float32
	Reject  string
	Failure string
}

// entry is the outcome of the picture that had this size and modification
// time, in nanoseconds, when it was analysed.
type entry struct {
	Size, ModTime int64
	Outcome       outcome
}

// cache keeps outcomes from one run to the next, by the name of the
// picture. Outcomes of pictures that a run does not look at stay, so a
// quick look at a few people does not throw away the vectors of the rest.
type cache struct {
	// file is empty when nothing is to be kept.
	file    string
	basis   basis
	entries map[string]entry
	changed bool
}

// openCache reads the cache in file, if there is one. A cache made with
// other models or another policy is of no use: it is left alone, with a
// note on stderr, until save replaces it.
func openCache(file string, made basis, stderr io.Writer) (*cache, error) {
	c := &cache{file: file, basis: made, entries: map[string]entry{}}
	if file == "" {
		return c, nil
	}
	f, err := os.Open(file)
	if errors.Is(err, fs.ErrNotExist) {
		return c, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := bufio.NewReader(f)
	mark := make([]byte, len(cacheMark))
	switch _, err := io.ReadFull(r, mark); {
	case errors.Is(err, io.EOF):
		return c, nil // an empty file, as mktemp leaves one
	case err != nil || string(mark) != cacheMark:
		return nil, fmt.Errorf("%s is not a cache of likeness eval; name another file for --cache, or remove this one", file)
	}
	var from basis
	entries := map[string]entry{}
	dec := gob.NewDecoder(r)
	if err := errors.Join(dec.Decode(&from), dec.Decode(&entries)); err != nil {
		return nil, fmt.Errorf("%s: the cache is damaged; remove it: %w", file, err)
	}

	switch {
	case from.Fingerprint != made.Fingerprint || from.Detector != made.Detector:
		note(stderr, "the cache %s was made with other models (%s, %s); it is not used and will be replaced", file, from.Detector, from.Fingerprint)
	case from.Policy != made.Policy:
		note(stderr, "the cache %s was made with another policy; it is not used and will be replaced", file)
	default:
		c.entries = entries
	}
	return c, nil
}

// lookup returns what the picture came to last time, if the file is still
// the one that was analysed then.
func (c *cache) lookup(p picture) (outcome, bool) {
	e, ok := c.entries[p.name]
	if !ok || e.Size != p.size || e.ModTime != p.modTime {
		return outcome{}, false
	}
	return e.Outcome, true
}

func (c *cache) keep(p picture, o outcome) {
	c.entries[p.name] = entry{Size: p.size, ModTime: p.modTime, Outcome: o}
	c.changed = true
}

// save writes the cache to its file if there is anything new in it. The
// file is written under another name and then moved into place, so that
// a run that is stopped half-way leaves the old cache or the new one.
func (c *cache) save() (err error) {
	if c.file == "" || !c.changed {
		return nil
	}
	tmp, err := os.CreateTemp(filepath.Dir(c.file), filepath.Base(c.file)+".*.tmp")
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			tmp.Close()
			os.Remove(tmp.Name())
		}
	}()
	w := bufio.NewWriter(tmp)
	w.WriteString(cacheMark)
	enc := gob.NewEncoder(w)
	if err := errors.Join(enc.Encode(c.basis), enc.Encode(c.entries), w.Flush(), tmp.Sync(), tmp.Close()); err != nil {
		return err
	}
	if err := os.Rename(tmp.Name(), c.file); err != nil {
		return err
	}
	c.changed = false
	return nil
}
