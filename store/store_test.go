package store

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	mrand "math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	bolt "go.etcd.io/bbolt"
)

const (
	testDim         = 16
	testFingerprint = "sface:test:0123456789abcdef"
)

var ctx = context.Background()

func testOptions() Options { return Options{Dim: testDim, Fingerprint: testFingerprint} }

func open(t testing.TB, path string, o Options) Store {
	t.Helper()
	s, err := Open(path, o)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func newStore(t testing.TB, o Options) (Store, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "faces.db")
	return open(t, path, o), path
}

func vector(rng *mrand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}

func key(b byte) []byte { return bytes.Repeat([]byte{b}, KeySize) }

// contents reads everything back through Load.
func contents(t testing.TB, s Store) map[uint64]Template {
	t.Helper()
	all := map[uint64]Template{}
	var last string
	err := s.Load(ctx, func(tm Template) error {
		if tm.Subject < last {
			t.Errorf("Load is not grouped by subject: %q after %q", tm.Subject, last)
		}
		last = tm.Subject
		tm.Vector = append([]float32(nil), tm.Vector...)
		tm.Meta = bytes.Clone(tm.Meta)
		all[tm.ID] = tm
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return all
}

func sameVector(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

func TestRoundTrip(t *testing.T) {
	for name, o := range map[string]Options{"plain": testOptions(), "encrypted": {Dim: testDim, Fingerprint: testFingerprint, Key: key(7)}} {
		t.Run(name, func(t *testing.T) {
			rng := mrand.New(mrand.NewSource(1))
			s, path := newStore(t, o)
			before := time.Now().Add(-time.Second)
			info := s.Info()
			if info.Dim != testDim || info.Fingerprint != testFingerprint || info.Encrypted != (o.Key != nil) || info.Created.Before(before) {
				t.Fatalf("%+v", info)
			}

			want := map[uint64]Template{}
			var lastID uint64
			for i := 0; i < 40; i++ {
				subject := fmt.Sprintf("emp-%02d", i%15)
				v := vector(rng, testDim)
				v[0] = float32(math.Copysign(0, -1)) // survives only an exact round trip
				meta := []byte(fmt.Sprintf(`{"n":%d}`, i))
				if i%7 == 0 {
					meta = nil
				}
				tm, err := s.Add(ctx, subject, v, meta)
				if err != nil {
					t.Fatal(err)
				}
				if tm.ID <= lastID || tm.Subject != subject || tm.Created.Before(before) {
					t.Fatalf("template %+v after id %d", tm, lastID)
				}
				lastID = tm.ID
				want[tm.ID] = Template{ID: tm.ID, Subject: subject, Created: tm.Created, Vector: v, Meta: meta}
			}

			check := func(s Store) {
				t.Helper()
				got := contents(t, s)
				if len(got) != len(want) {
					t.Fatalf("%d templates, want %d", len(got), len(want))
				}
				for id, w := range want {
					g := got[id]
					if g.Subject != w.Subject || !sameVector(g.Vector, w.Vector) || !bytes.Equal(g.Meta, w.Meta) || !g.Created.Equal(w.Created) {
						t.Fatalf("template %d came back as %+v, want %+v", id, g, w)
					}
				}
				if subjects, templates, err := s.Count(ctx); err != nil || subjects != 15 || templates != 40 {
					t.Fatalf("count %d/%d %v", subjects, templates, err)
				}
				sub, err := s.Subject(ctx, "emp-03")
				if err != nil || sub.ID != "emp-03" || len(sub.Templates) != 3 || sub.Created.Before(before) {
					t.Fatalf("%+v %v", sub, err)
				}
				for i, ti := range sub.Templates {
					w := want[ti.ID]
					if w.Subject != "emp-03" || !bytes.Equal(ti.Meta, w.Meta) || !ti.Created.Equal(w.Created) {
						t.Fatalf("template info %+v, want %+v", ti, w)
					}
					if i > 0 && ti.ID <= sub.Templates[i-1].ID {
						t.Fatal("templates are not oldest first")
					}
				}
				if !sub.Created.Equal(sub.Templates[0].Created) {
					t.Fatalf("subject created %v, its first template %v", sub.Created, sub.Templates[0].Created)
				}
			}
			check(s)

			// and again from disk
			if err := s.Close(); err != nil {
				t.Fatal(err)
			}
			again := open(t, path, o)
			check(again)
			if got := again.Info(); !got.Created.Equal(info.Created) || got.Dim != info.Dim || got.Fingerprint != info.Fingerprint {
				t.Fatalf("info changed across a restart: %+v, was %+v", got, info)
			}
			tm, err := again.Add(ctx, "emp-99", vector(rng, testDim), nil)
			if err != nil || tm.ID <= lastID {
				t.Fatalf("ids must keep growing after a restart: %d after %d (%v)", tm.ID, lastID, err)
			}

			if runtime.GOOS != "windows" {
				if st, err := os.Stat(path); err != nil || st.Mode().Perm() != 0o600 {
					t.Fatalf("store file mode %v, want 0600 (%v)", st.Mode().Perm(), err)
				}
			}
		})
	}
}

func TestModelMismatch(t *testing.T) {
	s, path := newStore(t, testOptions())
	if _, err := s.Add(ctx, "a", vector(mrand.New(mrand.NewSource(2)), testDim), nil); err != nil {
		t.Fatal(err)
	}
	s.Close()

	for name, o := range map[string]Options{
		"another model":  {Dim: testDim, Fingerprint: "arcface:w600k_r50:fedcba9876543210"},
		"another length": {Dim: 512, Fingerprint: testFingerprint},
	} {
		_, err := Open(path, o)
		if !errors.Is(err, ErrMismatch) || !strings.Contains(err.Error(), testFingerprint) || !strings.Contains(err.Error(), o.Fingerprint) {
			t.Errorf("%s: %v", name, err)
		}
	}
	// a tool that takes the store as it is
	tool := open(t, path, Options{})
	if info := tool.Info(); info.Dim != testDim || info.Fingerprint != testFingerprint {
		t.Fatalf("%+v", info)
	}
	if n, _, _ := tool.Count(ctx); n != 1 {
		t.Fatalf("%d subjects", n)
	}
}

func TestNewStoreNeedsAModel(t *testing.T) {
	dir := t.TempDir()
	for i, o := range []Options{{}, {Dim: 128}, {Fingerprint: "x"}, {Dim: -1, Fingerprint: "x"}} {
		if _, err := Open(filepath.Join(dir, fmt.Sprintf("%d.db", i)), o); err == nil {
			t.Errorf("Open(%+v) made a store without knowing its model", o)
		}
	}
	if _, err := Open(filepath.Join(dir, "missing", "faces.db"), testOptions()); err == nil {
		t.Error("a store in a directory that does not exist")
	}
}

func TestEncryption(t *testing.T) {
	rng := mrand.New(mrand.NewSource(3))
	secret := vector(rng, testDim)
	raw := make([]byte, 0, 4*testDim)
	for _, v := range secret {
		raw = binary.LittleEndian.AppendUint32(raw, math.Float32bits(v))
	}
	note := []byte("captured at the north gate")

	fill := func(o Options) string {
		s, path := newStore(t, o)
		if _, err := s.Add(ctx, "emp-1", secret, note); err != nil {
			t.Fatal(err)
		}
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
		return path
	}
	holds := func(path string, needle []byte) bool {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return bytes.Contains(b, needle)
	}

	plain := fill(testOptions())
	if !holds(plain, raw) || !holds(plain, note) {
		t.Fatal("the test cannot see a vector in a plain store, so it proves nothing about an encrypted one")
	}
	sealed := fill(Options{Dim: testDim, Fingerprint: testFingerprint, Key: key(1)})
	if holds(sealed, raw) || holds(sealed, note) || holds(sealed, raw[:16]) {
		t.Fatal("an encrypted store holds a vector or its metadata in the clear")
	}
	if !holds(sealed, []byte("emp-1")) {
		t.Fatal("subject ids are documented to stay in the clear")
	}

	for name, c := range map[string]struct {
		path string
		key  []byte
	}{
		"wrong key":              {sealed, key(2)},
		"no key":                 {sealed, nil},
		"key for a plain store":  {plain, key(1)},
		"short key":              {sealed, []byte("too short")},
		"short key, plain store": {plain, make([]byte, 16)},
	} {
		if _, err := Open(c.path, Options{Key: c.key}); !errors.Is(err, ErrKey) {
			t.Errorf("%s: %v", name, err)
		}
	}
	s := open(t, sealed, Options{Key: key(1)})
	got := contents(t, s)
	if len(got) != 1 || !sameVector(got[1].Vector, secret) || !bytes.Equal(got[1].Meta, note) {
		t.Fatalf("%+v", got)
	}

	// two encryptions of the same thing do not look alike
	a, _ := os.ReadFile(sealed)
	b, _ := os.ReadFile(fill(Options{Dim: testDim, Fingerprint: testFingerprint, Key: key(1)}))
	if bytes.Equal(a, b) {
		t.Fatal("two encrypted stores with the same content are byte-identical")
	}
}

// tamper edits a store file behind the store's back.
func tamper(t *testing.T, path string, fn func(templates *bolt.Bucket) error) {
	t.Helper()
	db, err := bolt.Open(path, 0o600, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if err := db.Update(func(tx *bolt.Tx) error { return fn(tx.Bucket(bucketTemplates)) }); err != nil {
		t.Fatal(err)
	}
}

func TestTamperingIsDetected(t *testing.T) {
	rng := mrand.New(mrand.NewSource(4))
	o := Options{Dim: testDim, Fingerprint: testFingerprint, Key: key(9)}
	build := func() string {
		s, path := newStore(t, o)
		for _, subject := range []string{"alice", "bob"} {
			if _, err := s.Add(ctx, subject, vector(rng, testDim), []byte(subject)); err != nil {
				t.Fatal(err)
			}
		}
		s.Close()
		return path
	}
	broken := func(path string) {
		t.Helper()
		s := open(t, path, o)
		defer s.Close()
		if err := s.Load(ctx, func(Template) error { return nil }); !errors.Is(err, ErrCorrupt) {
			t.Fatalf("Load: %v", err)
		}
	}

	// one bit of a sealed vector
	path := build()
	tamper(t, path, func(b *bolt.Bucket) error {
		k, v := b.Cursor().First()
		v = bytes.Clone(v)
		v[len(v)-20] ^= 0x04
		return b.Put(bytes.Clone(k), v)
	})
	broken(path)

	// bob's record under alice's name: a valid ciphertext in the wrong place
	path = build()
	tamper(t, path, func(b *bolt.Bucket) error {
		c := b.Cursor()
		ak, _ := c.First()
		_, bv := c.Next()
		return b.Put(bytes.Clone(ak), bytes.Clone(bv))
	})
	broken(path)
	s := open(t, path, o)
	if _, err := s.Subject(ctx, "alice"); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("Subject: %v", err)
	}

	// a plain store notices a record of the wrong size
	plain, ppath := newStore(t, testOptions())
	if _, err := plain.Add(ctx, "carol", vector(rng, testDim), nil); err != nil {
		t.Fatal(err)
	}
	plain.Close()
	tamper(t, ppath, func(b *bolt.Bucket) error {
		k, v := b.Cursor().First()
		return b.Put(bytes.Clone(k), bytes.Clone(v[:len(v)-3]))
	})
	p := open(t, ppath, testOptions())
	if err := p.Load(ctx, func(Template) error { return nil }); !errors.Is(err, ErrCorrupt) || !strings.Contains(err.Error(), "carol") {
		t.Fatalf("Load of a truncated record: %v", err)
	}
}

func TestRemove(t *testing.T) {
	rng := mrand.New(mrand.NewSource(5))
	s, path := newStore(t, testOptions())
	add := func(subject string) uint64 {
		t.Helper()
		tm, err := s.Add(ctx, subject, vector(rng, testDim), nil)
		if err != nil {
			t.Fatal(err)
		}
		return tm.ID
	}
	a1, a2, b1 := add("anna"), add("anna"), add("boris")
	count := func(wantSubjects, wantTemplates int) {
		t.Helper()
		if subjects, templates, _ := s.Count(ctx); subjects != wantSubjects || templates != wantTemplates {
			t.Fatalf("count %d/%d, want %d/%d", subjects, templates, wantSubjects, wantTemplates)
		}
	}
	count(2, 3)

	for _, c := range []struct {
		subject string
		id      uint64
		want    bool
	}{{"anna", a1, true}, {"anna", a1, false}, {"anna", b1, false}, {"nobody", a2, false}, {"bad id", 1, false}} {
		if got, err := s.Remove(ctx, c.subject, c.id); err != nil || got != c.want {
			t.Fatalf("Remove(%q, %d) = %v, %v", c.subject, c.id, got, err)
		}
	}
	count(2, 2)
	if sub, err := s.Subject(ctx, "anna"); err != nil || len(sub.Templates) != 1 || sub.Templates[0].ID != a2 {
		t.Fatalf("%+v %v", sub, err)
	}
	// the last template takes the subject with it
	if ok, _ := s.Remove(ctx, "anna", a2); !ok {
		t.Fatal("Remove")
	}
	if _, err := s.Subject(ctx, "anna"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("%v", err)
	}
	count(1, 1)

	add("boris")
	if n, err := s.RemoveSubject(ctx, "boris"); err != nil || n != 2 {
		t.Fatalf("%d %v", n, err)
	}
	if n, err := s.RemoveSubject(ctx, "boris"); err != nil || n != 0 {
		t.Fatalf("%d %v", n, err)
	}
	if n, _ := s.RemoveSubject(ctx, "not/valid"); n != 0 {
		t.Fatal("RemoveSubject of an impossible id")
	}
	count(0, 0)
	if len(contents(t, s)) != 0 {
		t.Fatal("templates left behind")
	}

	// a subject that comes back starts afresh, and the counts survive a restart
	id := add("anna")
	if id <= b1 {
		t.Fatalf("id %d was used before", id)
	}
	s.Close()
	s = open(t, path, testOptions())
	count(1, 1)

	// removing one subject leaves a neighbour with a longer name alone
	add("ann")
	add("anna-maria")
	if n, _ := s.RemoveSubject(ctx, "ann"); n != 1 {
		t.Fatalf("removed %d", n)
	}
	count(2, 2)
}

func TestSubjectListing(t *testing.T) {
	rng := mrand.New(mrand.NewSource(6))
	s, _ := newStore(t, testOptions())
	var entries []Entry
	var ids []string
	for i := 0; i < 250; i++ {
		id := fmt.Sprintf("u%03d", (i*37)%250) // not in order
		ids = append(ids, id)
		entries = append(entries, Entry{Subject: id, Vector: vector(rng, testDim)})
		if i%10 == 0 {
			entries = append(entries, Entry{Subject: id, Vector: vector(rng, testDim)})
		}
	}
	if _, err := s.AddMany(ctx, entries); err != nil {
		t.Fatal(err)
	}
	sort.Strings(ids)

	var got []string
	after := ""
	for page := 0; ; page++ {
		list, err := s.Subjects(ctx, after, 60)
		if err != nil {
			t.Fatal(err)
		}
		if len(list) == 0 {
			break
		}
		if len(list) > 60 || page > 10 {
			t.Fatalf("page %d has %d entries", page, len(list))
		}
		for _, sm := range list {
			got = append(got, sm.ID)
			want := 1
			if n := (indexOf(ids, sm.ID)); n >= 0 && twice(entries, sm.ID) {
				want = 2
			}
			if sm.Templates != want || sm.Created.IsZero() {
				t.Fatalf("%+v, want %d templates", sm, want)
			}
		}
		after = list[len(list)-1].ID
	}
	if strings.Join(got, ",") != strings.Join(ids, ",") {
		t.Fatalf("listing returned %d subjects out of order or incomplete", len(got))
	}

	if list, _ := s.Subjects(ctx, "", 0); len(list) != 100 {
		t.Fatalf("default page of %d", len(list))
	}
	if list, _ := s.Subjects(ctx, "", 5000); len(list) != 250 {
		t.Fatalf("large page of %d", len(list))
	}
	// the cursor need not be an existing subject
	if list, _ := s.Subjects(ctx, "u1005", 2); len(list) != 2 || list[0].ID != "u101" {
		t.Fatalf("%+v", list)
	}
	if list, _ := s.Subjects(ctx, "zzz", 10); len(list) != 0 {
		t.Fatalf("%+v", list)
	}
}

func indexOf(list []string, s string) int {
	for i, v := range list {
		if v == s {
			return i
		}
	}
	return -1
}

func twice(entries []Entry, subject string) bool {
	n := 0
	for _, e := range entries {
		if e.Subject == subject {
			n++
		}
	}
	return n == 2
}

func TestValidation(t *testing.T) {
	rng := mrand.New(mrand.NewSource(7))
	s, _ := newStore(t, testOptions())
	good := vector(rng, testDim)
	nan := append([]float32(nil), good...)
	nan[2] = float32(math.NaN())

	for name, c := range map[string]struct {
		subject string
		vec     []float32
		meta    []byte
		want    error
	}{
		"empty id":       {"", good, nil, ErrSubject},
		"long id":        {strings.Repeat("x", MaxSubject+1), good, nil, ErrSubject},
		"space":          {"jane doe", good, nil, ErrSubject},
		"slash":          {"dept/jane", good, nil, ErrSubject},
		"newline":        {"jane\n", good, nil, ErrSubject},
		"zero byte":      {"jane\x00x", good, nil, ErrSubject},
		"invalid UTF-8":  {"jane\xff", good, nil, ErrSubject},
		"short vector":   {"jane", good[:testDim-1], nil, ErrVector},
		"zero vector":    {"jane", make([]float32, testDim), nil, ErrVector},
		"NaN":            {"jane", nan, nil, ErrVector},
		"oversized meta": {"jane", good, make([]byte, MaxMeta+1), ErrMeta},
	} {
		if _, err := s.Add(ctx, c.subject, c.vec, c.meta); !errors.Is(err, c.want) {
			t.Errorf("%s: %v, want %v", name, err, c.want)
		}
	}
	for _, id := range []string{"jane", "jane.doe@example.com", "30112345678901", "Жасур", "a", strings.Repeat("x", MaxSubject), "emp:42+test=~ok"} {
		if err := ValidSubject(id); err != nil {
			t.Errorf("%q should be a valid id: %v", id, err)
		}
	}
	if _, err := s.Add(ctx, "jane", good, make([]byte, MaxMeta)); err != nil {
		t.Errorf("metadata of exactly the limit: %v", err)
	}

	// one bad entry and nothing is written
	_, err := s.AddMany(ctx, []Entry{{Subject: "a", Vector: good}, {Subject: "b", Vector: nan}, {Subject: "c", Vector: good}})
	if !errors.Is(err, ErrVector) || !strings.Contains(err.Error(), "entry 1") {
		t.Fatalf("%v", err)
	}
	if subjects, templates, _ := s.Count(ctx); subjects != 1 || templates != 1 {
		t.Fatalf("a rejected batch left %d subjects and %d templates", subjects, templates)
	}
	if added, err := s.AddMany(ctx, nil); err != nil || added != nil {
		t.Fatalf("an empty batch: %v %v", added, err)
	}
	if _, err := s.Subject(ctx, "not valid"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("%v", err)
	}
}

func TestAddMany(t *testing.T) {
	rng := mrand.New(mrand.NewSource(8))
	s, _ := newStore(t, testOptions())
	entries := make([]Entry, 2000)
	for i := range entries {
		entries[i] = Entry{Subject: fmt.Sprintf("s%04d", i%700), Vector: vector(rng, testDim), Meta: []byte{byte(i)}}
	}
	added, err := s.AddMany(ctx, entries)
	if err != nil || len(added) != len(entries) {
		t.Fatal(len(added), err)
	}
	for i, tm := range added {
		if tm.ID != uint64(i+1) || tm.Subject != entries[i].Subject {
			t.Fatalf("entry %d became %+v", i, tm)
		}
	}
	got := contents(t, s)
	for i, tm := range added {
		if !sameVector(got[tm.ID].Vector, entries[i].Vector) || got[tm.ID].Meta[0] != byte(i) {
			t.Fatalf("entry %d did not come back", i)
		}
	}
	if subjects, templates, _ := s.Count(ctx); subjects != 700 || templates != 2000 {
		t.Fatalf("%d/%d", subjects, templates)
	}
}

func TestOneProcessAtATime(t *testing.T) {
	s, path := newStore(t, testOptions())
	start := time.Now()
	_, err := Open(path, Options{Timeout: 150 * time.Millisecond})
	if !errors.Is(err, ErrBusy) {
		t.Fatalf("%v", err)
	}
	if waited := time.Since(start); waited < 100*time.Millisecond || waited > 3*time.Second {
		t.Fatalf("waited %v for the lock", waited)
	}
	s.Close()
	open(t, path, testOptions())
}

func TestReadOnly(t *testing.T) {
	rng := mrand.New(mrand.NewSource(9))
	s, path := newStore(t, testOptions())
	v := vector(rng, testDim)
	tm, err := s.Add(ctx, "a", v, []byte("m"))
	if err != nil {
		t.Fatal(err)
	}
	s.Close()

	ro := open(t, path, Options{ReadOnly: true})
	also := open(t, path, Options{ReadOnly: true}) // readers share
	for _, r := range []Store{ro, also} {
		if got := contents(t, r); len(got) != 1 || !sameVector(got[tm.ID].Vector, v) {
			t.Fatalf("%+v", got)
		}
	}
	if _, err := ro.Add(ctx, "b", v, nil); !errors.Is(err, ErrReadOnly) {
		t.Errorf("Add: %v", err)
	}
	if _, err := ro.AddMany(ctx, []Entry{{Subject: "b", Vector: v}}); !errors.Is(err, ErrReadOnly) {
		t.Errorf("AddMany: %v", err)
	}
	if _, err := ro.Remove(ctx, "a", tm.ID); !errors.Is(err, ErrReadOnly) {
		t.Errorf("Remove: %v", err)
	}
	if _, err := ro.RemoveSubject(ctx, "a"); !errors.Is(err, ErrReadOnly) {
		t.Errorf("RemoveSubject: %v", err)
	}
	if sub, err := ro.Subject(ctx, "a"); err != nil || len(sub.Templates) != 1 {
		t.Errorf("Subject: %+v %v", sub, err)
	}
	if list, err := ro.Subjects(ctx, "", 10); err != nil || len(list) != 1 {
		t.Errorf("Subjects: %+v %v", list, err)
	}

	if _, err := Open(filepath.Join(t.TempDir(), "absent.db"), Options{ReadOnly: true}); err == nil {
		t.Error("a read-only store appeared out of nothing")
	}
	// a file that is a database but not a store
	other := filepath.Join(t.TempDir(), "other.db")
	db, err := bolt.Open(other, 0o600, nil)
	if err != nil {
		t.Fatal(err)
	}
	db.Close()
	if _, err := Open(other, Options{ReadOnly: true}); err == nil {
		t.Error("an empty database passed for a store")
	}
}

func TestClose(t *testing.T) {
	rng := mrand.New(mrand.NewSource(10))
	s, _ := newStore(t, testOptions())
	v := vector(rng, testDim)
	if _, err := s.Add(ctx, "a", v, nil); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("closing twice: %v", err)
	}
	checks := map[string]error{}
	_, checks["Add"] = s.Add(ctx, "b", v, nil)
	_, checks["Remove"] = s.Remove(ctx, "a", 1)
	_, checks["RemoveSubject"] = s.RemoveSubject(ctx, "a")
	_, checks["Subject"] = s.Subject(ctx, "a")
	_, checks["Subjects"] = s.Subjects(ctx, "", 1)
	_, _, checks["Count"] = s.Count(ctx)
	checks["Load"] = s.Load(ctx, func(Template) error { return nil })
	checks["Backup"] = s.Backup(ctx, &bytes.Buffer{})
	for name, err := range checks {
		if !errors.Is(err, ErrClosed) {
			t.Errorf("%s after Close: %v", name, err)
		}
	}
	if s.Info().Dim != testDim {
		t.Error("Info should still answer")
	}
}

func TestLoadStops(t *testing.T) {
	rng := mrand.New(mrand.NewSource(11))
	s, _ := newStore(t, testOptions())
	entries := make([]Entry, 1000)
	for i := range entries {
		entries[i] = Entry{Subject: fmt.Sprintf("s%d", i), Vector: vector(rng, testDim)}
	}
	if _, err := s.AddMany(ctx, entries); err != nil {
		t.Fatal(err)
	}

	stop := errors.New("enough")
	seen := 0
	if err := s.Load(ctx, func(Template) error {
		if seen++; seen == 10 {
			return stop
		}
		return nil
	}); !errors.Is(err, stop) || seen != 10 {
		t.Fatalf("%v after %d templates", err, seen)
	}

	cancelled, cancel := context.WithCancel(ctx)
	seen = 0
	err := s.Load(cancelled, func(Template) error {
		if seen++; seen == 300 {
			cancel()
		}
		return nil
	})
	if !errors.Is(err, context.Canceled) || seen >= 1000 {
		t.Fatalf("%v after %d templates", err, seen)
	}
	if _, err := s.Add(cancelled, "x", vector(rng, testDim), nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("Add with a cancelled context: %v", err)
	}
}

func TestBackup(t *testing.T) {
	rng := mrand.New(mrand.NewSource(12))
	o := Options{Dim: testDim, Fingerprint: testFingerprint, Key: key(3)}
	s, _ := newStore(t, o)
	for i := 0; i < 50; i++ {
		if _, err := s.Add(ctx, fmt.Sprintf("s%d", i%20), vector(rng, testDim), []byte{byte(i)}); err != nil {
			t.Fatal(err)
		}
	}
	var buf bytes.Buffer
	if err := s.Backup(ctx, &buf); err != nil {
		t.Fatal(err)
	}
	// the original moves on; the copy does not
	if _, err := s.RemoveSubject(ctx, "s0"); err != nil {
		t.Fatal(err)
	}

	copyPath := filepath.Join(t.TempDir(), "copy.db")
	if err := os.WriteFile(copyPath, buf.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	restored := open(t, copyPath, o)
	if subjects, templates, _ := restored.Count(ctx); subjects != 20 || templates != 50 {
		t.Fatalf("the copy holds %d subjects and %d templates", subjects, templates)
	}
	got := contents(t, restored)
	if len(got) != 50 || got[1].Subject != "s0" {
		t.Fatalf("%d templates in the copy", len(got))
	}
	if _, err := Open(copyPath+".x", Options{Key: key(4)}); err == nil {
		t.Fatal("unreachable")
	}
}

func TestConcurrentUse(t *testing.T) {
	s, _ := newStore(t, testOptions())
	var wg sync.WaitGroup
	const writers, perWriter = 6, 40
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			rng := mrand.New(mrand.NewSource(int64(w)))
			for i := 0; i < perWriter; i++ {
				subject := fmt.Sprintf("w%d-%d", w, i)
				tm, err := s.Add(ctx, subject, vector(rng, testDim), nil)
				if err != nil {
					t.Error(err)
					return
				}
				if i%4 == 0 {
					if ok, err := s.Remove(ctx, subject, tm.ID); err != nil || !ok {
						t.Error(ok, err)
						return
					}
				}
			}
		}(w)
	}
	stop := make(chan struct{})
	var readers sync.WaitGroup
	for r := 0; r < 3; r++ {
		readers.Add(1)
		go func() {
			defer readers.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				n := 0
				if err := s.Load(ctx, func(Template) error { n++; return nil }); err != nil {
					t.Error(err)
					return
				}
				if _, err := s.Subjects(ctx, "", 50); err != nil {
					t.Error(err)
					return
				}
				s.Subject(ctx, "w1-1")
			}
		}()
	}
	wg.Wait()
	close(stop)
	readers.Wait()
	want := writers * perWriter * 3 / 4
	if subjects, templates, _ := s.Count(ctx); subjects != want || templates != want {
		t.Fatalf("%d subjects and %d templates, want %d of each", subjects, templates, want)
	}
	if got := contents(t, s); len(got) != want {
		t.Fatalf("%d templates on disk, want %d", len(got), want)
	}
}

func TestUnknownFormatIsRefused(t *testing.T) {
	s, path := newStore(t, testOptions())
	s.Close()
	db, err := bolt.Open(path, 0o600, nil)
	if err != nil {
		t.Fatal(err)
	}
	err = db.Update(func(tx *bolt.Tx) error {
		return tx.Bucket(bucketMeta).Put(metaFormat, binary.LittleEndian.AppendUint32(nil, 99))
	})
	db.Close()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Open(path, testOptions()); err == nil || !strings.Contains(err.Error(), "format") {
		t.Fatalf("%v", err)
	}
}

func TestSealer(t *testing.T) {
	if _, err := newSealer(make([]byte, 31)); !errors.Is(err, ErrKey) {
		t.Fatalf("%v", err)
	}
	k := make([]byte, KeySize)
	if _, err := rand.Read(k); err != nil {
		t.Fatal(err)
	}
	enc, err := newSealer(k)
	if err != nil {
		t.Fatal(err)
	}
	plain, _ := newSealer(nil)
	record := []byte("a record")
	for name, s := range map[string]sealer{"encrypted": enc, "plain": plain} {
		stored, err := s.seal([]byte("here"), record)
		if err != nil {
			t.Fatal(err)
		}
		if got, err := s.open([]byte("here"), stored); err != nil || !bytes.Equal(got, record) {
			t.Fatalf("%s: %q %v", name, got, err)
		}
		if _, err := s.open([]byte("here"), nil); !errors.Is(err, ErrCorrupt) {
			t.Fatalf("%s, empty value: %v", name, err)
		}
		if _, err := s.open([]byte("here"), []byte{42, 1, 2, 3}); !errors.Is(err, ErrCorrupt) {
			t.Fatalf("%s, unknown envelope: %v", name, err)
		}
	}
	stored, _ := enc.seal([]byte("here"), record)
	if _, err := enc.open([]byte("there"), stored); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("moved record: %v", err)
	}
	if _, err := enc.open([]byte("here"), stored[:10]); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("truncated record: %v", err)
	}
	if _, err := plain.open([]byte("here"), stored); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("sealed record in a plain store: %v", err)
	}
	clear, _ := plain.seal(nil, record)
	if _, err := enc.open(nil, clear); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("plain record in an encrypted store: %v", err)
	}
}

func TestRecordDecoding(t *testing.T) {
	vec := []float32{1, 2, 3, 4}
	record := encodeTemplate(12345, []byte("meta"), vec)
	out := make([]float32, 4)
	created, meta, err := decodeTemplate(record, 4, out)
	if err != nil || created != 12345 || string(meta) != "meta" || !sameVector(out, vec) {
		t.Fatalf("%d %q %v %v", created, meta, out, err)
	}
	for name, bad := range map[string][]byte{
		"empty":            nil,
		"too short":        record[:5],
		"unknown version":  append([]byte{9}, record[1:]...),
		"vector cut short": record[:len(record)-1],
		"vector too long":  append(bytes.Clone(record), 0),
		"meta length lies": append(bytes.Clone(record[:9]), 0xff, 0xff, 0x03),
	} {
		if _, _, err := decodeTemplate(bad, 4, out); !errors.Is(err, ErrCorrupt) {
			t.Errorf("%s: %v", name, err)
		}
	}
	if _, _, err := decodeTemplate(record, 5, make([]float32, 5)); !errors.Is(err, ErrCorrupt) {
		t.Errorf("a record of another dimension: %v", err)
	}
	if _, _, err := decodeSubject([]byte{1, 2, 3}); !errors.Is(err, ErrCorrupt) {
		t.Errorf("short subject record: %v", err)
	}
	for _, key := range [][]byte{nil, []byte("short"), []byte("no-separator-here")} {
		if _, _, err := splitTemplateKey(key); !errors.Is(err, ErrCorrupt) {
			t.Errorf("key %q: %v", key, err)
		}
	}
	if subject, id, err := splitTemplateKey(templateKey("ann", 77)); err != nil || subject != "ann" || id != 77 {
		t.Errorf("%q %d %v", subject, id, err)
	}
}
