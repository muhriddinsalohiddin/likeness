package cli

import (
	"bytes"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/muhriddinsalohiddin/likeness/store"
)

// vars makes a getenv out of pairs of names and values.
func vars(pairs ...string) func(string) string {
	set := make(map[string]string)
	for i := 0; i < len(pairs); i += 2 {
		set[pairs[i]] = pairs[i+1]
	}
	return func(name string) string { return set[name] }
}

func TestStoreKey(t *testing.T) {
	raw := make([]byte, store.KeySize)
	for i := range raw {
		raw[i] = byte(i * 7)
	}
	for name, text := range map[string]string{
		"hex":             hex.EncodeToString(raw),
		"upper-case hex":  strings.ToUpper(hex.EncodeToString(raw)),
		"base64":          base64.StdEncoding.EncodeToString(raw),
		"unpadded base64": base64.RawStdEncoding.EncodeToString(raw),
		"URL base64":      base64.URLEncoding.EncodeToString(raw),
		"with a newline":  hex.EncodeToString(raw) + "\n",
	} {
		if key, err := StoreKey(vars(StoreKeyVar, text)); err != nil || !bytes.Equal(key, raw) {
			t.Errorf("%s: %x %v", name, key, err)
		}
	}
	if key, err := StoreKey(vars()); key != nil || err != nil {
		t.Errorf("no key: %x %v", key, err)
	}
	for name, text := range map[string]string{
		"too short":   hex.EncodeToString(raw[:16]),
		"too long":    hex.EncodeToString(raw) + "00",
		"a password":  "correct horse battery staple",
		"31 bytes":    base64.StdEncoding.EncodeToString(raw[:31]),
		"almost hex":  "zz" + hex.EncodeToString(raw)[2:],
		"just spaces": "   ",
	} {
		key, err := StoreKey(vars(StoreKeyVar, text))
		if name == "just spaces" {
			if key != nil || err != nil {
				t.Errorf("%s: %x %v", name, key, err)
			}
			continue
		}
		if err == nil || strings.Contains(err.Error(), strings.TrimSpace(text)) {
			t.Errorf("%s: %x %v", name, key, err)
		}
	}
}

func TestSecret(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "secret")
	if err := os.WriteFile(file, []byte("  from a file\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got, err := Secret(vars("TOKEN", "from the environment"), "TOKEN"); err != nil || got != "from the environment" {
		t.Errorf("a variable: %q %v", got, err)
	}
	if got, err := Secret(vars("TOKEN_FILE", file), "TOKEN"); err != nil || got != "from a file" {
		t.Errorf("a file: %q %v", got, err)
	}
	if got, err := Secret(vars(), "TOKEN"); err != nil || got != "" {
		t.Errorf("neither: %q %v", got, err)
	}
	if _, err := Secret(vars("TOKEN", "sesame-9000", "TOKEN_FILE", file), "TOKEN"); err == nil || strings.Contains(err.Error(), "sesame") {
		t.Errorf("both: %v", err)
	}
	if _, err := Secret(vars("TOKEN_FILE", filepath.Join(dir, "missing")), "TOKEN"); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("a file that is not there: %v", err)
	}
}

func TestFlags(t *testing.T) {
	var out bytes.Buffer
	env := Env{Getenv: vars(), Stdout: &out, Stderr: &out}
	newSet := func() *flag.FlagSet {
		fs := Flags("likeness try", env, "usage: likeness try\n\nflags:\n", "\nthe end\n")
		fs.Int("count", 3, "how many")
		EngineFlags(fs)
		return fs
	}

	fs := newSet()
	if err := Parse(fs, []string{"--count", "5", "--preset", "buffalo_s", "left"}); err != nil || fs.NArg() != 1 {
		t.Errorf("a good command line: %v", err)
	}

	out.Reset()
	if err := Parse(newSet(), []string{"-h"}); !errors.Is(err, flag.ErrHelp) {
		t.Errorf("-h: %v", err)
	}
	for _, want := range []string{"usage: likeness try", "-count int", "-preset string", "-home directory", "-threads int", "the end"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("the help lacks %q:\n%s", want, out.String())
		}
	}

	out.Reset()
	if err := Parse(newSet(), []string{"--cuont", "5"}); !errors.Is(err, ErrUsage) || !strings.Contains(out.String(), "cuont") {
		t.Errorf("a misspelt flag: %v, explained with %q", err, out.String())
	}

	out.Reset()
	if err := Usage(newSet(), "%d is too many", 9); !errors.Is(err, ErrUsage) {
		t.Errorf("Usage: %v", err)
	}
	if got := out.String(); got != "likeness try: 9 is too many\nRun 'likeness try -h' for the flags.\n" {
		t.Errorf("Usage wrote %q", got)
	}
}
