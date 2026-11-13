package store

import (
	"bufio"
	"fmt"
	mrand "math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// The crash test runs a writer in a child process, kills it without warning
// and checks what it left behind. The writer announces every operation
// before starting it and confirms it afterwards, so the parent knows what
// must be on disk, what must not, and which single operation was in flight.
const (
	crashPath = "LIKENESS_STORE_CRASH_PATH"
	crashSeed = "LIKENESS_STORE_CRASH_SEED"
)

func crashSubject(seed int64) string { return fmt.Sprintf("s%02d", seed%23) }

func crashVector(seed int64) []float32 { return vector(mrand.New(mrand.NewSource(seed)), testDim) }

// crashWriter never returns: it works until it is killed.
func crashWriter(path string, seed int64) {
	s, err := Open(path, testOptions())
	if err != nil {
		fmt.Println("fail", err)
		os.Exit(3)
	}
	fmt.Println("ready")
	rng := mrand.New(mrand.NewSource(seed))
	type mine struct {
		subject string
		id      uint64
	}
	var added []mine
	next := seed * 1_000_000
	for {
		switch op := rng.Intn(10); {
		case op < 6: // one template
			next++
			fmt.Println("try add", next)
			tm, err := s.Add(ctx, crashSubject(next), crashVector(next), []byte(strconv.FormatInt(next, 10)))
			if err != nil {
				fmt.Println("fail", err)
				os.Exit(3)
			}
			fmt.Println("ok", tm.ID)
			added = append(added, mine{tm.Subject, tm.ID})
		case op < 8: // a batch, which must land whole or not at all
			n := 2 + rng.Intn(30)
			first := next + 1
			entries := make([]Entry, n)
			for i := range entries {
				next++
				entries[i] = Entry{Subject: crashSubject(next), Vector: crashVector(next), Meta: []byte(strconv.FormatInt(next, 10))}
			}
			fmt.Println("try many", first, n)
			tms, err := s.AddMany(ctx, entries)
			if err != nil {
				fmt.Println("fail", err)
				os.Exit(3)
			}
			fmt.Println("ok", tms[0].ID)
			for _, tm := range tms {
				added = append(added, mine{tm.Subject, tm.ID})
			}
		case len(added) > 0: // take one back
			i := rng.Intn(len(added))
			m := added[i]
			added = append(added[:i], added[i+1:]...)
			fmt.Println("try del", m.subject, m.id)
			if ok, err := s.Remove(ctx, m.subject, m.id); err != nil || !ok {
				fmt.Println("fail", ok, err)
				os.Exit(3)
			}
			fmt.Println("ok", 0)
		}
	}
}

func TestSurvivesBeingKilled(t *testing.T) {
	if path := os.Getenv(crashPath); path != "" {
		seed, _ := strconv.ParseInt(os.Getenv(crashSeed), 10, 64)
		crashWriter(path, seed)
	}
	if testing.Short() {
		t.Skip("starts and kills child processes")
	}
	path := filepath.Join(t.TempDir(), "faces.db")
	rng := mrand.New(mrand.NewSource(time.Now().UnixNano()))

	// what the parent knows to be on disk: template id -> the seed it was made from
	expect := map[uint64]int64{}
	operations, inFlight := 0, 0

	for round := int64(1); round <= 8; round++ {
		cmd := exec.Command(os.Args[0], "-test.run=^TestSurvivesBeingKilled$")
		cmd.Env = append(os.Environ(), crashPath+"="+path, crashSeed+"="+strconv.FormatInt(round, 10))
		out, err := cmd.StdoutPipe()
		if err != nil {
			t.Fatal(err)
		}
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		lines := make(chan string, 1<<16)
		go func() {
			sc := bufio.NewScanner(out)
			for sc.Scan() {
				lines <- sc.Text()
			}
			close(lines)
		}()
		select {
		case first := <-lines:
			if first != "ready" {
				cmd.Process.Kill()
				t.Fatalf("round %d: the writer could not open the store it was killed on: %q", round, first)
			}
		case <-time.After(20 * time.Second):
			cmd.Process.Kill()
			t.Fatalf("round %d: the writer never started", round)
		}
		time.Sleep(time.Duration(30+rng.Intn(250)) * time.Millisecond)
		if err := cmd.Process.Kill(); err != nil {
			t.Fatal(err)
		}
		cmd.Wait()

		// Replay what the writer reported. A "try" with no "ok" is the
		// operation the kill interrupted.
		var pending []string
		for line := range lines {
			f := strings.Fields(line)
			switch {
			case len(f) == 0:
			case f[0] == "fail":
				t.Fatalf("round %d: the writer failed: %s", round, line)
			case f[0] == "try":
				pending = f[1:]
			case f[0] == "ok" && pending != nil:
				operations++
				id, _ := strconv.ParseUint(f[1], 10, 64)
				switch pending[0] {
				case "add":
					seed, _ := strconv.ParseInt(pending[1], 10, 64)
					expect[id] = seed
				case "many":
					first, _ := strconv.ParseInt(pending[1], 10, 64)
					n, _ := strconv.Atoi(pending[2])
					for i := 0; i < n; i++ {
						expect[id+uint64(i)] = first + int64(i)
					}
				case "del":
					gone, _ := strconv.ParseUint(pending[2], 10, 64)
					delete(expect, gone)
				}
				pending = nil
			}
		}

		s, err := Open(path, testOptions())
		if err != nil {
			t.Fatalf("round %d: the store does not open after the kill: %v", round, err)
		}
		got := contents(t, s)

		// Settle the interrupted operation: it happened entirely or not at all.
		if pending != nil {
			inFlight++
			switch pending[0] {
			case "add", "many":
				first, _ := strconv.ParseInt(pending[1], 10, 64)
				n := 1
				if pending[0] == "many" {
					n, _ = strconv.Atoi(pending[2])
				}
				landed := map[int64]uint64{}
				for id, tm := range got {
					if _, known := expect[id]; known {
						continue
					}
					seed, err := strconv.ParseInt(string(tm.Meta), 10, 64)
					if err != nil || seed < first || seed >= first+int64(n) {
						t.Fatalf("round %d: template %d (%q) was never requested", round, id, tm.Meta)
					}
					landed[seed] = id
				}
				if len(landed) != 0 && len(landed) != n {
					t.Fatalf("round %d: a batch of %d was cut in two: %d templates landed", round, n, len(landed))
				}
				for seed, id := range landed {
					expect[id] = seed
				}
			case "del":
				gone, _ := strconv.ParseUint(pending[2], 10, 64)
				if _, still := got[gone]; !still {
					delete(expect, gone)
				}
			}
		}

		if len(got) != len(expect) {
			t.Fatalf("round %d: %d templates on disk, %d confirmed", round, len(got), len(expect))
		}
		perSubject := map[string]int{}
		for id, seed := range expect {
			tm, ok := got[id]
			if !ok {
				t.Fatalf("round %d: confirmed template %d is gone", round, id)
			}
			if tm.Subject != crashSubject(seed) || !sameVector(tm.Vector, crashVector(seed)) || string(tm.Meta) != strconv.FormatInt(seed, 10) {
				t.Fatalf("round %d: template %d came back changed", round, id)
			}
			perSubject[tm.Subject]++
		}
		// the subject table agrees with the templates
		list, err := s.Subjects(ctx, "", 1000)
		if err != nil {
			t.Fatal(err)
		}
		if len(list) != len(perSubject) {
			t.Fatalf("round %d: %d subjects listed, %d have templates", round, len(list), len(perSubject))
		}
		for _, sm := range list {
			if sm.Templates != perSubject[sm.ID] {
				t.Fatalf("round %d: subject %s says %d templates, has %d", round, sm.ID, sm.Templates, perSubject[sm.ID])
			}
		}
		if subjects, templates, _ := s.Count(ctx); subjects != len(perSubject) || templates != len(expect) {
			t.Fatalf("round %d: count %d/%d, want %d/%d", round, subjects, templates, len(perSubject), len(expect))
		}
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("8 kills: %d confirmed operations, %d caught in flight, %d templates at the end", operations, inFlight, len(expect))
	if operations < 20 {
		t.Fatalf("only %d operations were confirmed; the test did not exercise the store", operations)
	}
}
