package setup

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"time"

	"github.com/muhriddinsalohiddin/likeness/internal/models"
)

// item is one file to install.
type item struct {
	// what names the file for people: "detector model yunet_2023mar".
	what     string
	artifact models.Artifact
	// dst is where the file ends up.
	dst string
	// licence says on what terms a model may be used and where it comes
	// from; the runtime library has none here.
	licence string
}

// attempts is how often a download is tried before setup gives up on it.
const attempts = 4

// elsewhere is the way out when this machine cannot get a file itself.
const elsewhere = "download the file on another machine and pass its directory as --from " +
	"('likeness setup --list' shows the URLs)"

// installer brings files into place through models.Fetch, which writes
// next to the destination, checks the digest and only then renames: a run
// that fails or is stopped leaves nothing behind.
type installer struct {
	client *http.Client
	// from is the directory the client reads instead of the network, if
	// any. Here it only changes what people are told.
	from string
	// attempts is how many times one download is tried.
	attempts int
	// wait pauses between two attempts. It returns early, with the error
	// of ctx, when ctx is done.
	wait func(ctx context.Context, d time.Duration) error
	// log receives a line when an item starts and one when it is done.
	log io.Writer
	// redraw says that log is a terminal, where a percentage can be
	// rewritten in place.
	redraw bool
}

// offline makes the installer take the files from dir. Nothing is tried
// twice: a file that is wrong now is as wrong a second later.
func (in *installer) offline(dir string) {
	in.from, in.attempts = dir, 1
	in.client = &http.Client{Transport: directory(dir)}
}

// install brings every item into place, in order, and stops at the first
// one that fails. What is already there and verifies is left alone.
func (in *installer) install(ctx context.Context, items []item) error {
	for _, it := range items {
		if err := ctx.Err(); err != nil {
			return stopped(it, err)
		}
		switch err := models.Verify(it.dst, it.artifact); {
		case err == nil:
			say(in.log, "%s: already installed\n", it.what)
			continue
		case errors.Is(err, models.ErrChecksum):
			say(in.log, "%s: %s is not the file it should be, replacing it\n", it.what, it.dst)
		}
		if err := in.fetch(ctx, it); err != nil {
			return err
		}
	}
	return nil
}

// fetch installs one item, trying again after a failure that another
// attempt may cure.
func (in *installer) fetch(ctx context.Context, it item) error {
	seen := &answers{next: in.client.Transport}
	if seen.next == nil {
		seen.next = http.DefaultTransport
	}
	client := *in.client
	client.Transport = seen

	say(in.log, "%s: fetching %s\n", it.what, in.origin(it))

	var m meter
	var progress models.Progress
	if in.redraw {
		m.w, progress = in.log, m.update
	}
	start := time.Now()
	for attempt := 1; ; attempt++ {
		seen.status = 0
		err := models.Fetch(ctx, &client, it.artifact, it.dst, progress)
		m.clear()
		switch {
		case err == nil:
			say(in.log, "%s: installed as %s (%s in %s)\n", it.what, it.dst,
				size(it.artifact.Size), time.Since(start).Round(100*time.Millisecond))
			return nil
		case ctx.Err() != nil:
			return stopped(it, ctx.Err())
		case attempt >= in.attempts || !transient(err, seen.status):
			return in.explain(it, err, seen.status, attempt)
		}
		pause := time.Second << (attempt - 1)
		say(in.log, "%s: %v; trying again in %s\n", it.what, err, pause)
		if err := in.wait(ctx, pause); err != nil {
			return stopped(it, err)
		}
	}
}

// origin is where an item is taken from: its URL, or the file that stands
// in for it.
func (in *installer) origin(it item) string {
	if in.from == "" {
		return it.artifact.URL
	}
	return filepath.Join(in.from, fileName(it.artifact.URL))
}

// stopped is the error of a run that was told to stop.
func stopped(it item, cause error) error {
	return fmt.Errorf("stopped at the %s; run the command again to finish: %w", it.what, cause)
}

// transient reports whether a download that failed may succeed when it is
// tried again. A file with the wrong digest, a server that says no and a
// disk that cannot be written will be the same the next time.
func transient(err error, status int) bool {
	var file *fs.PathError
	var link *os.LinkError
	switch {
	case errors.Is(err, models.ErrChecksum), status >= 400 && status < 500,
		errors.As(err, &file), errors.As(err, &link):
		return false
	}
	return true
}

// explain adds to a failure what the user can do about it.
func (in *installer) explain(it item, err error, status, tried int) error {
	var file *fs.PathError
	switch {
	case in.from != "" && errors.Is(err, fs.ErrNotExist) && errors.As(err, &file):
		return fmt.Errorf("%s: %w: download %s and put it in %s", it.what, file, it.artifact.URL, in.from)
	case in.from != "" && errors.Is(err, models.ErrChecksum):
		return fmt.Errorf("%s: %w: %s is not the file this version of likeness expects, download it again from %s",
			it.what, err, in.origin(it), it.artifact.URL)
	case errors.Is(err, models.ErrChecksum):
		return fmt.Errorf("%s: %w: what arrived is not the file the catalogue describes, "+
			"as happens behind a proxy that rewrites downloads; %s", it.what, err, elsewhere)
	case status >= 400 && status < 500:
		return fmt.Errorf("%s: %w: the server will not hand the file over and asking again does not help; "+
			"check the proxy settings, or %s", it.what, err, elsewhere)
	case !transient(err, status):
		return fmt.Errorf("%s: %w", it.what, err)
	case in.from != "":
		return fmt.Errorf("%s: %w: %s cannot be used as it is, download it again from %s",
			it.what, err, in.origin(it), it.artifact.URL)
	}
	return fmt.Errorf("%s: %w: gave up after %d attempts; %s", it.what, err, tried, elsewhere)
}

// answers notes the status of the last answer, which models.Fetch reports
// in words only.
type answers struct {
	next   http.RoundTripper
	status int
}

func (a *answers) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := a.next.RoundTrip(req)
	if err == nil {
		a.status = resp.StatusCode
	}
	return resp, err
}

// directory is a transport that answers every request with a file of the
// directory: the one named like the last element of the URL, which is what
// a download on another machine leaves there. Files that were carried over
// thus go through models.Fetch like files that were downloaded, and are
// unpacked and verified by the same code.
type directory string

func (d directory) RoundTrip(req *http.Request) (*http.Response, error) {
	f, err := os.Open(filepath.Join(string(d), path.Base(req.URL.Path)))
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	return &http.Response{
		Status: "200 OK", StatusCode: http.StatusOK,
		Proto: "HTTP/1.1", ProtoMajor: 1, ProtoMinor: 1,
		Header:        make(http.Header),
		Body:          localBody{ctx: req.Context(), f: f},
		ContentLength: st.Size(),
		Request:       req,
	}, nil
}

// localBody stops being read when its request is cancelled, as the body of
// an answer from the network does.
type localBody struct {
	ctx context.Context
	f   *os.File
}

func (b localBody) Read(p []byte) (int, error) {
	if err := b.ctx.Err(); err != nil {
		return 0, err
	}
	return b.f.Read(p)
}

func (b localBody) Close() error { return b.f.Close() }

// fileName is the name a download of rawURL is saved under: the last
// element of its path.
func fileName(rawURL string) string {
	if u, err := url.Parse(rawURL); err == nil {
		return path.Base(u.Path)
	}
	return path.Base(rawURL)
}

// newClient returns the client the downloads go through. It finds the
// proxy in the environment, as office networks need. It has no overall
// timeout, because the files are large and the lines can be slow, but it
// does not wait for ever on a server that accepts a connection and then
// says nothing.
func newClient() *http.Client {
	return &http.Client{Transport: &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           (&net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		ForceAttemptHTTP2:     true,
		TLSHandshakeTimeout:   15 * time.Second,
		ResponseHeaderTimeout: 30 * time.Second,
		IdleConnTimeout:       90 * time.Second,
	}}
}

// wait pauses for d, or until ctx is done.
func wait(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// meter redraws how far a download has got, on one line of a terminal.
// The zero value draws nothing.
type meter struct {
	w     io.Writer
	shown string
}

func (m *meter) update(done, total int64) {
	text := size(done)
	if total > 0 {
		text = fmt.Sprintf("%d%%", done*100/total)
	}
	if text == m.shown {
		return
	}
	m.shown = text
	say(m.w, "\r  %-12s", text)
}

// clear wipes the line, so that what is written next starts on a clean one.
func (m *meter) clear() {
	if m.shown != "" {
		say(m.w, "\r%14s\r", "")
		m.shown = ""
	}
}

// size writes a number of bytes the way people read it.
func size(n int64) string {
	if n < 1e6 {
		return fmt.Sprintf("%.0f kB", float64(n)/1e3)
	}
	return fmt.Sprintf("%.1f MB", float64(n)/1e6)
}

func say(w io.Writer, format string, args ...any) { _, _ = fmt.Fprintf(w, format, args...) }
