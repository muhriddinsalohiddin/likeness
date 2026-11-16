// Package serve is the `likeness serve` command: it reads the flags and the
// environment, opens the engine and the store, and runs the HTTP server.
package serve

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"strconv"
	"strings"

	"github.com/muhriddinsalohiddin/likeness"
	"github.com/muhriddinsalohiddin/likeness/internal/cli"
	"github.com/muhriddinsalohiddin/likeness/internal/gallery"
	"github.com/muhriddinsalohiddin/likeness/internal/models"
	"github.com/muhriddinsalohiddin/likeness/internal/server"
	"github.com/muhriddinsalohiddin/likeness/store"
)

const usage = `usage: likeness serve --store FILE [flags]

Serves the HTTP API described in docs/api.md over the faces kept in FILE.

flags:
`

const environment = `
environment:
  LIKENESS_ADMIN_KEYS    API keys that may do everything, separated by commas
  LIKENESS_SEARCH_KEYS   API keys that may search and verify
  LIKENESS_METRICS_KEYS  API keys that may read /metrics and nothing else
  LIKENESS_STORE_KEY     32 bytes in hex or base64: encrypts the store at rest
  LIKENESS_HOME          where the runtime library and the models are

Each of the first four can be read from a file instead, named by the same
variable with _FILE appended: LIKENESS_ADMIN_KEYS_FILE and so on.
`

type options struct {
	listen, store   string
	engine          *likeness.Config
	storeKey        []byte
	tlsCert, tlsKey string
	logFormat       string
	logLevel        slog.Level
	// server holds what the flags and the environment say about the
	// server itself; the engine and the gallery are added when they exist.
	server server.Config
}

// size is a flag holding a number of bytes, written as 8MB or 10MiB.
type size int64

func (s *size) String() string { return strconv.FormatInt(int64(*s), 10) }

func (s *size) Set(text string) error {
	n, err := parseSize(text)
	*s = size(n)
	return err
}

func parseSize(text string) (int64, error) {
	text = strings.TrimSpace(text)
	digits := strings.IndexFunc(text, func(r rune) bool { return r < '0' || r > '9' })
	if digits < 0 {
		digits = len(text)
	}
	units := map[string]int64{
		"": 1, "b": 1,
		"kb": 1e3, "mb": 1e6, "gb": 1e9,
		"kib": 1 << 10, "mib": 1 << 20, "gib": 1 << 30,
	}
	n, err := strconv.ParseInt(text[:digits], 10, 64)
	unit, known := units[strings.ToLower(strings.TrimSpace(text[digits:]))]
	if err != nil || !known || n <= 0 || n > (1<<40)/unit {
		return 0, fmt.Errorf("%q is not a size; write it like 8MB or 10MiB", text)
	}
	return n * unit, nil
}

// parse reads the command line and the environment.
func parse(args []string, env cli.Env) (*options, error) {
	o := &options{}
	var (
		search, verify float64
		maxBody        size
		level          string
	)
	fs := cli.Flags("likeness serve", env, usage, environment)
	fs.StringVar(&o.store, "store", "", "`file` that keeps the enrolled faces; made if it does not exist (required)")
	fs.StringVar(&o.listen, "listen", "127.0.0.1:8080", "`address` to listen on")
	o.engine = cli.EngineFlags(fs)
	fs.IntVar(&o.server.MaxInflight, "max-inflight", 0, "pictures analysed at the same time (default 2)")
	fs.IntVar(&o.server.Queue, "queue", 0, "requests that may wait for a free slot, -1 for none (default 8 per slot)")
	fs.DurationVar(&o.server.QueueTimeout, "queue-timeout", 0, "longest a request may wait for a free slot (default 50ms)")
	fs.Float64Var(&search, "search-threshold", 0, "similarity from which a search names a subject (default the model's)")
	fs.Float64Var(&verify, "verify-threshold", 0, "similarity from which a verification passes (default the model's)")
	fs.Var(&maxBody, "max-body", "largest request body `size`, as in 8MB (default 10MiB)")
	fs.IntVar(&o.server.MaxTemplates, "max-templates", 0, "templates a subject may hold, -1 for no bound (default 10)")
	fs.BoolVar(&o.server.AllowCandidates, "allow-candidates", false, "let clients ask for the nearest subjects below the threshold")
	fs.BoolVar(&o.server.NoAuth, "insecure-no-auth", false, "serve without API keys: whoever reaches the port may do anything")
	fs.DurationVar(&o.server.DrainDelay, "drain-delay", 0, "keep serving this long after a stop signal while reporting not ready")
	fs.StringVar(&o.tlsCert, "tls-cert", "", "certificate `file`, to serve HTTPS")
	fs.StringVar(&o.tlsKey, "tls-key", "", "private key `file` of the certificate")
	fs.StringVar(&o.logFormat, "log-format", "text", "log as text or json")
	fs.StringVar(&level, "log-level", "info", "debug, info, warn or error; debug names the subjects of requests")
	if err := cli.Parse(fs, args); err != nil {
		return nil, err
	}

	problem := ""
	switch {
	case fs.NArg() > 0:
		problem = fmt.Sprintf("unexpected argument %q", fs.Arg(0))
	case o.store == "":
		problem = "--store is required: the file that keeps the enrolled faces"
	case o.engine.Threads < 0, o.server.MaxInflight < 0:
		problem = "--threads and --max-inflight cannot be negative"
	case o.server.QueueTimeout < 0, o.server.DrainDelay < 0:
		problem = "--queue-timeout and --drain-delay cannot be negative"
	case search < 0 || search > 1 || verify < 0 || verify > 1:
		problem = "a threshold is a similarity: above 0 and at most 1"
	case (o.tlsCert == "") != (o.tlsKey == ""):
		problem = "--tls-cert and --tls-key go together"
	case o.logFormat != "text" && o.logFormat != "json":
		problem = "--log-format is text or json"
	case o.logLevel.UnmarshalText([]byte(level)) != nil:
		problem = "--log-level is debug, info, warn or error"
	}
	if problem != "" {
		return nil, cli.Usage(fs, "%s", problem)
	}
	o.server.SearchThreshold, o.server.VerifyThreshold = float32(search), float32(verify)
	o.server.MaxBody = int64(maxBody)

	var lists [3][]string
	for i, name := range []string{"LIKENESS_ADMIN_KEYS", "LIKENESS_SEARCH_KEYS", "LIKENESS_METRICS_KEYS"} {
		value, err := cli.Secret(env.Getenv, name)
		if err != nil {
			return nil, err
		}
		lists[i] = splitKeys(value)
	}
	keys, err := server.NewKeys(lists[0], lists[1], lists[2])
	if err != nil {
		return nil, err
	}
	if keys.Len() == 0 && !o.server.NoAuth {
		return nil, errors.New("no API keys: set LIKENESS_ADMIN_KEYS and LIKENESS_SEARCH_KEYS, " +
			"or pass --insecure-no-auth if the port is not reachable by anyone else")
	}
	o.server.Keys = keys

	if o.storeKey, err = cli.StoreKey(env.Getenv); err != nil {
		return nil, err
	}
	return o, nil
}

// splitKeys takes keys separated by commas or line breaks.
func splitKeys(list string) []string {
	var keys []string
	for _, k := range strings.FieldsFunc(list, func(r rune) bool { return r == ',' || r == '\n' || r == '\r' }) {
		if k = strings.TrimSpace(k); k != "" {
			keys = append(keys, k)
		}
	}
	return keys
}

func (o *options) logger(w io.Writer) *slog.Logger {
	opts := &slog.HandlerOptions{Level: o.logLevel}
	if o.logFormat == "json" {
		return slog.New(slog.NewJSONHandler(w, opts))
	}
	return slog.New(slog.NewTextHandler(w, opts))
}

// Run serves until ctx is done. It is a cli.Command.
func Run(ctx context.Context, args []string, env cli.Env) error {
	return run(ctx, args, env, nil)
}

// run is Run with a hook for tests: listening, if not nil, is told the
// address once the server accepts connections.
func run(ctx context.Context, args []string, env cli.Env, listening func(net.Addr)) (err error) {
	o, err := parse(args, env)
	if err != nil {
		return err
	}
	log := o.logger(env.Stderr)

	eng, err := cli.OpenEngine(*o.engine)
	if err != nil {
		return err
	}
	defer eng.Close()
	info := eng.Info()

	st, err := store.Open(o.store, store.Options{Dim: info.Dim, Fingerprint: info.Fingerprint, Key: o.storeKey})
	if err != nil {
		return fmt.Errorf("%s: %w", o.store, describeStoreError(err, info))
	}
	faces, err := gallery.Open(ctx, st, gallery.Options{})
	if err != nil {
		_ = st.Close()
		return fmt.Errorf("%s: %w", o.store, err)
	}
	defer func() {
		if failed := faces.Close(); err == nil && failed != nil {
			err = fmt.Errorf("%s: %w", o.store, failed)
		}
	}()

	cfg := o.server
	cfg.Engine, cfg.Gallery, cfg.Log = eng, faces, log
	srv, err := server.New(cfg)
	if err != nil {
		return err
	}

	ln, err := net.Listen("tcp", o.listen)
	if err != nil {
		return err
	}
	addr := ln.Addr()
	if o.tlsCert != "" {
		cert, err := tls.LoadX509KeyPair(o.tlsCert, o.tlsKey)
		if err != nil {
			_ = ln.Close()
			return fmt.Errorf("the TLS certificate: %w", err)
		}
		ln = tls.NewListener(ln, &tls.Config{
			Certificates: []tls.Certificate{cert},
			MinVersion:   tls.VersionTLS12,
			NextProtos:   []string{"h2", "http/1.1"},
		})
	}

	held := faces.Info()
	log.Info("listening",
		"addr", addr.String(), "tls", o.tlsCert != "", "version", likeness.Version,
		"preset", info.Preset, "detector", info.Detector, "embedder", info.Embedder, "threads", info.Threads,
		"subjects", held.Subjects, "templates", held.Templates, "loaded_ms", held.LoadTime.Milliseconds(),
		"encrypted", held.Store.Encrypted, "keys", cfg.Keys.Len())
	if cfg.NoAuth {
		log.Warn("authentication is off: whoever reaches this port can enrol and remove people")
	}
	if tcp, ok := addr.(*net.TCPAddr); ok && !tcp.IP.IsLoopback() && o.tlsCert == "" {
		log.Warn("listening beyond this machine without TLS: pictures and API keys cross the network in the clear " +
			"unless a proxy in front encrypts them")
	}
	if preset, ok := models.LookupPreset(info.Preset); ok && preset.Embedder.NonCommercial {
		log.Warn("these models are licensed for non-commercial research only", "preset", info.Preset)
	}
	if listening != nil {
		listening(addr)
	}

	err = srv.Serve(ctx, ln)
	log.Info("stopped")
	return err
}

// describeStoreError says what to do about a store that would not open.
func describeStoreError(err error, info likeness.Info) error {
	switch {
	case errors.Is(err, store.ErrMismatch):
		return fmt.Errorf("%w; it cannot be used with the %s models (%s): "+
			"pick the preset it was made with, or another file", err, info.Preset, info.Fingerprint)
	case errors.Is(err, store.ErrKey):
		return fmt.Errorf("%w: check LIKENESS_STORE_KEY", err)
	case errors.Is(err, store.ErrBusy):
		return fmt.Errorf("%w: is another likeness already serving it?", err)
	}
	return err
}
