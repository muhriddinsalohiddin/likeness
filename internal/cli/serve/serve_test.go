package serve

import (
	"bytes"
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"flag"
	"io"
	"log/slog"
	"math/big"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/muhriddinsalohiddin/likeness"
	"github.com/muhriddinsalohiddin/likeness/internal/cli"
	"github.com/muhriddinsalohiddin/likeness/internal/enginetest"
	"github.com/muhriddinsalohiddin/likeness/store"
)

const (
	testAdminKey  = "an-admin-key-for-tests"
	testSearchKey = "a-search-key-for-tests"
)

// parseServe is parse with the environment and the output spelt out.
func parseServe(args []string, getenv func(string) string, out io.Writer) (*options, error) {
	return parse(args, cli.Env{Getenv: getenv, Stdout: io.Discard, Stderr: out})
}

// serve is run with the environment and the output spelt out.
func serve(ctx context.Context, args []string, getenv func(string) string, stderr io.Writer, listening func(net.Addr)) error {
	return run(ctx, args, cli.Env{Getenv: getenv, Stdout: io.Discard, Stderr: stderr}, listening)
}

// vars makes a getenv out of pairs of names and values.
func vars(pairs ...string) func(string) string {
	vars := make(map[string]string)
	for i := 0; i < len(pairs); i += 2 {
		vars[pairs[i]] = pairs[i+1]
	}
	return func(name string) string { return vars[name] }
}

func TestParseServe(t *testing.T) {
	env := vars("LIKENESS_ADMIN_KEYS", testAdminKey)

	o, err := parseServe([]string{"--store", "faces.db"}, env, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if o.store != "faces.db" || o.listen != "127.0.0.1:8080" || o.engine.Preset != "default" || o.engine.Threads != 0 ||
		o.logFormat != "text" || o.logLevel != slog.LevelInfo || o.storeKey != nil {
		t.Errorf("defaults: %+v", o)
	}
	// what is left zero here is filled in by the server
	if c := o.server; c.MaxInflight != 0 || c.Queue != 0 || c.MaxBody != 0 || c.NoAuth || c.AllowCandidates || c.Keys.Len() != 1 {
		t.Errorf("server defaults: %+v", c)
	}

	o, err = parseServe([]string{
		"-store", "/var/lib/likeness/faces.db", "-listen", ":9000", "-preset", "buffalo_s", "-home", "/opt/likeness",
		"-threads", "4", "-max-inflight", "3", "-queue", "-1", "-queue-timeout", "250ms",
		"-search-threshold", "0.55", "-verify-threshold", "0.4", "-max-body", "4MiB", "-max-templates", "5",
		"-allow-candidates", "-drain-delay", "5s", "-tls-cert", "cert.pem", "-tls-key", "key.pem",
		"-log-format", "json", "-log-level", "debug",
	}, env, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if o.listen != ":9000" || o.engine.Preset != "buffalo_s" || o.engine.Home != "/opt/likeness" || o.engine.Threads != 4 ||
		o.tlsCert != "cert.pem" || o.tlsKey != "key.pem" || o.logFormat != "json" || o.logLevel != slog.LevelDebug {
		t.Errorf("flags: %+v", o)
	}
	if c := o.server; c.MaxInflight != 3 || c.Queue != -1 || c.QueueTimeout != 250*time.Millisecond ||
		c.SearchThreshold != 0.55 || c.VerifyThreshold != 0.4 || c.MaxBody != 4<<20 || c.MaxTemplates != 5 ||
		!c.AllowCandidates || c.DrainDelay != 5*time.Second {
		t.Errorf("server flags: %+v", c)
	}
}

func TestParseServeRefuses(t *testing.T) {
	env := vars("LIKENESS_ADMIN_KEYS", testAdminKey)
	for name, c := range map[string]struct {
		args []string
		says string
	}{
		"no store":         {nil, "--store is required"},
		"stray argument":   {[]string{"--store", "f.db", "now"}, `unexpected argument "now"`},
		"unknown flag":     {[]string{"--store", "f.db", "--fast"}, "flag provided but not defined"},
		"negative threads": {[]string{"--store", "f.db", "--threads", "-2"}, "cannot be negative"},
		"negative wait":    {[]string{"--store", "f.db", "--queue-timeout", "-1s"}, "cannot be negative"},
		"threshold":        {[]string{"--store", "f.db", "--search-threshold", "60"}, "a threshold is above 0 and at most 1"},
		"body size":        {[]string{"--store", "f.db", "--max-body", "huge"}, "is not a size"},
		"half a TLS pair":  {[]string{"--store", "f.db", "--tls-cert", "c.pem"}, "go together"},
		"log format":       {[]string{"--store", "f.db", "--log-format", "xml"}, "text or json"},
		"log level":        {[]string{"--store", "f.db", "--log-level", "loud"}, "debug, info, warn or error"},
	} {
		var out bytes.Buffer
		if _, err := parseServe(c.args, env, &out); !errors.Is(err, cli.ErrUsage) {
			t.Errorf("%s: %v", name, err)
		}
		if !strings.Contains(out.String(), c.says) {
			t.Errorf("%s: explained with %q, want %q", name, out.String(), c.says)
		}
	}

	var out bytes.Buffer
	if _, err := parseServe([]string{"-h"}, env, &out); !errors.Is(err, flag.ErrHelp) {
		t.Errorf("-h: %v", err)
	}
	for _, want := range []string{"usage: likeness serve", "-store file", "LIKENESS_ADMIN_KEYS", "_FILE"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("the help lacks %q", want)
		}
	}
}

func TestServeKeys(t *testing.T) {
	args := []string{"--store", "f.db"}

	// keys are required, unless doing without is asked for by name
	if _, err := parseServe(args, vars(), io.Discard); err == nil || !strings.Contains(err.Error(), "LIKENESS_ADMIN_KEYS") {
		t.Errorf("no keys: %v", err)
	}
	o, err := parseServe(append(args, "--insecure-no-auth"), vars(), io.Discard)
	if err != nil || !o.server.NoAuth || o.server.Keys.Len() != 0 {
		t.Errorf("--insecure-no-auth: %+v %v", o, err)
	}

	// several of each kind
	o, err = parseServe(args, vars(
		"LIKENESS_ADMIN_KEYS", testAdminKey,
		"LIKENESS_SEARCH_KEYS", testSearchKey+", another-search-key-0001 ,",
		"LIKENESS_METRICS_KEYS", "a-key-for-prometheus",
	), io.Discard)
	if err != nil || o.server.Keys.Len() != 4 {
		t.Fatalf("four keys: %+v %v", o, err)
	}

	// a weak key is refused without being repeated
	_, err = parseServe(args, vars("LIKENESS_ADMIN_KEYS", testAdminKey, "LIKENESS_SEARCH_KEYS", "hunter2"), io.Discard)
	if err == nil || !strings.Contains(err.Error(), "search-1") || strings.Contains(err.Error(), "hunter2") {
		t.Errorf("a short key: %v", err)
	}

	// from files, as container secrets arrive
	dir := t.TempDir()
	admin := filepath.Join(dir, "admin")
	if err := os.WriteFile(admin, []byte(testAdminKey+"\nsecond-admin-key-0001\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	storeKey := filepath.Join(dir, "store-key")
	if err := os.WriteFile(storeKey, []byte(strings.Repeat("ab", 32)+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	o, err = parseServe(args, vars("LIKENESS_ADMIN_KEYS_FILE", admin, "LIKENESS_STORE_KEY_FILE", storeKey), io.Discard)
	if err != nil || o.server.Keys.Len() != 2 || len(o.storeKey) != store.KeySize {
		t.Fatalf("secrets from files: %+v %v", o, err)
	}
	if _, err := parseServe(args, vars("LIKENESS_ADMIN_KEYS", testAdminKey, "LIKENESS_ADMIN_KEYS_FILE", admin), io.Discard); err == nil ||
		!strings.Contains(err.Error(), "both set") {
		t.Errorf("a key given twice: %v", err)
	}
	if _, err := parseServe(args, vars("LIKENESS_ADMIN_KEYS_FILE", filepath.Join(dir, "missing")), io.Discard); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("a missing secret file: %v", err)
	}
}

func TestParseSize(t *testing.T) {
	for text, want := range map[string]int64{
		"1048576": 1 << 20, "10MiB": 10 << 20, "8MB": 8e6, "8 mb": 8e6, "500KB": 500e3, "64kib": 64 << 10,
		"1GiB": 1 << 30, "2gb": 2e9, "512b": 512, " 3MiB ": 3 << 20,
	} {
		if got, err := parseSize(text); err != nil || got != want {
			t.Errorf("%q: %d %v, want %d", text, got, err, want)
		}
	}
	for _, text := range []string{"", "MB", "0", "0MB", "-5MB", "1.5MB", "10XB", "10 M B", "99999999999GiB", "ten"} {
		if got, err := parseSize(text); err == nil {
			t.Errorf("%q: accepted as %d", text, got)
		}
	}
}

func TestSplitKeys(t *testing.T) {
	for list, want := range map[string]string{
		"":                 "",
		"one":              "one",
		"one,two":          "one|two",
		" one , two ,":     "one|two",
		"one\ntwo\r\n":     "one|two",
		",,\n,":            "",
		"one two,three":    "one two|three", // a space is not a separator; the server will refuse the key
		"one,\n  two\n\n ": "one|two",
	} {
		if got := strings.Join(splitKeys(list), "|"); got != want {
			t.Errorf("%q: %q, want %q", list, got, want)
		}
	}
}

// fakes are the stand-in engines serve has opened, in order.
type fakes struct {
	mu     sync.Mutex
	opened []*enginetest.Fake
}

func (f *fakes) allClosed() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, e := range f.opened {
		if !e.Closed() {
			return false
		}
	}
	return len(f.opened) > 0
}

// useFake puts stand-in engines in place of the real one: a new one for
// every server, as each closes its own.
func useFake(t *testing.T) *fakes {
	t.Helper()
	f := &fakes{}
	real := cli.OpenEngine
	cli.OpenEngine = func(likeness.Config) (cli.Engine, error) {
		e := &enginetest.Fake{}
		f.mu.Lock()
		f.opened = append(f.opened, e)
		f.mu.Unlock()
		return e, nil
	}
	t.Cleanup(func() { cli.OpenEngine = real })
	return f
}

// running is a server started by serve.
type running struct {
	t      *testing.T
	base   string
	stop   context.CancelFunc
	done   chan error
	logs   *syncBuffer
	client *http.Client
}

type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

// start runs serve in the background and waits until it listens.
func start(t *testing.T, env func(string) string, args ...string) *running {
	t.Helper()
	ctx, stop := context.WithCancel(context.Background())
	r := &running{t: t, stop: stop, done: make(chan error, 1), logs: &syncBuffer{}, client: http.DefaultClient}
	listening := make(chan net.Addr, 1)
	go func() {
		r.done <- serve(ctx, append([]string{"--listen", "127.0.0.1:0"}, args...), env, r.logs, func(a net.Addr) { listening <- a })
	}()
	t.Cleanup(func() {
		stop()
		select {
		case <-r.done:
		case <-time.After(10 * time.Second):
			t.Error("serve did not stop")
		}
	})
	select {
	case addr := <-listening:
		r.base = "http://" + addr.String()
	case err := <-r.done:
		r.done <- err
		t.Fatalf("serve gave up before listening: %v\n%s", err, r.logs)
	case <-time.After(10 * time.Second):
		t.Fatal("serve did not start listening")
	}
	return r
}

// shutdown stops the server and returns what serve returned.
func (r *running) shutdown() error {
	r.t.Helper()
	r.stop()
	select {
	case err := <-r.done:
		r.done <- err
		return err
	case <-time.After(10 * time.Second):
		r.t.Fatal("serve did not stop")
		return nil
	}
}

func (r *running) call(method, path, key, body string) (int, string) {
	r.t.Helper()
	req, err := http.NewRequest(method, r.base+path, strings.NewReader(body))
	if err != nil {
		r.t.Fatal(err)
	}
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	res, err := r.client.Do(req)
	if err != nil {
		r.t.Fatal(err)
	}
	defer res.Body.Close()
	text, err := io.ReadAll(res.Body)
	if err != nil {
		r.t.Fatal(err)
	}
	return res.StatusCode, string(text)
}

func TestServe(t *testing.T) {
	useFake(t)
	path := filepath.Join(t.TempDir(), "faces.db")
	env := vars("LIKENESS_ADMIN_KEYS", testAdminKey, "LIKENESS_SEARCH_KEYS", testSearchKey)

	r := start(t, env, "--store", path, "--log-format", "json")
	if status, body := r.call("POST", "/v1/subjects/anna/faces", testAdminKey, "face:anna"); status != http.StatusCreated {
		t.Fatalf("enrol: %d %s", status, body)
	}
	if status, body := r.call("POST", "/v1/search", testSearchKey, "face:anna#2"); status != http.StatusOK || !strings.Contains(body, `"subject":"anna"`) {
		t.Fatalf("search: %d %s", status, body)
	}
	if status, _ := r.call("POST", "/v1/subjects/boris/faces", testSearchKey, "face:boris"); status != http.StatusForbidden {
		t.Errorf("enrolling with a search key: %d", status)
	}
	if status, _ := r.call("GET", "/v1/info", "", ""); status != http.StatusUnauthorized {
		t.Errorf("no key: %d", status)
	}
	if err := r.shutdown(); err != nil {
		t.Fatalf("serve: %v", err)
	}
	logs := r.logs.String()
	for _, want := range []string{`"msg":"listening"`, `"preset":"fake"`, `"keys":2`, `"subjects":0`, `"route":"search"`, `"msg":"stopped"`} {
		if !strings.Contains(logs, want) {
			t.Errorf("the log lacks %s:\n%s", want, logs)
		}
	}
	for _, private := range []string{testAdminKey, testSearchKey, `"anna"`} {
		if strings.Contains(logs, private) {
			t.Errorf("the log has %s in it:\n%s", private, logs)
		}
	}

	// what was enrolled is there after a restart
	r = start(t, env, "--store", path)
	if status, body := r.call("POST", "/v1/verify?subject=anna", testSearchKey, "face:anna"); status != http.StatusOK || !strings.Contains(body, `"verified":true`) {
		t.Errorf("after a restart: %d %s", status, body)
	}
	if !strings.Contains(r.logs.String(), "subjects=1") {
		t.Errorf("the start-up line counts the subjects:\n%s", r.logs)
	}

	// and the file is in use: a second server cannot have it
	err := serve(context.Background(), []string{"--store", path, "--listen", "127.0.0.1:0"}, env, io.Discard, nil)
	if !errors.Is(err, store.ErrBusy) || !strings.Contains(err.Error(), "another likeness") {
		t.Errorf("two servers on one store: %v", err)
	}
}

func TestServeEncryptedStore(t *testing.T) {
	useFake(t)
	path := filepath.Join(t.TempDir(), "faces.db")
	key := strings.Repeat("0123456789abcdef", 4)
	env := vars("LIKENESS_ADMIN_KEYS", testAdminKey, "LIKENESS_STORE_KEY", key)

	r := start(t, env, "--store", path)
	if status, body := r.call("POST", "/v1/subjects/anna/faces", testAdminKey, "face:anna"); status != http.StatusCreated {
		t.Fatalf("enrol: %d %s", status, body)
	}
	if _, body := r.call("GET", "/v1/info", testAdminKey, ""); !strings.Contains(body, `"encrypted":true`) {
		t.Errorf("info: %s", body)
	}
	if err := r.shutdown(); err != nil {
		t.Fatal(err)
	}

	// without the key, or with another, the store stays shut
	for name, env := range map[string]func(string) string{
		"no key":      vars("LIKENESS_ADMIN_KEYS", testAdminKey),
		"another key": vars("LIKENESS_ADMIN_KEYS", testAdminKey, "LIKENESS_STORE_KEY", strings.Repeat("f", 64)),
	} {
		err := serve(context.Background(), []string{"--store", path, "--listen", "127.0.0.1:0"}, env, io.Discard, nil)
		if !errors.Is(err, store.ErrKey) || !strings.Contains(err.Error(), "LIKENESS_STORE_KEY") || strings.Contains(err.Error(), key) {
			t.Errorf("%s: %v", name, err)
		}
	}
	r = start(t, env, "--store", path)
	if status, body := r.call("POST", "/v1/search", testAdminKey, "face:anna"); status != http.StatusOK || !strings.Contains(body, `"match":true`) {
		t.Errorf("with the key again: %d %s", status, body)
	}
}

func TestServeRefusesAnotherModelsStore(t *testing.T) {
	path := filepath.Join(t.TempDir(), "faces.db")
	st, err := store.Open(path, store.Options{Dim: enginetest.Dim, Fingerprint: "some:other:model"})
	if err != nil {
		t.Fatal(err)
	}
	st.Close()

	useFake(t)
	err = serve(context.Background(), []string{"--store", path, "--listen", "127.0.0.1:0", "--insecure-no-auth"}, vars(), io.Discard, nil)
	if !errors.Is(err, store.ErrMismatch) || !strings.Contains(err.Error(), "preset it was made with") {
		t.Errorf("a store made by another model: %v", err)
	}
}

func TestServeFailsBeforeListening(t *testing.T) {
	path := filepath.Join(t.TempDir(), "faces.db")
	args := []string{"--store", path, "--insecure-no-auth"}

	// no models
	real := cli.OpenEngine
	cli.OpenEngine = func(likeness.Config) (cli.Engine, error) { return nil, errors.New("no models here") }
	if err := serve(context.Background(), args, vars(), io.Discard, nil); err == nil || !strings.Contains(err.Error(), "no models here") {
		t.Errorf("an engine that will not open: %v", err)
	}
	cli.OpenEngine = real

	// the port is taken
	engines := useFake(t)
	taken, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer taken.Close()
	if err := serve(context.Background(), append(args, "--listen", taken.Addr().String()), vars(), io.Discard, nil); err == nil {
		t.Error("serve listened on a port that was taken")
	}
	// the certificate is not there
	if err := serve(context.Background(), append(args, "--listen", "127.0.0.1:0", "--tls-cert", "nowhere.pem", "--tls-key", "nowhere.key"),
		vars(), io.Discard, nil); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("a certificate that is not there: %v", err)
	}
	// every failure let go of the store and closed the engine
	if !engines.allClosed() {
		t.Error("an engine was left open")
	}
	st, err := store.Open(path, store.Options{})
	if err != nil {
		t.Fatalf("the store was left locked: %v", err)
	}
	st.Close()
}

func TestServeWarns(t *testing.T) {
	useFake(t)
	r := start(t, vars(), "--store", filepath.Join(t.TempDir(), "faces.db"), "--insecure-no-auth")
	if status, body := r.call("POST", "/v1/subjects/anna/faces", "", "face:anna"); status != http.StatusCreated {
		t.Fatalf("enrol without a key: %d %s", status, body)
	}
	if logs := r.logs.String(); !strings.Contains(logs, "authentication is off") || strings.Contains(logs, "without TLS") {
		t.Errorf("warnings on a loopback port without keys:\n%s", logs)
	}
}

func TestServeTLS(t *testing.T) {
	useFake(t)
	dir := t.TempDir()
	cert, key, pool := selfSigned(t, dir)

	r := start(t, vars("LIKENESS_ADMIN_KEYS", testAdminKey),
		"--store", filepath.Join(dir, "faces.db"), "--tls-cert", cert, "--tls-key", key)
	r.base = strings.Replace(r.base, "http://", "https://", 1)
	r.client = &http.Client{Transport: &http.Transport{
		TLSClientConfig:   &tls.Config{RootCAs: pool, MinVersion: tls.VersionTLS12},
		ForceAttemptHTTP2: true,
	}}
	defer r.client.CloseIdleConnections()

	if status, body := r.call("POST", "/v1/subjects/anna/faces", testAdminKey, "face:anna"); status != http.StatusCreated {
		t.Fatalf("enrol over TLS: %d %s", status, body)
	}
	res, err := r.client.Get(r.base + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if res.ProtoMajor != 2 {
		t.Errorf("served over %s, want HTTP/2", res.Proto)
	}
	if !strings.Contains(r.logs.String(), "tls=true") {
		t.Errorf("the start-up line:\n%s", r.logs)
	}

	// plain HTTP does not get through to the API
	plain, err := http.Get(strings.Replace(r.base, "https://", "http://", 1) + "/healthz")
	if err == nil {
		plain.Body.Close()
		if plain.StatusCode == http.StatusOK {
			t.Error("the TLS port answered plain HTTP")
		}
	}
}

// selfSigned writes a certificate for 127.0.0.1 and its key into dir.
func selfSigned(t *testing.T, dir string) (certFile, keyFile string, pool *x509.CertPool) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	template := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: "likeness test"},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		IPAddresses:  []net.IP{net.IPv4(127, 0, 0, 1)},
	}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	certFile, keyFile = filepath.Join(dir, "cert.pem"), filepath.Join(dir, "key.pem")
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
	if err := os.WriteFile(certFile, certPEM, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(keyFile, pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}), 0o600); err != nil {
		t.Fatal(err)
	}
	pool = x509.NewCertPool()
	pool.AppendCertsFromPEM(certPEM)
	return certFile, keyFile, pool
}
