package likeness

import (
	"errors"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/muhriddinsalohiddin/likeness/internal/detect"
	"github.com/muhriddinsalohiddin/likeness/internal/embed"
	"github.com/muhriddinsalohiddin/likeness/internal/home"
	"github.com/muhriddinsalohiddin/likeness/internal/models"
	"github.com/muhriddinsalohiddin/likeness/internal/ortx"
	"github.com/muhriddinsalohiddin/likeness/internal/vision"
)

// Config says which models an Engine uses and where it finds them. The zero
// value selects the default preset in the default location.
type Config struct {
	// Preset names a detector and embedder pair. "default" uses the OpenCV
	// Zoo models, whose weights are permissively licensed; "buffalo_s" and
	// "buffalo_l" use InsightFace models, which may only be used for
	// non-commercial research. Empty means "default".
	Preset string
	// Home is the directory that holds the runtime library (in lib/) and the
	// model files (in models/). Empty means $LIKENESS_HOME, or a likeness
	// directory in the user's cache directory.
	Home string
	// Library is the path of the ONNX Runtime shared library. Empty means
	// $LIKENESS_ORT_LIBRARY, or its place under Home.
	Library string
	// Threads is the number of CPU threads one inference may use; 0 means
	// all of them. A process can hold several engines only if they agree on
	// Library and Threads.
	Threads int
	// MaxPixels and MaxSide bound the pictures Analyze accepts; pictures
	// beyond them are refused before they are decoded. Zero means 25
	// megapixels and 8192 pixels.
	MaxPixels, MaxSide int
}

// Info describes a running Engine.
type Info struct {
	Preset   string
	Detector string
	Embedder string
	// Fingerprint identifies the embedder's weights. Vectors made under
	// different fingerprints must never be compared with each other.
	Fingerprint string
	// Dim is the length of the vectors the engine produces.
	Dim int
	// Runtime is the version of the ONNX Runtime library in use.
	Runtime string
	Threads int
}

// ErrClosed is returned by an Engine that has been closed.
var ErrClosed = errors.New("likeness: engine is closed")

// Engine finds faces in pictures and turns them into vectors. It is safe for
// concurrent use; every call runs to completion on the caller's goroutine.
type Engine struct {
	mu     sync.RWMutex // held for reading by calls in flight, for writing by Close
	closed bool

	info   Info
	det    *detect.Detector
	emb    *embed.Embedder
	limits vision.Limits
	sizes  []int // detector input sizes to try, smallest first
	crops  sync.Pool
}

// The runtime is one per process; engines share it and the last one to
// close releases it.
var sharedRuntime struct {
	sync.Mutex
	rt   *ortx.Runtime
	refs int
}

func acquireRuntime(cfg ortx.Config) (*ortx.Runtime, error) {
	sharedRuntime.Lock()
	defer sharedRuntime.Unlock()
	rt, err := ortx.Load(cfg)
	if err != nil {
		return nil, err
	}
	sharedRuntime.rt = rt
	sharedRuntime.refs++
	return rt, nil
}

func releaseRuntime() {
	sharedRuntime.Lock()
	defer sharedRuntime.Unlock()
	if sharedRuntime.refs--; sharedRuntime.refs == 0 {
		sharedRuntime.rt.Close()
		sharedRuntime.rt = nil
	}
}

// Presets lists the names Config.Preset accepts.
func Presets() []string {
	names := make([]string, len(models.Presets))
	for i, p := range models.Presets {
		names[i] = p.Name
	}
	return names
}

// Open loads the models and gets them ready, including a first inference so
// that the first real request is not the slow one. It fails if the runtime
// library or a model file is missing or does not match its checksum.
func Open(cfg Config) (*Engine, error) {
	name := cfg.Preset
	if name == "" {
		name = "default"
	}
	preset, ok := models.LookupPreset(name)
	if !ok {
		return nil, fmt.Errorf("likeness: unknown preset %q (have %s)", name, strings.Join(Presets(), ", "))
	}

	dir := cfg.Home
	if dir == "" {
		dir = home.Dir()
	}
	library := cfg.Library
	if library == "" {
		library = os.Getenv("LIKENESS_ORT_LIBRARY")
	}
	if library == "" {
		library = filepath.Join(dir, "lib", home.LibraryName())
	}
	if _, err := os.Stat(library); err != nil {
		return nil, fmt.Errorf("likeness: ONNX Runtime library: %w", err)
	}
	modelsDir := filepath.Join(dir, "models")
	for _, m := range []*models.Manifest{preset.Detector, preset.Embedder} {
		if _, err := models.Installed(modelsDir, m); err != nil {
			return nil, fmt.Errorf("likeness: model %s: %w", m.Name, err)
		}
	}

	rt, err := acquireRuntime(ortx.Config{Library: library, Threads: cfg.Threads})
	if err != nil {
		return nil, fmt.Errorf("likeness: %w", err)
	}
	e := &Engine{
		limits: vision.Limits{MaxPixels: cfg.MaxPixels, MaxSide: cfg.MaxSide},
	}
	fail := func(err error) (*Engine, error) {
		if e.det != nil {
			e.det.Close()
		}
		if e.emb != nil {
			e.emb.Close()
		}
		releaseRuntime()
		return nil, fmt.Errorf("likeness: %w", err)
	}
	if e.det, err = detect.New(rt, preset.Detector, modelsDir); err != nil {
		return fail(err)
	}
	if e.emb, err = embed.New(rt, preset.Embedder, modelsDir); err != nil {
		return fail(err)
	}
	e.info = Info{
		Preset: preset.Name, Detector: preset.Detector.Name, Embedder: preset.Embedder.Name,
		Fingerprint: e.emb.Fingerprint(), Dim: e.emb.Dim(),
		Runtime: rt.Version(), Threads: rt.Config().Threads,
	}
	e.sizes = detectSizes(e.det)
	cropLen := 3 * e.emb.CropSize() * e.emb.CropSize()
	e.crops.New = func() any {
		buf := make([]float32, cropLen)
		return &buf
	}
	if err := e.warmUp(); err != nil {
		return fail(err)
	}
	return e, nil
}

// detectSizes lists the network input sides to try. A detector that takes
// any size starts small, which is three to four times cheaper and enough
// for a face that fills a good part of the frame; one with a fixed input
// has no choice.
func detectSizes(d *detect.Detector) []int {
	full := d.InputSize(detect.Options{})
	sizes := []int{full}
	if small := d.InputSize(detect.Options{Size: full / 2}); small < full {
		sizes = []int{small, full}
	}
	sort.Ints(sizes)
	return sizes
}

// warmUp runs both models once: the first inference of a session allocates
// and tunes, and should not be paid by a caller.
func (e *Engine) warmUp() error {
	blank := vision.FromImage(image.NewGray(image.Rect(0, 0, 64, 64)), 1)
	for _, size := range e.sizes {
		if _, err := e.det.Detect(blank, detect.Options{Size: size}); err != nil {
			return err
		}
	}
	crop := e.crops.Get().(*[]float32)
	defer e.crops.Put(crop)
	clear(*crop)
	return e.emb.EmbedCrop(*crop, make([]float32, e.emb.Dim()))
}

// Info reports which models the engine runs.
func (e *Engine) Info() Info { return e.info }

// Close waits for calls in flight and releases the models. Closing twice is
// harmless.
func (e *Engine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return
	}
	e.closed = true
	e.det.Close()
	e.emb.Close()
	releaseRuntime()
}
