// facebench measures every stage of a 1:N face search request
// (JPEG decode -> detect -> align -> embed -> search) to check a 100 ms budget.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"image"
	"image/jpeg"
	"math"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/viterin/vek/vek32"
	ort "github.com/yalue/onnxruntime_go"
)

type stats struct{ mean, p50, p95, p99, max time.Duration }

func summarize(ds []time.Duration) stats {
	s := append([]time.Duration(nil), ds...)
	sort.Slice(s, func(a, b int) bool { return s[a] < s[b] })
	q := func(p float64) time.Duration {
		i := int(math.Ceil(p*float64(len(s)))) - 1
		return s[max(0, min(i, len(s)-1))]
	}
	var sum time.Duration
	for _, d := range s {
		sum += d
	}
	return stats{sum / time.Duration(len(s)), q(0.50), q(0.95), q(0.99), s[len(s)-1]}
}

func ms(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }

func (s stats) String() string {
	return fmt.Sprintf("mean %7.2f  p50 %7.2f  p95 %7.2f  p99 %7.2f  max %7.2f ms", ms(s.mean), ms(s.p50), ms(s.p95), ms(s.p99), ms(s.max))
}

func timeIt(warm, n int, fn func()) stats {
	for i := 0; i < warm; i++ {
		fn()
	}
	ds := make([]time.Duration, n)
	for i := range ds {
		t := time.Now()
		fn()
		ds[i] = time.Since(t)
	}
	return summarize(ds)
}

func rssMB() float64 {
	// Linux: no external tools needed (slim container images have no ps)
	if b, err := os.ReadFile("/proc/self/status"); err == nil {
		for _, line := range strings.Split(string(b), "\n") {
			if rest, ok := strings.CutPrefix(line, "VmRSS:"); ok {
				kb, _ := strconv.ParseFloat(strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(rest), "kB")), 64)
				return kb / 1024
			}
		}
	}
	out, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(os.Getpid())).Output()
	if err != nil {
		return 0
	}
	kb, _ := strconv.ParseFloat(strings.TrimSpace(string(out)), 64)
	return kb / 1024
}

func section(title string) { fmt.Printf("\n== %s ==\n", title) }

type pipeline struct {
	det, rec      *Model
	db, rb        *Binding
	detIn, recIn  []float32
	heads         []scrfdHead
	ix            *Index
	jpg           []byte
	resizeWorkers int
	searchWorkers int
	dot           func(a, b []float32) float32
	emb           []float32
}

type stageTimes struct {
	decode, pre, det, post, warp, rec, search, total time.Duration
	faces                                            int
	best                                             Face
}

func newPipeline(det, rec *Model, ix *Index, jpg []byte, resizeW, searchW int, dot func(a, b []float32) float32) (*pipeline, error) {
	db, err := det.NewBinding()
	if err != nil {
		return nil, err
	}
	rb, err := rec.NewBinding()
	if err != nil {
		return nil, err
	}
	p := &pipeline{det: det, rec: rec, db: db, rb: rb, detIn: db.In.GetData(), recIn: rb.In.GetData(),
		ix: ix, jpg: jpg, resizeWorkers: resizeW, searchWorkers: searchW, dot: dot, emb: make([]float32, ix.d)}
	if det.Spec.Kind == "scrfd" {
		p.heads = scrfdLayout(det)
	}
	return p, nil
}

func (p *pipeline) run() (stageTimes, error) {
	var st stageTimes
	t0 := time.Now()
	img, err := jpeg.Decode(bytes.NewReader(p.jpg))
	if err != nil {
		return st, err
	}
	yc, ok := img.(*image.YCbCr)
	if !ok {
		return st, fmt.Errorf("unsupported pixel format %T (spike handles YCbCr JPEG only)", img)
	}
	t1 := time.Now()

	S := p.det.Spec.Size
	W, H := yc.Rect.Dx(), yc.Rect.Dy()
	scale := math.Min(float64(S)/float64(W), float64(S)/float64(H))
	cw, ch := int(math.Round(float64(W)*scale)), int(math.Round(float64(H)*scale))
	resizeYCbCr(yc, p.detIn, S, S, cw, ch, p.det.Spec.Pre, p.resizeWorkers)
	t2 := time.Now()

	if err := p.db.Run(); err != nil {
		return st, err
	}
	t3 := time.Now()

	var kps [5][2]float32
	found := false
	if p.heads != nil {
		faces := decodeSCRFD(p.db, p.heads, 0.5, 0.4)
		st.faces = len(faces)
		var bestArea float32
		for _, f := range faces {
			if a := (f.X2 - f.X1) * (f.Y2 - f.Y1); a > bestArea {
				bestArea, st.best, found = a, f, true
			}
		}
		if found {
			for i := range kps {
				kps[i] = [2]float32{st.best.Kps[i][0] / float32(scale), st.best.Kps[i][1] / float32(scale)}
			}
		}
	}
	if !found {
		// No face (synthetic frame, or a detector whose decoder the spike
		// does not implement): use a plausible landmark layout so the rest of
		// the pipeline does exactly the same amount of work.
		k := float32(H) / 112 * 0.45
		for i := range kps {
			kps[i] = [2]float32{(arcfaceTemplate[i][0]-56)*k + float32(W)/2, (arcfaceTemplate[i][1]-56)*k + float32(H)/2}
		}
	}
	t4 := time.Now()

	warpYCbCr(yc, p.recIn, p.rec.Spec.Size, kps, p.rec.Spec.Pre)
	t5 := time.Now()

	if err := p.rb.Run(); err != nil {
		return st, err
	}
	copy(p.emb, p.rb.Out[0])
	l2normalize(p.emb)
	t6 := time.Now()

	_ = p.ix.Search(p.emb, 5, p.searchWorkers, p.dot)
	t7 := time.Now()

	st.decode, st.pre, st.det, st.post = t1.Sub(t0), t2.Sub(t1), t3.Sub(t2), t4.Sub(t3)
	st.warp, st.rec, st.search, st.total = t5.Sub(t4), t6.Sub(t5), t7.Sub(t6), t7.Sub(t0)
	return st, nil
}

func main() {
	lib := flag.String("lib", "", "path to the onnxruntime shared library")
	modelDir := flag.String("models", "models", "directory with .onnx files")
	threads := flag.Int("threads", runtime.NumCPU(), "ONNX Runtime intra-op threads per session")
	iters := flag.Int("iters", 100, "iterations per measurement")
	combos := flag.String("combos", "det_500m+w600k_mbf,det_500m+w600k_r50,det_500m@320+w600k_r50,det_10g+w600k_r50,yunet_2023mar+sface_2021dec", "detector+embedder pairs for end-to-end runs")
	concCombo := flag.String("conc-combo", "det_500m+w600k_r50", "pair used for the concurrency test")
	conc := flag.String("conc", "1,2,4,8", "concurrent request counts")
	concSecs := flag.Int("conc-secs", 6, "seconds per concurrency level")
	gallery := flag.Int("gallery", 100000, "enrolled faces in the search index")
	spin := flag.Bool("spin", true, "let idle ORT threads busy-wait (ORT default)")
	detThreads := flag.Int("det-threads", 0, "ORT threads for detector sessions (0 = same as -threads)")
	searchWorkers := flag.Int("search-workers", runtime.NumCPU(), "goroutines for the index scan in end-to-end runs")
	imagePath := flag.String("image", "", "optional real JPEG to use instead of the synthetic 1280x720 frame")
	sections := flag.String("sections", "decode,pre,models,search,e2e,conc", "which parts to run")
	flag.Parse()
	want := map[string]bool{}
	for _, s := range strings.Split(*sections, ",") {
		want[strings.TrimSpace(s)] = true
	}

	if *lib != "" {
		ort.SetSharedLibraryPath(*lib)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		fmt.Fprintln(os.Stderr, "init onnxruntime:", err)
		os.Exit(1)
	}
	defer ort.DestroyEnvironment()

	fmt.Printf("facebench  %s/%s  cpus=%d  ort=%s  ort-threads=%d det-threads=%d spin=%v search-workers=%d  go=%s\n",
		runtime.GOOS, runtime.GOARCH, runtime.NumCPU(), ort.GetVersion(), *threads, *detThreads, *spin, *searchWorkers, runtime.Version())

	frame := synthJPEG(1280, 720, 88, 5)
	if *imagePath != "" {
		b, err := os.ReadFile(*imagePath)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		frame = b
	}

	if want["decode"] {
		section("JPEG decode (Go stdlib image/jpeg)")
		for _, sz := range [][2]int{{640, 480}, {1280, 720}, {1920, 1080}} {
			b := synthJPEG(sz[0], sz[1], 88, 5)
			st := timeIt(5, *iters, func() { jpeg.Decode(bytes.NewReader(b)) })
			fmt.Printf("%4dx%-4d %4d KB  %s\n", sz[0], sz[1], len(b)/1024, st)
		}
	}

	if want["pre"] {
		section("Preprocess (pure Go, fused resize+colour+normalise)")
		img, _ := jpeg.Decode(bytes.NewReader(synthJPEG(1280, 720, 88, 5)))
		yc := img.(*image.YCbCr)
		for _, S := range []int{640, 320} {
			buf := make([]float32, 3*S*S)
			ch := S * 720 / 1280
			for _, w := range []int{1, 4} {
				st := timeIt(5, *iters, func() { resizeYCbCr(yc, buf, S, S, S, ch, Pre{127.5, 128, false}, w) })
				fmt.Printf("1280x720 -> %3d letterbox, %d worker(s)  %s\n", S, w, st)
			}
		}
		buf := make([]float32, 3*112*112)
		var kps [5][2]float32
		for i := range kps {
			kps[i] = [2]float32{(arcfaceTemplate[i][0]-56)*3 + 640, (arcfaceTemplate[i][1]-56)*3 + 360}
		}
		st := timeIt(5, *iters, func() { warpYCbCr(yc, buf, 112, kps, Pre{127.5, 127.5, false}) })
		fmt.Printf("align + warp -> 112x112                 %s\n", st)
	}

	if want["models"] {
		section(fmt.Sprintf("Model inference, single request, %d ORT threads", *threads))
		for _, name := range []string{"det_500m@320", "det_500m", "det_10g@320", "det_10g", "yunet_2023mar", "w600k_mbf", "w600k_r50", "sface_2021dec"} {
			before := rssMB()
			m, err := LoadModel(*modelDir, name, *threads, *spin)
			if err != nil {
				fmt.Printf("%-16s SKIP: %v\n", name, err)
				continue
			}
			b, err := m.NewBinding()
			if err != nil {
				fmt.Printf("%-16s SKIP: %v\n", name, err)
				continue
			}
			if err := b.Run(); err != nil {
				fmt.Printf("%-16s RUN ERROR: %v\n", name, err)
				continue
			}
			st := timeIt(10, *iters, func() { b.Run() })
			fmt.Printf("%-16s in=%v load=%4.0fms rss+%4.0fMB  %s\n", name, m.inShape, ms(m.LoadTime), rssMB()-before, st)
			b.Destroy()
			m.sess.Destroy()
		}
	}

	dots := []struct {
		name string
		fn   func(a, b []float32) float32
	}{{"pure Go", dotGo}, {"vek32 (AVX2 on amd64)", vek32.Dot}}

	if want["search"] {
		section("1:N search, 512-d, top-5, exact")
		for _, n := range []int{7000, *gallery} {
			ix := randomIndex(n, 512, 7)
			q := ix.data[:512]
			for _, d := range dots {
				for _, w := range []int{1, runtime.NumCPU()} {
					st := timeIt(3, *iters, func() { ix.Search(q, 5, w, d.fn) })
					fmt.Printf("N=%-7d %-22s workers=%-2d %s\n", n, d.name, w, st)
				}
			}
		}
	}

	// pick the faster dot product for the end-to-end runs
	probe := randomIndex(2000, 512, 3)
	best := dots[0]
	if timeIt(2, 20, func() { probe.Search(probe.data[:512], 5, 1, dots[1].fn) }).mean <
		timeIt(2, 20, func() { probe.Search(probe.data[:512], 5, 1, dots[0].fn) }).mean {
		best = dots[1]
	}

	models := map[string]*Model{}
	load := func(name string) *Model {
		if m, ok := models[name]; ok {
			return m
		}
		t := *threads
		if sp, _ := specFor(name); sp.Kind != "rec" && *detThreads > 0 {
			t = *detThreads
		}
		m, err := LoadModel(*modelDir, name, t, *spin)
		if err != nil {
			fmt.Fprintf(os.Stderr, "load %s: %v\n", name, err)
			os.Exit(1)
		}
		models[name] = m
		return m
	}
	indexes := map[int]*Index{}
	indexFor := func(d int) *Index {
		if ix, ok := indexes[d]; ok {
			return ix
		}
		indexes[d] = randomIndex(*gallery, d, 11)
		return indexes[d]
	}
	embDim := func(m *Model) int { s := m.outShapes[0]; return int(s[len(s)-1]) }

	if want["e2e"] {
		section(fmt.Sprintf("End to end, one request at a time: %d KB JPEG -> top-5 of %d (dot: %s)", len(frame)/1024, *gallery, best.name))
		fmt.Println("stage means in ms: decode / resize / detect / decode-boxes / align / embed / search")
		for _, c := range strings.Split(*combos, ",") {
			dn, rn, _ := strings.Cut(strings.TrimSpace(c), "+")
			det, rec := load(dn), load(rn)
			p, err := newPipeline(det, rec, indexFor(embDim(rec)), frame, 4, *searchWorkers, best.fn)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
			var sum stageTimes
			var totals []time.Duration
			for i := 0; i < 10+*iters; i++ {
				st, err := p.run()
				if err != nil {
					fmt.Fprintln(os.Stderr, "pipeline:", err)
					os.Exit(1)
				}
				if i < 10 {
					continue
				}
				totals = append(totals, st.total)
				sum.decode += st.decode
				sum.pre += st.pre
				sum.det += st.det
				sum.post += st.post
				sum.warp += st.warp
				sum.rec += st.rec
				sum.search += st.search
				sum.faces, sum.best = st.faces, st.best
			}
			n := time.Duration(*iters)
			fmt.Printf("%-30s %s\n", c, summarize(totals))
			fmt.Printf("%-30s %.1f / %.1f / %.1f / %.2f / %.2f / %.1f / %.2f", "", ms(sum.decode/n), ms(sum.pre/n), ms(sum.det/n), ms(sum.post/n), ms(sum.warp/n), ms(sum.rec/n), ms(sum.search/n))
			if *imagePath != "" && det.Spec.Kind == "scrfd" {
				fmt.Printf("   faces=%d best score=%.2f", sum.faces, sum.best.Score)
			}
			fmt.Println()
		}
		fmt.Printf("process RSS now: %.0f MB (all models above + %d-face index loaded)\n", rssMB(), *gallery)
	}

	if want["conc"] {
		dn, rn, _ := strings.Cut(*concCombo, "+")
		det, rec := load(dn), load(rn)
		ix := indexFor(embDim(rec))
		section(fmt.Sprintf("Concurrent requests, %s, %d ORT threads shared", *concCombo, *threads))
		for _, cs := range strings.Split(*conc, ",") {
			c, err := strconv.Atoi(strings.TrimSpace(cs))
			if err != nil || c < 1 {
				continue
			}
			var mu sync.Mutex
			var all []time.Duration
			var wg sync.WaitGroup
			deadline := time.Now().Add(time.Duration(*concSecs) * time.Second)
			for w := 0; w < c; w++ {
				// under concurrency each request stays on one core for the
				// pure-Go stages; ORT's pool is the shared resource
				p, err := newPipeline(det, rec, ix, frame, 1, 1, best.fn)
				if err != nil {
					fmt.Fprintln(os.Stderr, err)
					os.Exit(1)
				}
				wg.Add(1)
				go func() {
					defer wg.Done()
					var mine []time.Duration
					for i := 0; time.Now().Before(deadline); i++ {
						st, err := p.run()
						if err != nil {
							fmt.Fprintln(os.Stderr, "pipeline:", err)
							return
						}
						if i >= 3 {
							mine = append(mine, st.total)
						}
					}
					mu.Lock()
					all = append(all, mine...)
					mu.Unlock()
				}()
			}
			wg.Wait()
			if len(all) == 0 {
				continue
			}
			fmt.Printf("in-flight=%-2d  %6.1f req/s  %s\n", c, float64(len(all))/float64(*concSecs), summarize(all))
		}
		fmt.Printf("process RSS now: %.0f MB\n", rssMB())
	}
}
