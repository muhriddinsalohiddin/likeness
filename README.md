# likeness

A face recognition server and Go library: send a picture, get back who it is.
One binary, no Python, no OpenCV, and a search answer in well under 100 ms on an
ordinary 8-core server without a GPU.

> **Status: early development.** Three pieces work as Go libraries: the
> engine, which reads a picture, finds the faces, checks their quality and
> returns vectors; the index, which finds the enrolled people most similar to
> a vector; and the store, which keeps them on disk. The HTTP API that ties
> them into a service is being built next — see the roadmap below.

## Why

Most open face recognition stacks are a Python service wrapped around a deep
learning framework. They work, but they idle at gigabytes of memory, take
hundreds of milliseconds per request on CPU and are awkward to ship. likeness
keeps the parts that need a neural network inside ONNX Runtime and does
everything else — decoding, resizing, alignment, search, storage, HTTP — in
plain Go, with each stage measured.

## Measured so far

A throwaway benchmark (`bench/`) ran the whole request path — JPEG decode,
detection, alignment, embedding, exact search over 100 000 faces — before any
production code was written. One request at a time, 1280x720 JPEG:

| Machine | Light models (YuNet + SFace) | ResNet50 embedder |
|---|---|---|
| Apple M1 Max, 8 threads | 28 ms p50 / 29 ms p99 | 55 / 62 ms |
| Xeon Gold 6258R VM, 8 vCPU | 46 ms p50 / 56 ms p99 | 104 / 129 ms |

On the Xeon the light models sustain about 50 requests per second with p99 under
100 ms. Full numbers and what they taught us are in
[bench/RESULTS.md](bench/RESULTS.md).

## Design

- **Model-agnostic.** A detector and an embedder are described by a small
  manifest (input size, normalisation, output layout). No weights live in this
  repository; you choose a preset and the tool downloads and verifies it.
- **Exact search.** Up to 100 000 identities are scanned with SIMD dot products,
  so results are exact and there is no index to tune.
- **Embedded storage.** Runs without an external database; PostgreSQL support
  for multi-replica setups comes later.
- **Honest latency.** Every response carries a per-stage timing breakdown, and
  `likeness bench` measures the same stages on your own hardware.
- **Careful with inputs.** Image dimensions are checked before decoding, EXIF
  orientation is honoured, large faces are pre-filtered before they are shrunk.

## Roadmap to 0.1

- [x] image decoding, tensor sampling, alignment (`internal/vision`, `internal/align`)
- [x] ONNX Runtime binding, model manifests and downloads, SCRFD and YuNet
      decoders, embedders (`internal/ortx`, `internal/models`, `internal/detect`,
      `internal/embed`)
- [x] engine: detect, quality checks, align, embed (package `likeness`)
- [x] exact index with multiple templates per identity (package `index`)
- [x] embedded store with optional encryption at rest (package `store`)
- [ ] HTTP API: search, verify, enrol, manage
- [ ] CLI: `setup`, `serve`, `bench`, `eval`

Liveness detection, GPU execution providers and an approximate index for
millions of faces are out of scope for 0.1.

## Using the engine

```go
engine, err := likeness.Open(likeness.Config{}) // the default models
if err != nil {
	return err
}
defer engine.Close()

result, err := engine.Analyze(ctx, jpegBytes, likeness.ForSearch())
if err != nil {
	return err // not a readable picture
}
face, reason := result.Best()
if reason != likeness.Accepted {
	return fmt.Errorf("no usable face: %s", reason)
}
// face.Embedding is a unit vector: the dot product of two is their similarity
```

`ForSearch` is lenient, `ForEnrolment` strict: a reference picture has to be a
sharp, well-lit, frontal face with at least 40 pixels between the eyes, alone
in the frame. A face that falls short is not embedded, and the reason comes
back as one of `no_face`, `multiple_faces`, `face_too_small`, `bad_pose`,
`blurry`, `too_dark`, `too_bright`, `out_of_frame` or `low_det_score`, along
with the measurements behind it. Every result carries the time each stage
took.

On an Apple M1 Max with four threads, one call on a 1280x720 JPEG takes 26 ms
with the default models and 15 ms with the InsightFace `buffalo_s` pair
(`go test -bench Analyze`). The benchmark frame is a portrait on a plain
background; a busy scene costs a few more milliseconds of JPEG decoding.

## Searching

```go
ix, err := index.New(engine.Info().Dim, index.Options{})
if err != nil {
	return err
}
defer ix.Close()

ix.Add("employee-1042", 1, face.Embedding) // subject, template id, vector
matches, err := ix.Search(probe.Embedding, 5)
// matches[0].Subject, matches[0].Score: the most similar person
```

The search is exact: every stored vector is compared with the query, so
there is nothing to tune and no recall to lose. A person can have several
templates and is reported once, by the best of them. Dot products run in a
SIMD kernel (AVX2 and FMA on x86-64, NEON on arm64) over memory kept outside
the Go heap, and adding a template never makes a search wait.

On an Apple M1 Max:

| Enrolled vectors | 128 values (default models) | 512 values (InsightFace) |
|---|---|---|
| 7 000 | 0.11 ms | 0.22 ms |
| 100 000 | 0.56 ms | 2.0 ms |

Checking one person against a probe (`Compare`) takes under a microsecond.

## Keeping it

```go
st, err := store.Open("faces.db", store.Options{
	Dim:         engine.Info().Dim,
	Fingerprint: engine.Info().Fingerprint,
})
if err != nil {
	return err
}
defer st.Close()

tm, err := st.Add(ctx, "employee-1042", face.Embedding, nil) // on disk when it returns
ix.Add(tm.Subject, tm.ID, face.Embedding)

// when the program starts again: the index is rebuilt from the store
st.Load(ctx, func(tm store.Template) error {
	return ix.Add(tm.Subject, tm.ID, tm.Vector)
})
```

The store is one file and the source of truth; the index is rebuilt from it
at start-up, which takes 75 ms for 100 000 people on an M1 Max.

- **It knows its model.** Vectors made by different models cannot be
  compared. The store records which model made its vectors and refuses to
  open for another one, instead of quietly returning wrong matches.
- **It survives being killed.** A change is durable when the call returns,
  and a batch lands whole or not at all. The test suite kills a writer
  mid-operation, over and over, and checks every confirmed template.
- **It can be encrypted.** With a 32-byte key, vectors and metadata are
  sealed with AES-256-GCM, each record bound to its place so that swapping
  two records is detected. Subject ids stay readable; use opaque ones if they
  are sensitive.
- **Pictures are not kept.** Only vectors and whatever small metadata the
  caller attaches.

### Known limits

- The default detector (YuNet) reports landmarks as if the head were upright,
  so a head leaning more than about twenty degrees is aligned badly and its
  vector drifts; at forty degrees it is of little use. The InsightFace
  detectors follow the lean and do not have this problem. A permissively
  licensed detector that does the same is on the list.
- The default detector's file has a fixed 640x640 input and is the slowest
  stage of the default pair; the InsightFace detectors run at 320x320 first.
- Small faces are fragile: with 25 pixels between the eyes, ordinary JPEG
  compression alone moves the vector noticeably. Ask for close-ups.
- Head pose is estimated from five points and a generic head. Yaw is usable,
  pitch is not much more than a hint.

## Verified against the originals

The detectors and embedders are checked against the implementations they
come from: OpenCV's `FaceDetectorYN` and `FaceRecognizerSF`, and InsightFace's
SCRFD and ArcFace pipelines. On the test pictures, boxes and landmarks agree
to within 0.4 pixel, scores to within 0.001, and embeddings have a cosine
similarity of at least 0.9998 with the reference vectors. The reference
results and the script that produces them live in `internal/fixtures`.

## Building

Go 1.26 or newer and a C compiler (the ONNX Runtime binding uses cgo). The
runtime library itself is loaded when the program starts, not at link time;
tests that need it, and the model files, skip when they are not installed.

```sh
make test     # unit tests with the race detector
make fuzz     # a short fuzzing pass over the image decoders
make bench    # micro-benchmarks of the image pipeline
```

## Models and licences

likeness itself is Apache-2.0. Model weights have their own licences and are
never bundled. The default preset uses OpenCV Zoo models (MIT and Apache-2.0).
InsightFace models are supported but are licensed for non-commercial research
only; the tool asks you to accept that before downloading them.

Face recognition processes biometric data. Check the law that applies to you
before deploying it, and store only what you need.
