# likeness

A face recognition server and Go library: send a picture, get back who it is.
One binary, no Python, no OpenCV, and a search answer in well under 100 ms on an
ordinary 8-core server without a GPU.

> **Status: early development.** The server runs, and so do the tools around
> it: `likeness setup` fetches the models, `serve` answers search and
> verification requests over HTTP, `enrol` imports people in bulk, `eval`
> measures accuracy on your own pictures, `bench` and `load` measure speed.
> What is missing is experience: nothing has run in production yet, and the
> permissively licensed models are good at confirming who someone is but not
> at picking one person out of thousands ([docs/accuracy.md](docs/accuracy.md)).

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
- [x] HTTP API: search, verify, enrol, manage (`likeness serve`,
      [docs/api.md](docs/api.md))
- [x] CLI: `setup`, `enrol`, `eval`, `bench`, `load`
- [x] a Go client for the API (package `client`)
- [x] accuracy measured on public data ([docs/accuracy.md](docs/accuracy.md))
- [x] a container image (`Dockerfile`)
- [ ] a stronger permissively licensed embedder

Liveness is judged from a single frame; a challenge the person has to
answer, GPU execution providers and an approximate index for millions of
faces are out of scope for 0.1.

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

## Serving

```sh
export LIKENESS_ADMIN_KEYS=$(openssl rand -hex 32)
export LIKENESS_SEARCH_KEYS=$(openssl rand -hex 32)
likeness serve --store faces.db
```

```sh
# enrol someone
curl -H "Authorization: Bearer $LIKENESS_ADMIN_KEYS" -F image=@anna.jpg \
     localhost:8080/v1/subjects/anna/faces
# who is this?
curl -H "Authorization: Bearer $LIKENESS_SEARCH_KEYS" -F image=@someone.jpg \
     localhost:8080/v1/search
```

```json
{
  "faces": [{
    "match": true,
    "matches": [{"subject": "anna", "score": 0.9405, "template": 1}],
    "margin": 0.8466,
    "box": {"left": 179.8, "top": 78, "right": 236.9, "bottom": 158.8},
    "quality": {"score": 0.944, "eye_distance": 28.4, "yaw": -4.4, "...": "..."}
  }],
  "threshold": 0.6,
  "timing_ms": {"read": 0.08, "queue": 0, "decode": 2.02, "detect": 12.21,
                "align": 0.28, "embed": 8.07, "search": 0.004, "total": 22.68}
}
```

That answer is from an Apple M1 Max, for a 383x480 JPEG and the default
models. The whole API is described in [docs/api.md](docs/api.md); in short:

- **Search, verify, enrol.** Search names the enrolled person in a picture,
  verify checks a picture against one claimed person, enrol adds a reference
  picture after checking that it is good enough to be one. An enrolment
  that would give one face to two people, or two unlike faces to one, is
  refused unless you insist.
- **A photograph is not a person.** Every face is also judged for being
  live, from the frame it came in. A picture of an enrolled person held up
  to the camera is turned down as `spoof` and compared with nobody.
- **It refuses rather than queues.** Only a few pictures are analysed at a
  time; a request that would wait more than a moment gets a `429` at once.
  Under more load than it can carry, the server answers fewer requests, not
  all of them late.
- **Keys with three levels of rights**: enrolling and removing people,
  recognising them, reading the metrics. Nothing is served without a key
  unless you ask for that by name.
- **Nothing about people leaks by default.** A search names nobody below
  the threshold; logs carry request ids, outcomes and timings, not subject
  ids; pictures never touch the disk.
- **It can be operated.** Prometheus metrics, a timing breakdown in every
  answer, health endpoints, graceful shutdown, TLS, secrets read from files.

## The command line

```sh
likeness setup                      # fetch the runtime library and the default models
likeness enrol --store faces.db staff/   # staff/<id>/*.jpg: one directory per person
likeness serve --store faces.db
```

| Command | |
|---|---|
| `setup` | downloads the ONNX Runtime library and a preset's models and checks every file against its SHA-256. `--list` prints the URLs and `--from DIR` installs from files carried over, for machines without internet access. |
| `enrol` | enrols many people from pictures on disk straight into a store, under the same rules as the API, and writes a CSV saying what became of every picture. Running it again adds nothing twice. |
| `serve` | the HTTP API. |
| `eval` | takes a directory of labelled pictures (`person/picture.jpg`, the layout of LFW) and reports false match and false non-match rates, how often a search names the wrong person, and the thresholds to use for a gallery of a given size. |
| `bench` | times each stage of a request on this machine and says how many pictures it can analyse at once: what to set `--max-inflight` to. |
| `load` | sends searches to a running server at rising rates and shows where it starts to refuse and what the accepted requests experience. |

Each explains itself with `-h`.

### In a container

```sh
docker build -t likeness .
docker run -p 8080:8080 -v likeness-data:/data \
    -e LIKENESS_ADMIN_KEYS=... -e LIKENESS_SEARCH_KEYS=... likeness
```

The image carries the runtime library and the default models, so it can be
built where there is internet access and run where there is none. It runs
as an unprivileged user and keeps the store in `/data`. The other commands
are there too: `docker run likeness bench`.

## Calling it from Go

```go
c, err := client.New("https://faces.internal:8080", client.WithKey(key))
if err != nil {
	return err
}
res, err := c.Search(ctx, jpegBytes, client.SearchOptions{})
if err != nil {
	return err // a *client.Error with the API's code and the request id
}
if m := res.Best(); m != nil {
	// m.Subject is in the picture, with similarity m.Score
}
```

Package `client` needs only the standard library. It sends a request again
when the server was too busy to take it, and never an enrolment that may
already have gone through.

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
- Searching thousands of people is beyond the default models. Measured on
  LFW ([docs/accuracy.md](docs/accuracy.md)), verification against one
  claimed person is sound: one stranger in ten thousand passes and 2% of
  genuine attempts fail. A search among 4 600 people at the default
  threshold misses 17% of the enrolled. Where people can say who they are
  first, verify; and run `likeness eval` on your own pictures before you
  rely on a match.
- One server per store file: there is no shared store for replicas yet.

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
make build    # bin/likeness
make test     # unit tests with the race detector
make fuzz     # a short fuzzing pass over the image decoders
make bench    # micro-benchmarks
```

## Models and licences

likeness itself is Apache-2.0. Model weights have their own licences and are
never bundled. The default preset uses OpenCV Zoo models (MIT and Apache-2.0).
InsightFace models are supported but are licensed for non-commercial research
only; the tool asks you to accept that before downloading them.

Face recognition processes biometric data. Check the law that applies to you
before deploying it, and store only what you need.
