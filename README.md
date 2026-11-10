# likeness

A face recognition server and Go library: send a picture, get back who it is.
One binary, no Python, no OpenCV, and a search answer in well under 100 ms on an
ordinary 8-core server without a GPU.

> **Status: early development.** Decoding, alignment, detection and embedding
> are in place and verified against independent implementations. The search
> index, storage and the HTTP API are being built next — see the roadmap
> below. Nothing here is usable as a service yet.

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
- [ ] engine: detect, quality checks, align, embed
- [ ] exact index with multiple templates per identity
- [ ] embedded store with optional encryption at rest
- [ ] HTTP API: search, verify, enrol, manage
- [ ] CLI: `setup`, `serve`, `bench`, `eval`

Liveness detection, GPU execution providers and an approximate index for
millions of faces are out of scope for 0.1.

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
