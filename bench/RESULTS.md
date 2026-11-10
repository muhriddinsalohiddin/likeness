# bench — latency spike results

Question: can one 1:N face search request (JPEG in, top-k identities out) be served in
under 100 ms on CPU, from Go?

Setup: Go 1.26 + ONNX Runtime 1.29 via `github.com/yalue/onnxruntime_go` (cgo). Everything
except inference is pure Go (JPEG decode from the standard library, fused
resize/colour/normalise straight from YCbCr into the input tensor, SCRFD decoding + NMS,
5-point similarity alignment, exact flat index). Gallery: 100 000 random 512-d vectors.
Input: synthetic photo-like 1280x720 JPEG, 200 KB.

## Apple M1 Max (8P+2E), 8 ORT threads, spinning disabled — 2026-09-21

Per stage, mean ms:

| Stage | ms | Notes |
|---|---|---|
| JPEG decode 1280x720 | 10.3 | 640x480: 3.5 · 1920x1080: 24 — second largest cost |
| resize -> tensor | 0.2–0.5 | fused, 4 goroutines |
| detect `det_500m` @320 | 4.5 | @640: 12.6 · YuNet @640: 10.5 · `det_10g` @640: 91 (ruled out) |
| box decode + NMS | <0.01 | |
| align + warp 112x112 | 0.18 | |
| embed `w600k_r50` (ResNet50) | 37 | `w600k_mbf`: 5.4 · SFace: 5.3 |
| search 100k x 512, exact | 3.1 | pure Go, 10 goroutines · 7k: 0.35 · single goroutine: 16 |

End to end, one request at a time:

| Detector + embedder | p50 | p99 | RSS (incl. 100k index) |
|---|---|---|---|
| `det_500m@320` + `w600k_r50` | 54.6 | 62.4 | 738 MB |
| `det_500m@320` + `w600k_mbf` | 23.5 | 24.6 | 492 MB |
| YuNet + SFace (permissive licences) | 27.5 | 29.4 | 289 MB |

Concurrent requests (each request single-threaded outside ORT):

| Preset | in-flight | req/s | p99 ms |
|---|---|---|---|
| r50 | 1 | 14 | 79 |
| r50 | 2 | 22 | 92 |
| r50 | 3 | 28 | 119 |
| YuNet + SFace | 4 | 102 | 47 |
| YuNet + SFace | 8 | 131 | 116 |
| `det_500m@320` + mbf | 4 | 92 | 60 |

### Pessimistic scenario: 4 cores (`GOMAXPROCS=4`, 4 ORT threads)

A stand-in for an 8-vCPU VM whose vCPUs are hyperthreads of 4 physical cores. Same
machine, so this bounds the effect of core count only, not of the x86 micro-architecture.

| Preset | p50 | p99 | Within 100 ms p99 |
|---|---|---|---|
| `det_500m@320` + `w600k_r50` | 73.9 | 77.5 | ~11 req/s (2 in flight already hits p99 124 ms) |
| YuNet + SFace | 26.3 | 32.6 | ~100 req/s (4 in flight, p99 41 ms) |

ResNet50-class fits on 4 cores with little headroom; a mid-size backbone (R18/R34-class)
is the likely default for CPU-only deployments. Decide with `bench` on the target machine.

## Intel Xeon Gold 6258R @ 2.70 GHz, VMware VM, 8 vCPU, AVX-512 + VNNI — 2026-09-21

Close to the production target (VMware, 8 vCPU Cascade Lake; production is a 6238R at
2.2 GHz without VNNI exposed, so expect it to be somewhat slower). Run in Docker, 8 ORT
threads, spinning disabled.

Per stage, mean ms (M1 Max in brackets):

| Stage | ms | Notes |
|---|---|---|
| JPEG decode 1280x720 | 17.4 (10.3) | 640x480: 5.9 · 1920x1080: 39 — now ~40% of the light preset |
| resize -> tensor, 4 goroutines | 0.4–1.5 | |
| detect `det_500m` @320 | 5.6 (4.5) | YuNet @640: 7.1 |
| align + warp | 0.6 | |
| embed `w600k_r50` | 72 (37) | isolated: 355 / 184 / 100 / 68 ms at 1 / 2 / 4 / 8 threads |
| embed `w600k_mbf` | 12.3 (5.4) | does not scale past 4 threads (9.5 -> 11.1 ms) |
| embed SFace | 16.5 (5.3) | isolated: 34 / 22 / 15 / 13 ms |
| search 100k x 512, AVX2, 8 goroutines | 7.8 (3.1) | single goroutine: 36 ms · 7k: 0.2–0.7 ms |

End to end, one request at a time:

| Detector + embedder | p50 | p99 | RSS (incl. 100k index) |
|---|---|---|---|
| YuNet + SFace | 45.8 | 56.3 | 237 MB |
| `det_500m@320` + `w600k_mbf` | 46.1 | 52.7 | 477 MB |
| `det_500m@320` + `w600k_r50` | 103.5 | 129.3 | 929 MB |

Concurrent requests (search and resize single-threaded per request, which costs ~30 ms
here — see note 10):

| Preset | in-flight | req/s | p99 ms |
|---|---|---|---|
| YuNet + SFace | 4 | 52 | 94 |
| `det_500m@320` + mbf | 4 | 50 | 87 |
| either light preset | 8 | 63–71 | ~200 |
| r50 | 1 | 7 | 175 |

What this machine adds to the picture:

- **The light presets meet the budget with room to spare; ResNet50 does not** on this
  CPU (72 ms for the embedder alone). An accurate CPU preset needs a mid-size backbone,
  int8 (VNNI is present here but not on production), or a GPU.
- **A Cascade Lake vCPU is ~2.3x slower than an M1 P-core for fp32 convolutions**, and
  scaling flattens after 4 threads (8 vCPUs behave like 5–6 cores).
- **JPEG decoding is 1.7x slower than on the M1** and is now the single largest stage of
  the light preset. Capturing at 640x480 saves ~12 ms; libjpeg-turbo would save more.

## Findings that shape the design

1. **ORT thread spinning hurts pipelines.** Each session owns a thread pool; after the
   detector finishes, its idle threads busy-wait and steal cores from the embedder and from
   Go's own goroutines. With default spinning: r50 p50 68 ms / p99 109 ms; with
   `session.intra_op.allow_spinning=0`: 55 / 62. A single global ORT thread pool would be
   better still, but `onnxruntime_go` does not expose it yet.
2. **InsightFace detector files declare static output shapes** (exported at 640). ORT
   rejects pre-allocated outputs for any other input size, so outputs must be
   ORT-allocated per run for those models.
3. **Keep the index off the Go heap.** A 205 MB `[]float32` plus GOGC=100 lets the heap
   grow to roughly twice that before a collection. An mmap-ed index file avoids it and
   gives instant restarts.
4. **JPEG decode is the second largest cost.** Cap input dimensions; consider an optional
   libjpeg-turbo build tag (DCT-domain downscaling) for large uploads.
5. **Detect at 320 first, retry at 640 only if nothing is found.** For login-style frames
   the face is large; 320 is 3x cheaper.
6. **The embedder dominates.** ResNet50-class costs ~37 ms on 8 fast cores. Faster
   options: an R18/R34-class model, int8 quantisation (needs an accuracy harness first),
   or a GPU execution provider.
7. **Admission control is mandatory.** One r50 request already uses every core, so
   latency grows linearly with in-flight requests. Bound concurrency, queue briefly,
   shed load beyond that.
8. `vek32` has no NEON path (slower than the unrolled Go loop on arm64). On the Xeon it is
   1.5x faster than pure Go, not more, because of the next point.
9. **The flat scan is memory-bandwidth-bound on the Xeon VM**, not compute-bound: 205 MB per
   query at roughly 6 GB/s per thread. Always scan in parallel; an int8 or fp16 copy of the
   index (scan compressed, re-rank the top candidates in fp32) is the lever for 100k+.
   At 7k identities the index fits in cache and the scan takes well under a millisecond.
10. **Disabling spinning costs small models in a VM.** Waking a sleeping vCPU is expensive,
    and depthwise networks have many tiny ops: SFace is 3.4x slower than on the M1 while
    ResNet50 is 1.9x slower. One shared ORT thread pool that may spin (no cross-session
    contention, no wake-ups) should recover several milliseconds; `onnxruntime_go` does not
    expose global thread pools, so this needs a small cgo addition.
11. The concurrency mode of this spike runs search and resize single-threaded per request,
    which is the wrong trade on the Xeon (36 ms scan). The server must keep the scan
    parallel under load.

## Running on the target server (linux/amd64)

    docker load < facebench-spike.tar.gz
    docker run --rm facebench:spike                 # uses all CPUs
    docker run --rm -e THREADS=4 facebench:spike    # cap ORT threads

The image bundles InsightFace weights, which are licensed for non-commercial research
only: use it for internal measurement and do not publish it.
