# HTTP API

`likeness serve` puts the engine, the index and the store behind a small
HTTP API: enrol a person from a picture, then ask who is in another picture
(search) or whether it shows a given person (verify).

```sh
export LIKENESS_ADMIN_KEYS=$(openssl rand -hex 32)
export LIKENESS_SEARCH_KEYS=$(openssl rand -hex 32)
likeness serve --store faces.db

curl -H "Authorization: Bearer $LIKENESS_ADMIN_KEYS" \
     -F image=@anna.jpg localhost:8080/v1/subjects/anna/faces
curl -H "Authorization: Bearer $LIKENESS_SEARCH_KEYS" \
     -F image=@someone.jpg localhost:8080/v1/search
```

- [Conventions](#conventions)
- [Sending a picture](#sending-a-picture)
- [Search](#search) · [Verify](#verify) · [Enrol](#enrol)
- [Subjects](#subjects) · [Info](#info)
- [Health and metrics](#health-and-metrics)
- [When the server is busy](#when-the-server-is-busy)
- [Errors](#errors) · [Why a face is turned down](#why-a-face-is-turned-down)
- [Running it](#running-it) · [What is kept and what is logged](#what-is-kept-and-what-is-logged)

## Conventions

Everything is under `/v1`. Answers are JSON, with `snake_case` names, and are
marked `Cache-Control: no-store`.

**Keys.** Every request carries an API key:

```
Authorization: Bearer <key>
```

There are three kinds of key, set in the environment of the server as
comma-separated lists:

| Variable | A key from it may |
|---|---|
| `LIKENESS_ADMIN_KEYS` | do everything: enrol and remove people, and all of the below |
| `LIKENESS_SEARCH_KEYS` | search, verify and read `/v1/info` |
| `LIKENESS_METRICS_KEYS` | read `/metrics`, nothing else |

A key is at least 16 characters of visible ASCII; use long random ones. The
server keeps only their hashes and compares them in constant time. Without a
key the answer is `401`, with a key that may not do what was asked `403`.
`/healthz` and `/readyz` need no key.

**Request ids.** Every answer has an `X-Request-Id` header, and the server's
log line for the request carries the same id. If the request brings its own
`X-Request-Id` (up to 64 letters, digits, `-`, `_` or `.`) that one is used,
so a request can be followed from your service into this one.

**Options** go in the query string. A request that sends its picture in a
form may put them in the form instead. An option the endpoint does not know
is an error rather than ignored: a misspelt `replace` should not quietly add
a template where you meant to swap one.

**Subject ids** are yours to choose: 1 to 128 bytes of UTF-8 without spaces,
control characters or slashes. Percent-encode them in URLs. They are stored
in the clear and appear in URLs, so prefer opaque ids to personal numbers.

## Sending a picture

A picture is a JPEG, PNG or WebP file, sent in one of two ways:

- as the body of the request, as it is (`Content-Type: image/jpeg`; the type
  is not checked, the bytes are);
- as the `image` field of a `multipart/form-data` body, which can carry the
  options of the request as further fields.

```sh
curl -H "Authorization: Bearer $KEY" -H 'Content-Type: image/jpeg' \
     --data-binary @face.jpg 'localhost:8080/v1/search?limit=3'
curl -H "Authorization: Bearer $KEY" -F image=@face.jpg -F limit=3 localhost:8080/v1/search
```

The body may be up to 10 MiB (`--max-body`), the picture up to 25 megapixels
and 8192 pixels on a side. EXIF orientation is honoured. Pictures are held
in memory only while the request runs; nothing is written to disk.

Every answer to a request with a picture says where the time went, in
milliseconds, in `timing_ms` and in a `Server-Timing` header:

```json
"timing_ms": {"read": 0.08, "queue": 0, "decode": 2.02, "detect": 12.21,
              "align": 0.28, "embed": 8.07, "search": 0.004, "total": 22.68}
```

`read` is receiving the upload, `queue` waiting for a free slot (see
[below](#when-the-server-is-busy)), `decode` to `embed` the work on the
picture, and the last stage before `total` what the endpoint did with the
result: `search`, `compare` or `store`.

## Search

`POST /v1/search` — who is in this picture? Needs a search key.

| Option | Default | |
|---|---|---|
| `limit` | 1 | how many subjects to name, best first; up to 50 |
| `threshold` | the server's | a stricter similarity to require; it can be lowered only on a server started with `--allow-candidates` |
| `faces` | `largest` | `largest` looks at the person at the camera, `all` at up to 16 faces |
| `candidates` | `false` | name the nearest subjects even below the threshold; only on a server started with `--allow-candidates` |

```json
{
  "faces": [
    {
      "box": {"left": 179.8, "top": 78, "right": 236.9, "bottom": 158.8},
      "quality": {"score": 0.944, "eye_distance": 28.4, "yaw": -4.4, "pitch": 13.2,
                  "roll": 0, "in_frame": 1, "brightness": 142, "sharpness": 41.3},
      "match": true,
      "matches": [{"subject": "hathaway", "score": 0.9405, "template": 1}],
      "margin": 0.8466
    }
  ],
  "faces_found": 1,
  "threshold": 0.6,
  "width": 383,
  "height": 480,
  "timing_ms": {"read": 0.08, "queue": 0, "decode": 2.02, "detect": 12.21,
                "align": 0.28, "embed": 8.07, "search": 0.004, "total": 22.68}
}
```

For each face:

- `match` says whether the face is somebody enrolled, and `matches[0]` is
  who. `matches` holds up to `limit` subjects whose similarity reaches the
  threshold, each with their best template; it is empty when nobody does.
- `score` is a cosine similarity: 1 for the same picture, around 0 for two
  unrelated people.
- `margin` is how far the best subject is ahead of the next one. A match
  with a small margin deserves less trust than its score suggests: someone
  else enrolled looks nearly as much like the face. It is absent when fewer
  than two people are enrolled or when nobody matched.
- `box` is in pixels of the upright picture. `quality` is described
  [below](#why-a-face-is-turned-down).

A picture in which nobody is recognised is still a `200`: `match` is false.
If a face is found but is too poor to compare, the face has a `reject`
reason and no matches; if no face is found at all, `faces` is empty and the
answer has `"reject": "no_face"`.

Subjects below the threshold are never named unless the server allows
candidates. Without that rule a search key would be a way to learn who
resembles whom.

## Verify

`POST /v1/verify` — is this a picture of that subject? Needs a search key.

| Option | Default | |
|---|---|---|
| `subject` | required | whom to compare with |
| `threshold` | the server's | a stricter similarity to require |

```json
{
  "subject": "cassada",
  "verified": false,
  "score": 0.0945,
  "template": 2,
  "threshold": 0.363,
  "face": {"box": {"left": 150.6, "top": 116.2, "right": 295.5, "bottom": 326.8},
           "quality": {"score": 0.953, "eye_distance": 69.8, "yaw": -3.1, "pitch": 10.9,
                       "roll": -0.3, "in_frame": 1, "brightness": 143.8, "sharpness": 298.6}},
  "timing_ms": {"read": 0.03, "queue": 0, "decode": 2.32, "detect": 13.34,
                "align": 1.01, "embed": 10.22, "compare": 0.002, "total": 26.94}
}
```

The largest face in the picture is compared with every template of the
subject, and the best one counts. If the picture has no usable face the
answer is `"verified": false` with a `reject` reason and neither `score`
nor `template`. A
subject that is not enrolled is a `404`, decided before the picture is
looked at.

Verification asks less than search does: comparing with one claimed person
leaves far less room for a chance resemblance than picking one out of
thousands, so its threshold is lower.

Sending `subject` as a form field keeps it out of the URL, and so out of
the access logs of anything between your service and this one.

## Enrol

`POST /v1/subjects/{id}/faces` — remember this face as that subject's. Needs
an admin key. The subject is created by its first face.

| Option | Default | |
|---|---|---|
| `replace` | `false` | drop the subject's earlier templates once the new one is in |
| `allow_duplicate` | `false` | enrol even if the face matches another subject |
| `allow_mismatch` | `false` | enrol even if the face does not look like the subject's earlier ones |

`201 Created`:

```json
{
  "subject": "hathaway",
  "template": 1,
  "templates": 1,
  "created": "2026-11-13T14:56:27.502936Z",
  "face": {"box": {"left": 150.6, "top": 116.2, "right": 295.5, "bottom": 326.8},
           "quality": {"score": 0.953, "eye_distance": 69.8, "yaw": -3.1, "pitch": 10.9,
                       "roll": -0.3, "in_frame": 1, "brightness": 143.8, "sharpness": 298.6}},
  "timing_ms": {"read": 0.09, "queue": 0, "decode": 2.24, "detect": 14.85,
                "align": 0.3, "embed": 9.25, "store": 10.36, "total": 37.42}
}
```

When the answer arrives the template is on disk and searchable. `template`
is its id, `templates` how many the subject holds now. A subject can hold
several (ten unless `--max-templates` says otherwise): pictures taken on
different days, with and without glasses, make it easier to recognise. A
search or verification uses the best of them.

A reference picture is held to a higher standard than a probe, since a poor
one spoils every later comparison. The picture has to show one person,
close up (at least 40 pixels between the eyes), facing the camera, sharp and
evenly lit. If it does not, nothing is enrolled and the answer is a `422`
that says why, with the measurements behind it:

```json
{
  "error": {
    "code": "face_rejected",
    "message": "the picture cannot serve as a reference: the face is too small; move closer to the camera",
    "reject": "face_too_small",
    "quality": {"score": 0.944, "eye_distance": 25.4, "yaw": -20, "pitch": 11.8, "roll": -1.7, "in_frame": 1}
  },
  "request_id": "6b3ece6a65865007"
}
```

Three checks guard the gallery, each answering `409`:

- `duplicate_face` — the face already belongs to another subject, named in
  `subject` with the similarity in `score`. One person with two ids is the
  usual cause. For twins, pass `allow_duplicate`.
- `subject_mismatch` — the subject has templates and the new face does not
  look like them (`score` says how little). Pass `allow_mismatch` if the
  person has changed, or `replace` to start afresh.
- `too_many_templates` — remove one, or `replace`.

Enrolment is not idempotent: sending the same picture twice adds two
templates. With `replace=true` a repeat does no harm.

## Subjects

All need an admin key.

`GET /v1/subjects/{id}` — a subject and its templates, oldest first, each
with the quality measured when it was enrolled. Vectors are not handed out.

```json
{
  "subject": "hathaway",
  "created": "2026-11-13T14:56:27.502936Z",
  "templates": [
    {"id": 1, "created": "2026-11-13T14:56:27.502936Z",
     "quality": {"score": 0.953, "eye_distance": 69.8, "yaw": -3.1, "pitch": 10.9,
                 "roll": -0.3, "in_frame": 1, "brightness": 143.8, "sharpness": 298.6}}
  ]
}
```

`GET /v1/subjects?after=&limit=` — everyone, in id order, `limit` at a time
(100 by default, at most 1000). When there may be more, the answer has
`next`: pass it as `after` to continue.

```json
{
  "subjects": [
    {"subject": "cassada", "created": "2026-11-13T14:56:27.548451Z", "templates": 1},
    {"subject": "hathaway", "created": "2026-11-13T14:56:27.502936Z", "templates": 1}
  ],
  "next": "hathaway"
}
```

`DELETE /v1/subjects/{id}` — remove a subject and all its templates.

```json
{"subject": "hathaway", "removed": 2, "templates": 0}
```

`DELETE /v1/subjects/{id}/faces/{template}` — remove one template.
`templates` is how many are left; a subject whose last template is removed
is gone.

```json
{"subject": "hathaway", "template": 1, "removed": 1, "templates": 1}
```

A removal is on disk and out of the index when the answer arrives.

## Info

`GET /v1/info` — what the server runs and holds. Needs a search key.

```json
{
  "version": "0.1.0-dev",
  "preset": "default",
  "detector": "yunet_2023mar",
  "embedder": "sface_2021dec",
  "fingerprint": "sface:sface_2021dec:0ba9fbfa01b5270c",
  "dim": 128,
  "runtime": "1.29.0",
  "threads": 10,
  "kernel": "neon",
  "thresholds": {"search": 0.6, "verify": 0.363},
  "gallery": {"subjects": 2, "templates": 2, "encrypted": false, "created": "2026-11-13T14:56:27.442125Z"},
  "limits": {"max_body": 10485760, "max_inflight": 2, "queue": 16, "queue_timeout_ms": 50,
             "max_templates": 10, "search_limit": 50, "candidates": false}
}
```

`fingerprint` identifies the model that made the vectors in the gallery. A
store only opens for the model it was made with.

## Health and metrics

- `GET /healthz` answers `200` while the process is alive.
- `GET /readyz` answers `200` while the server wants requests and `503` once
  it has been told to stop. A server is ready as soon as it listens: models
  are loaded and the gallery is in memory before it opens its port.
- `GET /metrics` is in the Prometheus text format and needs a metrics or
  admin key.

| Metric | |
|---|---|
| `likeness_requests_total{route,code}` | requests answered |
| `likeness_request_duration_seconds{route}` | histogram, arrival to answer |
| `likeness_stage_duration_seconds{stage}` | histogram per stage of handling a picture |
| `likeness_results_total{route,result}` | `match`, `no_match`, `rejected`; `verified`, `not_verified`; `enrolled`, `duplicate`, `mismatch`, `full` |
| `likeness_rejected_faces_total{route,reason}` | pictures turned down, by reason |
| `likeness_overloaded_total{cause}` | `429`s: `queue_full`, `queue_timeout`, `too_many_uploads` |
| `likeness_unauthorized_total{reason}` | `no_key`, `unknown_key`, `no_right` |
| `likeness_inflight_requests`, `likeness_queued_requests`, `likeness_pending_requests` | pictures being analysed, waiting, and anywhere in between |
| `likeness_subjects`, `likeness_templates` | size of the gallery |
| `likeness_build_info{version,preset,detector,embedder,kernel,go}` | always 1 |

Health checks and scrapes are not counted in the request metrics, and no
metric carries a subject id.

## When the server is busy

Analysing a picture uses every core it is given, so the server analyses only
a few at a time (`--max-inflight`, 2 by default). A request that finds them
all taken waits for its turn, but not for long. The server knows how long
pictures have been taking and how many requests are ahead; one that could
not expect a free slot within `--queue-timeout` (50 ms) is refused as it
arrives, before its picture is even read:

```
HTTP/1.1 429 Too Many Requests
Retry-After: 1

{"error": {"code": "overloaded", "message": "..."}, "request_id": "..."}
```

Nothing was done with a refused request, so it is always safe to send it
again. The point of refusing is that the requests which are accepted stay
fast: under twice the load it can carry, the server answers half the
requests in the usual time instead of all of them late.

`--queue-timeout` is the knob: the longest an accepted request waits, and
so the price in latency of riding out a burst instead of refusing part of
it. `--queue` (8 per slot) only bounds the line when that estimate has
nothing to go by yet.

`likeness bench` measures how many pictures this machine analyses at once
before each one slows down, and says what to set `--max-inflight` to.
`likeness load` drives a running server at rising rates and shows where it
starts to refuse and what the accepted requests experience:

```sh
likeness load --url http://localhost:8080 --steps 10,20,40,80,160 face.jpg
```

## Errors

An error is an HTTP status and a body of this shape:

```json
{"error": {"code": "subject_not_found", "message": "no such subject is enrolled"}, "request_id": "8f0c9d1e2a3b4c5d"}
```

`code` is stable and meant for programs; `message` is for the person reading
a log. Some codes bring further fields, described where they occur.

| Status | Code | |
|---|---|---|
| 400 | `bad_request` | the body or the query string is malformed |
| 400 | `invalid_parameter` | an option has a bad value, is given twice, or is not one the endpoint has |
| 400 | `missing_image` | no picture in the request |
| 400 | `missing_subject`, `invalid_subject` | verify without a subject; an id that breaks the rules |
| 401 | `unauthorized` | no API key, or one the server does not know |
| 403 | `forbidden` | the key may not do this |
| 403 | `candidates_disabled` | `candidates=true` on a server that does not allow it |
| 403 | `read_only` | the store was opened read-only: nothing can be enrolled or removed |
| 404 | `subject_not_found`, `template_not_found` | |
| 404 | `not_found` | no such path |
| 405 | `method_not_allowed` | the `Allow` header lists what the path answers to |
| 409 | `duplicate_face`, `subject_mismatch`, `too_many_templates` | see [Enrol](#enrol) |
| 413 | `body_too_large` | the request exceeds `--max-body` |
| 413 | `image_too_large` | the picture has too many pixels |
| 415 | `unsupported_image` | not a JPEG, PNG or WebP picture |
| 422 | `corrupt_image`, `image_too_small` | the picture is damaged, or under 16 pixels on a side |
| 422 | `face_rejected` | enrolment only: the picture cannot serve as a reference; `reject` says why |
| 429 | `overloaded` | see [above](#when-the-server-is-busy); retry after `Retry-After` seconds |
| 500 | `internal` | a fault in the server; its log has the details under the request id |
| 503 | `shutting_down` | the server is stopping |

## Why a face is turned down

`reject` is one of:

| Reason | |
|---|---|
| `no_face` | no face was found |
| `multiple_faces` | enrolment only: a second face rivals the first in size |
| `low_det_score` | the detector is not confident this is a face |
| `out_of_frame` | too much of the face is cut off by the edge of the picture |
| `face_too_small` | too few pixels between the eyes |
| `bad_pose` | the head is turned, tilted or leaning too far |
| `too_dark`, `too_bright` | the face is under- or over-exposed |
| `blurry` | the face lacks detail: motion, focus, or an upscaled picture |

and `quality` holds what was measured:

| Field | |
|---|---|
| `score` | the detector's confidence, 0 to 1 |
| `eye_distance` | pixels between the centres of the eyes; a face is about 2.3 times as wide |
| `yaw`, `pitch`, `roll` | rough head angles in degrees; expect errors of ten degrees, more for pitch |
| `in_frame` | the share of the face box that lies inside the picture |
| `brightness` | mean brightness of the face, 0 to 255 |
| `sharpness` | fine detail in the face relative to its brightness; a sharp close-up scores a few hundred, below 40 the detail is gone |

`brightness` and `sharpness` are absent when the face was turned down for
its size, position or pose, before they were measured.

| Limit | Search and verify | Enrolment |
|---|---|---|
| `score` at least | 0.5 | 0.7 |
| `eye_distance` at least | 20 px | 40 px |
| yaw, pitch, roll at most | 55°, 55°, any | 30°, 40°, 25° |
| `in_frame` at least | 0.6 | 0.95 |
| `brightness` | 15 to 240 | 50 to 215 |
| `sharpness` at least | 4 | 40 |

## Running it

`likeness serve -h` lists the flags. The ones that matter first:

| Flag | Default | |
|---|---|---|
| `--store` | required | the file that keeps the enrolled faces; made if it does not exist |
| `--listen` | `127.0.0.1:8080` | address to listen on |
| `--preset` | `default` | the models to use |
| `--search-threshold`, `--verify-threshold` | the model's | similarities from which a face counts as a subject's |
| `--max-inflight`, `--queue`, `--queue-timeout` | 2, 16, 50ms | see [When the server is busy](#when-the-server-is-busy) |
| `--tls-cert`, `--tls-key` | | serve HTTPS |
| `--drain-delay` | 0 | keep serving this long after a stop signal while `/readyz` says 503 |
| `--log-format`, `--log-level` | `text`, `info` | |

**Thresholds are starting points.** The defaults come from the models'
authors and from a handful of test pictures, not from your cameras and your
people. Before relying on a match for anything that matters, measure false
accepts and false rejects on your own pictures and set the thresholds from
that.

**Keys and the store key** come from the environment, never from flags,
which any user of the machine can read off the process list. Each variable
can name a file instead (`LIKENESS_ADMIN_KEYS_FILE` and so on), which is how
Docker and Kubernetes deliver secrets. `LIKENESS_STORE_KEY`, 32 bytes in hex
or base64, encrypts vectors and metadata at rest; a store made with a key
opens only with that key.

**The network.** By default the server listens on this machine only. Before
it listens anywhere else, give it TLS, either its own (`--tls-cert`) or that
of a proxy in front of it: pictures of faces and API keys should not cross a
network in the clear. The API is meant to be called by your backend, not by
browsers: a key that reaches a browser is no longer a secret, and the server
sends no CORS headers.

**Stopping.** On `SIGTERM` or `SIGINT` the server reports itself not ready,
waits `--drain-delay` for load balancers to notice, stops accepting
connections and lets the requests under way finish (up to 15 seconds). A
second signal stops it at once.

**One server per store.** The store file is locked by the process that
serves it. To run several replicas you need a store they can share, which
is on the roadmap.

## What is kept and what is logged

- Pictures are never written to disk and are dropped when the request ends.
- The store holds, per template, the vector, the time and the quality
  measured at enrolment; per subject, the id.
- The log has one line per request: time, request id, route, status,
  duration, size of the upload, client address, which key was used (as
  `admin-1`, never the key), the outcome (`match`, `rejected` and so on) and
  the stage timings. It does not contain pictures, vectors or, at the
  default level, subject ids; `--log-level debug` adds the subject.
- A vector is not a picture, but it is biometric data all the same: someone
  who has it can test pictures against it. Treat the store file and its
  backups accordingly.
