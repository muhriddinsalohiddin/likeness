# Third-party material in this repository

- `internal/ortx/onnxruntime_c_api.h`, `onnxruntime_ep_c_api.h`,
  `onnxruntime_error_code.h`: the ONNX Runtime C API headers, copyright
  Microsoft Corporation, MIT License. Only the headers are included; the
  runtime library is downloaded separately under its own licence.
- `internal/fixtures/faces/` and `internal/fixtures/closeups/`: photographs
  from the NASA Image and Video Library, works of the United States
  government not subject to copyright. See `internal/fixtures/README.md` for
  the source of each picture.
- `internal/ortx/testdata/*.onnx`: three-node graphs generated for the tests
  with the `onnx` Python package; no trained weights.

- `internal/liveness/testdata/`: `export.py` converts the Silent-Face
  anti-spoofing models to ONNX and records reference outputs with the code
  of https://github.com/minivision-ai/Silent-Face-Anti-Spoofing (Minivision,
  Apache-2.0), which it downloads; the crops and scores it wrote are derived
  from the NASA photographs above.

Model weights are never part of this repository. `internal/models` lists
where each model comes from and under which licence. The two anti-spoofing
models are attached to the `models-1` release of this project: the weights
are Minivision's, Apache-2.0, unchanged apart from their conversion to ONNX.
