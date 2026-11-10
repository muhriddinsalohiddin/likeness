# Test fixtures

Photographs used by the parity and integration tests. All four come from the
NASA Image and Video Library (images.nasa.gov) and are works of the United
States government, which are not subject to copyright. They were downscaled
to 480 pixels and converted to PNG so that every decoder sees identical
pixels.

| File | NASA ID | Content |
|---|---|---|
| `faces/hathaway.png` | jsc2025e068207_alt | official portrait of NASA astronaut Jack Hathaway |
| `faces/cassada.png` | jsc2022e064956_alt | official portrait of NASA astronaut Josh Cassada |
| `faces/adenot.png` | jsc2025e073950 | official portrait of ESA astronaut Sophie Adenot (NASA photo) |
| `faces/crew11.png` | jsc2025e011960_alt | official portrait of the four members of SpaceX Crew-11 |

NASA asks that its imagery not be used to imply endorsement, and the people
shown keep their rights of publicity. The pictures are here only so that the
code can be tested against known faces; do not reuse them for anything else.

Each picture also exists as a JPEG (`faces/*.jpg`), encoded by OpenCV at
quality 85 from the PNG. The JPEG is the production input format, and its
decoder differs between implementations, so embeddings computed from it are
kept as separate references.

`reference/` holds detector and embedder outputs for these pictures computed
by independent implementations (OpenCV's FaceDetectorYN and FaceRecognizerSF,
and a NumPy port of InsightFace's pipeline). `reference/generate.py`
reproduces them, including the JPEG files.
