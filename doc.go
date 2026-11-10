// Package likeness recognises faces: it finds the faces in a picture, judges
// whether each is good enough to work with, and turns it into a vector that
// can be compared with the vectors of other faces.
//
// An Engine does the work. Open one, give it encoded pictures, and read the
// faces it reports:
//
//	engine, err := likeness.Open(likeness.Config{})
//	if err != nil {
//		return err
//	}
//	defer engine.Close()
//
//	result, err := engine.Analyze(ctx, jpegBytes, likeness.ForSearch())
//	if err != nil {
//		return err // the picture could not be read
//	}
//	face, reason := result.Best()
//	if reason != likeness.Accepted {
//		return fmt.Errorf("no usable face: %s", reason)
//	}
//	// face.Embedding is a unit vector; the dot product of two of them is
//	// their similarity.
//
// Two faces of the same person have a high similarity, two different people
// a low one; where to draw the line depends on the models in use and on how
// costly each kind of mistake is.
//
// The neural networks run in ONNX Runtime, which is loaded as a shared
// library when an Engine is opened. Model files and the library live in a
// directory outside the program (see Config.Home); none of them are part of
// this module.
//
// The package is under construction; see the README for what exists today.
package likeness

// Version is the release this source tree belongs to.
const Version = "0.1.0-dev"
