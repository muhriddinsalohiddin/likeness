// Package fixtures holds the test pictures shared by several packages, with
// reference results computed by independent implementations.
package fixtures

import (
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"path"
	"sort"
	"strings"
)

//go:embed faces/*.png faces/*.jpg
var faces embed.FS

//go:embed closeups/*.jpg
var closeups embed.FS

//go:embed reference/*.json
var reference embed.FS

// Names lists the available pictures, without directory or extension.
func Names() []string {
	entries, err := fs.ReadDir(faces, "faces")
	if err != nil {
		panic(err)
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if path.Ext(e.Name()) == ".png" {
			names = append(names, strings.TrimSuffix(e.Name(), ".png"))
		}
	}
	sort.Strings(names)
	return names
}

// Picture returns the PNG bytes of a named picture.
func Picture(name string) ([]byte, error) {
	b, err := faces.ReadFile(path.Join("faces", name+".png"))
	if err != nil {
		return nil, fmt.Errorf("fixtures: no picture %q", name)
	}
	return b, nil
}

// JPEG returns the same picture as OpenCV encoded it at quality 85: the
// bytes the JPEG reference embeddings were computed from.
func JPEG(name string) ([]byte, error) {
	b, err := faces.ReadFile(path.Join("faces", name+".jpg"))
	if err != nil {
		return nil, fmt.Errorf("fixtures: no JPEG for %q", name)
	}
	return b, nil
}

// Closeups lists the head-and-shoulders pictures: one frontal face each,
// about 70 pixels between the eyes, the framing of someone at a webcam.
func Closeups() []string {
	entries, err := fs.ReadDir(closeups, "closeups")
	if err != nil {
		panic(err)
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		names = append(names, strings.TrimSuffix(e.Name(), path.Ext(e.Name())))
	}
	sort.Strings(names)
	return names
}

// Closeup returns the JPEG bytes of a named close-up.
func Closeup(name string) ([]byte, error) {
	b, err := closeups.ReadFile(path.Join("closeups", name+".jpg"))
	if err != nil {
		return nil, fmt.Errorf("fixtures: no close-up %q", name)
	}
	return b, nil
}

// Face is one detected face in a reference result.
type Face struct {
	// Box is x1, y1, x2, y2 in picture pixels.
	Box   [4]float32 `json:"box"`
	Score float32    `json:"score"`
	// Landmarks are left eye, right eye, nose, left mouth corner, right mouth
	// corner as seen in the picture, each as x, y.
	Landmarks [5][2]float32 `json:"landmarks"`
}

// Embedding is the L2-normalised feature of one face, computed from the
// landmarks it lists so that a test can reproduce exactly the same crop.
type Embedding struct {
	Landmarks [5][2]float32 `json:"landmarks"`
	Vector    []float32     `json:"vector"`
}

// Detection is a detector's output for one picture at one input size.
type Detection struct {
	InputSize int    `json:"input_size"` // side of the square network input
	Faces     []Face `json:"faces"`
}

// Reference is what an independent implementation computed for one picture.
type Reference struct {
	Picture string `json:"picture"`
	// Detections is keyed by model name, for example "yunet_2023mar".
	Detections map[string]Detection `json:"detections"`
	// Embeddings is keyed by model name, for example "sface_2021dec".
	Embeddings map[string][]Embedding `json:"embeddings"`
	// JPEGEmbeddings were computed from the JPEG version of the picture,
	// with the same landmarks as Embeddings.
	JPEGEmbeddings map[string][]Embedding `json:"jpeg_embeddings"`
}

// References loads every reference file.
func References() ([]Reference, error) {
	entries, err := fs.ReadDir(reference, "reference")
	if err != nil {
		return nil, err
	}
	var refs []Reference
	for _, e := range entries {
		if path.Ext(e.Name()) != ".json" {
			continue
		}
		b, err := reference.ReadFile(path.Join("reference", e.Name()))
		if err != nil {
			return nil, err
		}
		var r Reference
		if err := json.Unmarshal(b, &r); err != nil {
			return nil, fmt.Errorf("fixtures: %s: %w", e.Name(), err)
		}
		refs = append(refs, r)
	}
	return refs, nil
}
