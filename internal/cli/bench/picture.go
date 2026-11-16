package bench

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"os"

	"github.com/muhriddinsalohiddin/likeness/internal/fixtures"
)

// picture is what the benchmark analyses.
type picture struct {
	// name is how the report calls it: the file, or the built-in sample.
	name string
	data []byte
}

// load reads the picture in file, or builds the sample if there is none.
func load(file string) (picture, error) {
	if file == "" {
		data, err := sample()
		if err != nil {
			return picture{}, fmt.Errorf("the built-in sample: %w", err)
		}
		return picture{name: "built-in sample", data: data}, nil
	}
	data, err := os.ReadFile(file)
	if err != nil {
		return picture{}, fmt.Errorf("--image: %w", err)
	}
	return picture{name: file, data: data}, nil
}

// The sample is the sort of frame a webcam sends: one person in the middle
// of a 1280x720 picture.
const (
	sampleWidth   = 1280
	sampleHeight  = 720
	sampleQuality = 90
)

// sample builds the built-in picture: a close-up on a plain grey canvas,
// as a JPEG. The decoder and the detector have a whole frame to work
// through, as they do in service, and one face to find in it.
func sample() ([]byte, error) {
	closeup, err := fixtures.Closeup("hathaway")
	if err != nil {
		return nil, err
	}
	face, err := jpeg.Decode(bytes.NewReader(closeup))
	if err != nil {
		return nil, err
	}
	frame := image.NewRGBA(image.Rect(0, 0, sampleWidth, sampleHeight))
	draw.Draw(frame, frame.Bounds(), image.NewUniform(color.RGBA{110, 110, 110, 255}), image.Point{}, draw.Src)
	size := face.Bounds().Size()
	corner := image.Pt((sampleWidth-size.X)/2, (sampleHeight-size.Y)/2)
	draw.Draw(frame, image.Rectangle{Min: corner, Max: corner.Add(size)}, face, face.Bounds().Min, draw.Src)

	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, frame, &jpeg.Options{Quality: sampleQuality}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
