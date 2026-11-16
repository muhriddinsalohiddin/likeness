package bench

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"testing"
)

func TestSample(t *testing.T) {
	pic, err := load("")
	if err != nil {
		t.Fatal(err)
	}
	if pic.name != "built-in sample" {
		t.Errorf("the sample is called %q", pic.name)
	}
	cfg, format, err := image.DecodeConfig(bytes.NewReader(pic.data))
	if err != nil || format != "jpeg" || cfg.Width != 1280 || cfg.Height != 720 {
		t.Fatalf("the sample is a %dx%d %s picture (%v), want a 1280x720 jpeg", cfg.Width, cfg.Height, format, err)
	}

	// grey around the edges, and something other than grey in the middle
	frame, err := jpeg.Decode(bytes.NewReader(pic.data))
	if err != nil {
		t.Fatal(err)
	}
	grey := func(x, y int) bool {
		c := color.RGBAModel.Convert(frame.At(x, y)).(color.RGBA)
		near := func(v uint8) bool { return v > 100 && v < 120 }
		return near(c.R) && near(c.G) && near(c.B)
	}
	for _, p := range []image.Point{{0, 0}, {1279, 0}, {0, 719}, {1279, 719}, {200, 360}, {1080, 360}} {
		if !grey(p.X, p.Y) {
			t.Errorf("the canvas is not grey at %v: %v", p, frame.At(p.X, p.Y))
		}
	}
	coloured := 0
	for y := 160; y < 560; y += 8 {
		for x := 440; x < 840; x += 8 {
			if !grey(x, y) {
				coloured++
			}
		}
	}
	if coloured < 1250 {
		t.Errorf("%d of 2500 points in the middle are not grey: is the face there?", coloured)
	}
}
