package likeness

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"os"
	"testing"

	"golang.org/x/image/draw"

	"github.com/muhriddinsalohiddin/likeness/internal/fixtures"
	"github.com/muhriddinsalohiddin/likeness/internal/home"
	"github.com/muhriddinsalohiddin/likeness/internal/testenv"
)

// presets lists the model pairs the tests exercise; a test for a preset
// whose files are not installed is skipped.
var presets = []string{"default", "buffalo_s"}

var presetModels = map[string][2]string{
	"default":   {"yunet_2023mar", "sface_2021dec"},
	"buffalo_s": {"det_500m", "w600k_mbf"},
}

const testThreads = 4

// open returns an engine for the preset, or skips the test when the runtime
// library or the preset's models are not installed.
func open(t testing.TB, preset string) *Engine {
	t.Helper()
	if _, err := os.Stat(home.LibraryPath()); err != nil {
		t.Skipf("onnxruntime is not installed: %v", err)
	}
	for _, name := range presetModels[preset] {
		testenv.Model(t, name)
	}
	e, err := Open(Config{Preset: preset, Threads: testThreads})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(e.Close)
	return e
}

// analyze is Analyze without the plumbing.
func analyze(t testing.TB, e *Engine, picture []byte, o Options) *Result {
	t.Helper()
	res, err := e.Analyze(context.Background(), picture, o)
	if err != nil {
		t.Fatal(err)
	}
	return res
}

// portrait returns the pixels of one close-up, for tests that degrade it.
func portrait(t testing.TB) *image.RGBA {
	t.Helper()
	b, err := fixtures.Closeup("hathaway")
	if err != nil {
		t.Fatal(err)
	}
	src, err := jpeg.Decode(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	dst := image.NewRGBA(src.Bounds())
	draw.Draw(dst, dst.Bounds(), src, src.Bounds().Min, draw.Src)
	return dst
}

func encodePNG(t testing.TB, img image.Image) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func encodeJPEG(t testing.TB, img image.Image, quality int) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: quality}); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// blur applies a Gaussian of the given standard deviation, in pixels.
func blur(src *image.RGBA, sigma float64) *image.RGBA {
	radius := int(math.Ceil(3 * sigma))
	kernel := make([]float64, 2*radius+1)
	var sum float64
	for i := range kernel {
		d := float64(i - radius)
		kernel[i] = math.Exp(-d * d / (2 * sigma * sigma))
		sum += kernel[i]
	}
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	pass := func(in *image.RGBA, dx, dy int) *image.RGBA {
		out := image.NewRGBA(b)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				var acc [3]float64
				for k, wt := range kernel {
					sx := min(max(x+(k-radius)*dx, 0), w-1)
					sy := min(max(y+(k-radius)*dy, 0), h-1)
					c := in.RGBAAt(sx, sy)
					acc[0] += wt * float64(c.R)
					acc[1] += wt * float64(c.G)
					acc[2] += wt * float64(c.B)
				}
				out.SetRGBA(x, y, color.RGBA{uint8(acc[0]/sum + 0.5), uint8(acc[1]/sum + 0.5), uint8(acc[2]/sum + 0.5), 255})
			}
		}
		return out
	}
	return pass(pass(src, 1, 0), 0, 1)
}

// expose multiplies every channel by gain and adds lift, clamping to 0..255.
func expose(src *image.RGBA, gain, lift float64) *image.RGBA {
	out := image.NewRGBA(src.Bounds())
	for i := 0; i < len(src.Pix); i += 4 {
		for c := 0; c < 3; c++ {
			out.Pix[i+c] = uint8(math.Min(math.Max(float64(src.Pix[i+c])*gain+lift, 0), 255))
		}
		out.Pix[i+3] = 255
	}
	return out
}

// scale resizes the picture by the given factor with a smooth filter.
func scale(src *image.RGBA, factor float64) *image.RGBA {
	b := src.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, int(float64(b.Dx())*factor+0.5), int(float64(b.Dy())*factor+0.5)))
	draw.CatmullRom.Scale(dst, dst.Bounds(), src, b, draw.Src, nil)
	return dst
}

// onCanvas places the picture, shrunk by factor, in the middle of a larger
// mid-grey canvas: a small face in a big frame.
func onCanvas(src *image.RGBA, factor float64, w, h int) *image.RGBA {
	small := scale(src, factor)
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(dst, dst.Bounds(), image.NewUniform(color.RGBA{110, 110, 110, 255}), image.Point{}, draw.Src)
	off := image.Pt((w-small.Bounds().Dx())/2, (h-small.Bounds().Dy())/2)
	draw.Draw(dst, small.Bounds().Add(off), small, image.Point{}, draw.Src)
	return dst
}

// rotate turns the picture clockwise by the given angle about its centre.
func rotate(src *image.RGBA, degrees float64) *image.RGBA {
	b := src.Bounds()
	dst := image.NewRGBA(b)
	sin, cos := math.Sincos(degrees * math.Pi / 180)
	cx, cy := float64(b.Dx())/2, float64(b.Dy())/2
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			dx, dy := float64(x)-cx, float64(y)-cy
			sx, sy := int(cos*dx+sin*dy+cx+0.5), int(-sin*dx+cos*dy+cy+0.5)
			if sx >= 0 && sy >= 0 && sx < b.Dx() && sy < b.Dy() {
				dst.SetRGBA(x, y, src.RGBAAt(sx, sy))
			} else {
				dst.SetRGBA(x, y, color.RGBA{110, 110, 110, 255})
			}
		}
	}
	return dst
}

func cosine(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}
