package vision

import (
	"bytes"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"image"
	"image/png"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/muhriddinsalohiddin/likeness/internal/align"
)

// withOrientation inserts an EXIF APP1 segment carrying the orientation tag
// right after the JPEG start-of-image marker.
func withOrientation(jpg []byte, orientation uint16, order binary.ByteOrder) []byte {
	tiff := make([]byte, 8+2+12+4)
	copy(tiff, "II")
	if order == binary.BigEndian {
		copy(tiff, "MM")
	}
	order.PutUint16(tiff[2:], 42)
	order.PutUint32(tiff[4:], 8)
	order.PutUint16(tiff[8:], 1) // one entry
	order.PutUint16(tiff[10:], 0x0112)
	order.PutUint16(tiff[12:], 3)
	order.PutUint32(tiff[14:], 1)
	order.PutUint16(tiff[18:], orientation)

	payload := append([]byte("Exif\x00\x00"), tiff...)
	seg := []byte{0xFF, 0xE1, 0, 0}
	binary.BigEndian.PutUint16(seg[2:], uint16(len(payload)+2))
	out := append([]byte{}, jpg[:2]...)
	out = append(out, seg...)
	out = append(out, payload...)
	return append(out, jpg[2:]...)
}

func pngHeader(w, h uint32) []byte {
	ihdr := make([]byte, 13)
	binary.BigEndian.PutUint32(ihdr, w)
	binary.BigEndian.PutUint32(ihdr[4:], h)
	ihdr[8], ihdr[9] = 8, 2 // 8-bit RGB
	chunk := append([]byte("IHDR"), ihdr...)
	out := []byte("\x89PNG\r\n\x1a\n")
	out = binary.BigEndian.AppendUint32(out, 13)
	out = append(out, chunk...)
	return binary.BigEndian.AppendUint32(out, crc32.ChecksumIEEE(chunk))
}

func TestDecodeFormats(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	read := func(name string) []byte {
		b, err := os.ReadFile("testdata/" + name)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	cases := []struct {
		name string
		data []byte
		kind kind
		w, h int
	}{
		{"colour jpeg", encodeJPEG(t, smoothRGBA(rng, 64, 48), 90), kindYCbCr, 64, 48},
		{"grey jpeg", encodeJPEG(t, smoothGray(rng, 40, 56), 90), kindGray, 40, 56},
		{"png", read("gradient.png"), kindRGBA, 48, 32},
		{"lossy webp", read("gradient_lossy.webp"), kindYCbCr, 48, 32},
		{"lossless webp", read("gradient_lossless.webp"), kindRGBA, 48, 32},
	}
	for _, c := range cases {
		m, err := Decode(c.data, Limits{})
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if w, h := m.Size(); m.kind != c.kind || w != c.w || h != c.h {
			t.Fatalf("%s: kind %d size %dx%d, want kind %d size %dx%d", c.name, m.kind, w, h, c.kind, c.w, c.h)
		}
	}

	// the fixtures show a yellow square around (35, 12)
	for name, tol := range map[string]float64{"gradient.png": 0, "gradient_lossless.webp": 0, "gradient_lossy.webp": 30} {
		m, err := Decode(read(name), Limits{})
		if err != nil {
			t.Fatal(err)
		}
		px := make([]float32, 3)
		if err := m.Warp(px, 1, align.Affine{A: 1, C: -35, E: 1, F: -12}, raw, Options{}); err != nil {
			t.Fatal(err)
		}
		for i, want := range []float32{250, 240, 20} {
			if math.Abs(float64(px[i]-want)) > tol {
				t.Fatalf("%s: pixel (35,12) is %v, want about [250 240 20]", name, px)
			}
		}
	}
}

func TestDecodeAppliesLimitsBeforeDecoding(t *testing.T) {
	jpg := encodeJPEG(t, smoothGray(rand.New(rand.NewSource(1)), 640, 480), 80)
	for _, c := range []struct {
		name string
		data []byte
		lim  Limits
		want error
	}{
		// header-only files: a decoder that allocated first would need gigabytes
		{"pixel bomb", pngHeader(60000, 60000), Limits{MaxSide: 100000}, ErrTooLarge},
		{"long side", pngHeader(9000, 100), Limits{}, ErrTooLarge},
		{"custom pixel budget", jpg, Limits{MaxPixels: 100_000}, ErrTooLarge},
		{"tiny", pngHeader(8, 300), Limits{}, ErrTooSmall},
	} {
		if _, err := Decode(c.data, c.lim); !errors.Is(err, c.want) {
			t.Errorf("%s: got %v, want %v", c.name, err, c.want)
		}
	}
	if _, err := Decode(jpg, Limits{}); err != nil {
		t.Errorf("640x480 must pass the default limits: %v", err)
	}
}

func TestDecodeRejectsBadInput(t *testing.T) {
	jpg := encodeJPEG(t, smoothRGBA(rand.New(rand.NewSource(2)), 96, 96), 85)
	var pngBuf bytes.Buffer
	if err := png.Encode(&pngBuf, smoothRGBA(rand.New(rand.NewSource(2)), 96, 96)); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		data []byte
		want error
	}{
		{"empty", nil, ErrFormat},
		{"text", []byte("definitely not a picture"), ErrFormat},
		{"gif", []byte("GIF89a\x10\x00\x10\x00\x80\x00\x00"), ErrFormat},
		{"truncated jpeg", jpg[:len(jpg)/2], ErrCorrupt},
		{"truncated png", pngBuf.Bytes()[:pngBuf.Len()/2], ErrCorrupt},
		{"jpeg header only", jpg[:20], ErrCorrupt},
	} {
		if _, err := Decode(c.data, Limits{}); !errors.Is(err, c.want) {
			t.Errorf("%s: got %v, want %v", c.name, err, c.want)
		}
	}
}

func TestDecodeHonoursExifOrientation(t *testing.T) {
	jpg := encodeJPEG(t, smoothRGBA(rand.New(rand.NewSource(9)), 80, 48), 90)
	m, err := Decode(withOrientation(jpg, 6, binary.BigEndian), Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if w, h := m.Size(); m.orient != 6 || w != 48 || h != 80 {
		t.Fatalf("orientation %d, size %dx%d; want 6 and 48x80", m.orient, w, h)
	}
}

func TestExifOrientation(t *testing.T) {
	jpg := encodeJPEG(t, image.NewGray(image.Rect(0, 0, 16, 16)), 80)
	if got := exifOrientation(jpg); got != 1 {
		t.Fatalf("no EXIF: got %d, want 1", got)
	}
	for _, order := range []binary.ByteOrder{binary.LittleEndian, binary.BigEndian} {
		for o := uint16(1); o <= 8; o++ {
			if got := exifOrientation(withOrientation(jpg, o, order)); got != int(o) {
				t.Errorf("%v orientation %d: got %d", order, o, got)
			}
		}
		for _, bad := range []uint16{0, 9, 0xFFFF} {
			if got := exifOrientation(withOrientation(jpg, bad, order)); got != 1 {
				t.Errorf("%v invalid value %d: got %d, want 1", order, bad, got)
			}
		}
	}

	// an XMP packet also lives in APP1 and often comes first
	xmp := append([]byte{0xFF, 0xE1, 0x00, 0x22}, make([]byte, 0x20)...)
	copy(xmp[4:], "http://ns.adobe.com/xap/1.0/\x00")
	tagged := withOrientation(jpg, 8, binary.LittleEndian)
	both := append(append(append([]byte{}, tagged[:2]...), xmp...), tagged[2:]...)
	if got := exifOrientation(both); got != 8 {
		t.Errorf("XMP before EXIF: got %d, want 8", got)
	}
}

func TestExifOrientationSurvivesMalformedSegments(t *testing.T) {
	jpg := encodeJPEG(t, image.NewGray(image.Rect(0, 0, 16, 16)), 80)
	good := withOrientation(jpg, 3, binary.LittleEndian)
	tiff := 2 + 4 + 6 // SOI, APP1 header, "Exif\0\0"

	mutate := func(f func(b []byte)) []byte {
		b := append([]byte{}, good...)
		f(b)
		return b
	}
	cases := map[string][]byte{
		"cut inside the segment":   good[:tiff+10],
		"segment longer than file": mutate(func(b []byte) { binary.BigEndian.PutUint16(b[4:], 0xFFF0) }),
		"segment length below 2":   mutate(func(b []byte) { binary.BigEndian.PutUint16(b[4:], 1) }),
		"IFD offset out of range":  mutate(func(b []byte) { binary.LittleEndian.PutUint32(b[tiff+4:], 0xFFFFFFF0) }),
		"absurd entry count":       mutate(func(b []byte) { binary.LittleEndian.PutUint16(b[tiff+8:], 0xFFFF); b[tiff+10] = 0 }),
		"wrong magic":              mutate(func(b []byte) { b[tiff+2] = 43 }),
		"unknown byte order":       mutate(func(b []byte) { b[tiff] = 'X' }),
		"tag with the wrong type":  mutate(func(b []byte) { binary.LittleEndian.PutUint16(b[tiff+12:], 4) }),
		"not a jpeg":               []byte("\x89PNG\r\n\x1a\n"),
		"only a marker":            {0xFF, 0xD8, 0xFF},
	}
	for name, data := range cases {
		if got := exifOrientation(data); got != 1 {
			t.Errorf("%s: got %d, want 1", name, got)
		}
	}
}
