package vision

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"

	"golang.org/x/image/webp"
)

// Limits bound what Decode accepts. They are checked against the image
// header, before any pixel memory is allocated.
type Limits struct {
	MaxPixels int // width * height
	MaxSide   int
	MinSide   int
}

// DefaultLimits admit anything a phone camera produces and reject
// decompression bombs.
var DefaultLimits = Limits{MaxPixels: 25_000_000, MaxSide: 8192, MinSide: 16}

var (
	ErrFormat   = errors.New("vision: unsupported image format")
	ErrTooLarge = errors.New("vision: image is larger than the configured limit")
	ErrTooSmall = errors.New("vision: image is too small")
	ErrCorrupt  = errors.New("vision: image data is corrupt")
)

// Decode parses a JPEG, PNG or WebP picture. Zero fields of lim fall back to
// DefaultLimits.
func Decode(data []byte, lim Limits) (img *Image, err error) {
	if lim.MaxPixels <= 0 {
		lim.MaxPixels = DefaultLimits.MaxPixels
	}
	if lim.MaxSide <= 0 {
		lim.MaxSide = DefaultLimits.MaxSide
	}
	if lim.MinSide <= 0 {
		lim.MinSide = DefaultLimits.MinSide
	}

	// Decoders are fed untrusted bytes. The standard ones are fuzzed and
	// return errors, but a panic here must never take a server down.
	defer func() {
		if r := recover(); r != nil {
			img, err = nil, fmt.Errorf("%w: decoder panicked: %v", ErrCorrupt, r)
		}
	}()

	cfg, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		if errors.Is(err, image.ErrFormat) {
			return nil, ErrFormat
		}
		return nil, fmt.Errorf("%w: %w", ErrCorrupt, err)
	}
	if cfg.Width < lim.MinSide || cfg.Height < lim.MinSide {
		return nil, fmt.Errorf("%w: %dx%d", ErrTooSmall, cfg.Width, cfg.Height)
	}
	if cfg.Width > lim.MaxSide || cfg.Height > lim.MaxSide || int64(cfg.Width)*int64(cfg.Height) > int64(lim.MaxPixels) {
		return nil, fmt.Errorf("%w: %dx%d", ErrTooLarge, cfg.Width, cfg.Height)
	}

	var decoded image.Image
	orientation := 1
	switch format {
	case "jpeg":
		decoded, err = jpeg.Decode(bytes.NewReader(data))
		orientation = exifOrientation(data)
	case "png":
		decoded, err = png.Decode(bytes.NewReader(data))
	case "webp":
		decoded, err = webp.Decode(bytes.NewReader(data))
	default:
		// another package registered a format we do not serve
		return nil, ErrFormat
	}
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrCorrupt, err)
	}
	return FromImage(decoded, orientation), nil
}
