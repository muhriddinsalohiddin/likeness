// Package enginetest provides a stand-in for the face engine, so that the
// layers above it can be tested on machines with no models installed.
//
// The fake reads its "picture" as a line of text that says what the engine
// should have seen in it:
//
//	face:anna        one good face; the same name always gives the same vector
//	face:anna#2      the same person in another picture (similarity about 0.9)
//	faces:anna,boris several faces, largest first
//	spoof:anna       a picture of anna held up to the camera: her face, with a
//	                 liveness of 0.08
//	reject:blurry    one face, turned down for the given reason
//	reject:no_face   a picture with no usable face (also multiple_faces)
//	error:corrupt    a picture that cannot be read (also unsupported, large, small)
//	slow:anna        like face:anna, after waiting for Fake.Gate or Fake.Delay
//	panic            the engine blows up
package enginetest

import (
	"context"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"strings"
	"sync/atomic"
	"time"

	"github.com/muhriddinsalohiddin/likeness"
)

// Dim is the length of the fake's vectors.
const Dim = 64

// Fake implements the engine's methods. The zero value is ready to use.
type Fake struct {
	// Gate, if set, holds every "slow:" picture until it is closed or
	// receives a value.
	Gate chan struct{}
	// Delay is how long a "slow:" picture takes when Gate is nil.
	Delay time.Duration

	calls   atomic.Int64
	running atomic.Int64
	peak    atomic.Int64
	closed  atomic.Bool
}

// Calls is the number of pictures analysed so far.
func (f *Fake) Calls() int { return int(f.calls.Load()) }

// Peak is the largest number of pictures that were being analysed at once.
func (f *Fake) Peak() int { return int(f.peak.Load()) }

// Closed reports whether Close has been called.
func (f *Fake) Closed() bool { return f.closed.Load() }

func (f *Fake) Close() { f.closed.Store(true) }

func (f *Fake) Info() likeness.Info {
	return likeness.Info{
		Preset: "fake", Detector: "fake-detector", Embedder: "fake-embedder",
		Fingerprint: "fake:embedder:0000000000000000", Dim: Dim, Runtime: "0.0.0", Threads: 1,
		VerifyThreshold: 0.5, SearchThreshold: 0.7,
		Liveness: []string{"fake-liveness"},
	}
}

// Vector returns the vector the fake produces for a name such as "anna" or
// "anna#2".
func Vector(name string) []float32 {
	person, variant, varied := strings.Cut(name, "#")
	v := gaussian(person)
	if varied {
		noise := gaussian(person + "#" + variant)
		for i := range v {
			v[i] = 0.9*v[i] + float32(math.Sqrt(1-0.81))*noise[i]
		}
		normalise(v)
	}
	return v
}

func gaussian(seed string) []float32 {
	h := fnv.New64a()
	h.Write([]byte(seed))
	rng := rand.New(rand.NewSource(int64(h.Sum64())))
	v := make([]float32, Dim)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	normalise(v)
	return v
}

func normalise(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	scale := float32(1 / math.Sqrt(sum))
	for i := range v {
		v[i] *= scale
	}
}

func face(name string, rank int) likeness.Face {
	size := float32(200 - 30*rank) // later faces are smaller
	left := float32(100 + 220*rank)
	f := likeness.Face{
		Box:      likeness.Box{Left: left, Top: 80, Right: left + size, Bottom: 80 + size*1.25},
		Liveness: 0.97,
		Quality: likeness.Quality{
			Score: 0.95, EyeDistance: size * 0.42, Yaw: 3, Pitch: -2, Roll: 1,
			InFrame: 1, Brightness: 140, Sharpness: 310,
		},
	}
	for i := range f.Landmarks {
		f.Landmarks[i] = likeness.Point{X: left + size*float32(i+1)/6, Y: 120 + float32(i)*20}
	}
	if name != "" {
		f.Embedding = Vector(name)
	}
	return f
}

func (f *Fake) Analyze(ctx context.Context, picture []byte, o likeness.Options) (*likeness.Result, error) {
	if f.closed.Load() {
		return nil, likeness.ErrClosed
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	f.calls.Add(1)
	now := f.running.Add(1)
	defer f.running.Add(-1)
	for {
		peak := f.peak.Load()
		if now <= peak || f.peak.CompareAndSwap(peak, now) {
			break
		}
	}

	kind, arg, _ := strings.Cut(strings.TrimSpace(string(picture)), ":")
	res := &likeness.Result{
		Width: 1280, Height: 720, DetectSize: 640,
		Timing: likeness.Timing{
			Decode: 2 * time.Millisecond, Detect: 3 * time.Millisecond,
			Align: time.Millisecond, Embed: 4 * time.Millisecond, Total: 10 * time.Millisecond,
		},
	}
	switch kind {
	case "slow":
		if f.Gate != nil {
			select {
			case <-f.Gate:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		} else {
			select {
			case <-time.After(f.Delay):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		fallthrough
	case "face":
		res.Found = 1
		res.Faces = []likeness.Face{face(arg, 0)}
	case "spoof":
		res.Found = 1
		shown := face(arg, 0)
		shown.Liveness = 0.08
		res.Faces = []likeness.Face{shown}
	case "faces":
		names := strings.Split(arg, ",")
		res.Found = len(names)
		switch o.Faces {
		case likeness.All:
			for i, name := range names {
				res.Faces = append(res.Faces, face(name, i))
			}
		case likeness.Single:
			res.Faces = []likeness.Face{face("", 0)}
			res.Reject = likeness.MultipleFaces
		default:
			res.Faces = []likeness.Face{face(names[0], 0)}
		}
	case "reject":
		reason := likeness.Reason(arg)
		if reason == likeness.NoFace {
			res.Reject = reason
			res.Timing.Embed = 0
			break
		}
		res.Found = 1
		res.Faces = []likeness.Face{face("", 0)}
		if reason == likeness.MultipleFaces {
			res.Found = 2
			res.Reject = reason
			break
		}
		res.Faces[0].Reject = reason
		res.Faces[0].Quality.Sharpness = 3
	case "error":
		cause := map[string]error{
			"unsupported": likeness.ErrUnsupportedImage, "corrupt": likeness.ErrCorruptImage,
			"large": likeness.ErrImageTooLarge, "small": likeness.ErrImageTooSmall,
		}[arg]
		if cause == nil {
			return nil, fmt.Errorf("enginetest: the engine broke: %s", arg)
		}
		return nil, fmt.Errorf("likeness: %w", cause)
	case "panic":
		panic("enginetest: the engine blew up")
	default:
		return nil, fmt.Errorf("likeness: %w", likeness.ErrUnsupportedImage)
	}
	for i := range res.Faces {
		if f := &res.Faces[i]; f.Embedding != nil && f.Liveness < o.MinLiveness {
			f.Reject, f.Embedding = likeness.Spoof, nil
		}
	}
	return res, nil
}
