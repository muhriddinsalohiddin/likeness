package likeness

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/muhriddinsalohiddin/likeness/internal/align"
	"github.com/muhriddinsalohiddin/likeness/internal/detect"
	"github.com/muhriddinsalohiddin/likeness/internal/embed"
	"github.com/muhriddinsalohiddin/likeness/internal/quality"
	"github.com/muhriddinsalohiddin/likeness/internal/vision"
)

// Errors about the picture itself. Test for them with errors.Is.
var (
	// ErrUnsupportedImage: the data is not a JPEG, PNG or WebP picture.
	ErrUnsupportedImage = vision.ErrFormat
	// ErrImageTooLarge: the picture exceeds Config.MaxPixels or MaxSide.
	ErrImageTooLarge = vision.ErrTooLarge
	// ErrImageTooSmall: the picture is under 16 pixels on a side.
	ErrImageTooSmall = vision.ErrTooSmall
	// ErrCorruptImage: the picture is damaged or cut short.
	ErrCorruptImage = vision.ErrCorrupt
)

// Selection says which of the detected faces to analyse.
type Selection int

const (
	// Largest analyses the biggest face that meets the policy's geometric
	// limits and ignores the rest: the person at the camera, not the people
	// behind them. If no face meets them, the biggest one is reported with
	// the reason it was turned down.
	Largest Selection = iota
	// All analyses every face, up to Options.MaxFaces: usable ones first,
	// biggest first.
	All
	// Single expects one person: it analyses the biggest face and turns the
	// picture down with MultipleFaces if another face rivals it in size
	// (see Policy.MaxRival).
	Single
)

// Options tune one Analyze call. The zero value analyses the largest face
// and accepts it whatever its quality; ForSearch and ForEnrolment are the
// usual starting points.
type Options struct {
	Faces  Selection
	Policy Policy
	// MaxFaces bounds the faces analysed when Faces is All; 0 means 32.
	MaxFaces int
}

// ForSearch returns the options for recognising whoever is at the camera.
func ForSearch() Options { return Options{Faces: Largest, Policy: SearchPolicy()} }

// ForEnrolment returns the options for taking a reference picture.
func ForEnrolment() Options { return Options{Faces: Single, Policy: EnrolmentPolicy()} }

// Point is a position in picture pixels.
type Point struct{ X, Y float32 }

// Box is a rectangle in picture pixels. It may extend past the picture when
// a face is partly out of frame.
type Box struct{ Left, Top, Right, Bottom float32 }

// Width of the box.
func (b Box) Width() float32 { return b.Right - b.Left }

// Height of the box.
func (b Box) Height() float32 { return b.Bottom - b.Top }

// Face is one face found in a picture.
type Face struct {
	Box Box
	// Landmarks are the left eye, right eye, nose tip, left mouth corner and
	// right mouth corner, as seen in the picture.
	Landmarks [5]Point
	Quality   Quality
	// Reject says why the face was not embedded; Accepted if it was.
	Reject Reason
	// Embedding is the unit-length vector describing the face, nil if the
	// face was turned down. Its length is Info.Dim.
	Embedding []float32
}

// Timing is how long each stage of one call took.
type Timing struct {
	Decode time.Duration // reading the picture
	Detect time.Duration // finding faces, all passes
	Align  time.Duration // cropping faces and measuring their quality
	Embed  time.Duration // computing vectors
	Total  time.Duration
}

// Result is what Analyze found in one picture.
type Result struct {
	// Width and Height of the picture, upright.
	Width, Height int
	// Faces are the analysed faces: usable ones first, biggest first.
	Faces []Face
	// Found is the number of faces the detector reported, which can exceed
	// len(Faces).
	Found int
	// Reject is set when the picture as a whole is unusable: NoFace, or
	// MultipleFaces when one person was expected.
	Reject Reason
	// DetectSize is the detector input side that produced Faces.
	DetectSize int
	Timing     Timing
}

// Best returns the first face and whether it can be used: Accepted, the
// picture's Reject, or the face's own. The face is nil when there is none.
func (r *Result) Best() (*Face, Reason) {
	if len(r.Faces) == 0 {
		if r.Reject != Accepted {
			return nil, r.Reject
		}
		return nil, NoFace
	}
	f := &r.Faces[0]
	if r.Reject != Accepted {
		return f, r.Reject
	}
	return f, f.Reject
}

// A face whose eyes are closer than this in the detector's own input is
// located too coarsely to be aligned well; a larger input is tried when the
// detector offers one.
const minEyesInDetector = 12

const defaultMaxFaces = 32

// Analyze finds the faces in an encoded picture (JPEG, PNG or WebP), judges
// each against the policy and embeds the ones that pass.
//
// The error is non-nil only when the picture cannot be read or the call was
// cancelled; a readable picture with no usable face is a Result with a
// Reject reason.
func (e *Engine) Analyze(ctx context.Context, picture []byte, o Options) (*Result, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed {
		return nil, ErrClosed
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	start := time.Now()
	img, err := vision.Decode(picture, e.limits)
	if err != nil {
		return nil, fmt.Errorf("likeness: %w", err)
	}
	res := &Result{}
	res.Width, res.Height = img.Size()
	res.Timing.Decode = time.Since(start)

	mark := time.Now()
	found, size, err := e.detect(ctx, img)
	if err != nil {
		return nil, err
	}
	res.Timing.Detect = time.Since(mark)
	res.Found, res.DetectSize = len(found), size
	if len(found) == 0 {
		res.Reject = NoFace
		res.Timing.Total = time.Since(start)
		return res, nil
	}

	// Judge every face by what the detection alone tells, then put the
	// usable ones first: the largest face that could be recognised matters
	// more than a larger one that could not.
	policy := quality.Policy(o.Policy)
	cands := make([]candidate, len(found))
	for i, f := range found {
		m := quality.Geometry(f.Box, f.Landmarks, f.Score, res.Width, res.Height)
		cands[i] = candidate{face: f, measures: m, reason: policy.CheckGeometry(m), side: boxSide(f)}
	}
	rank(cands)

	take := 1
	switch o.Faces {
	case All:
		take = o.MaxFaces
		if take <= 0 {
			take = defaultMaxFaces
		}
		take = min(take, len(cands))
	case Single:
		if rivalled(cands, policy) {
			res.Reject = MultipleFaces
		}
	}

	res.Faces = make([]Face, take)
	for i := range res.Faces {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := e.analyzeFace(img, cands[i], policy, res.Reject == Accepted, &res.Faces[i], &res.Timing); err != nil {
			return nil, err
		}
	}
	res.Timing.Total = time.Since(start)
	return res, nil
}

type candidate struct {
	face     detect.Face
	measures quality.Measures
	reason   quality.Reason
	side     float32
}

// rank orders faces for analysis: usable ones first, biggest first.
func rank(cands []candidate) {
	sort.SliceStable(cands, func(a, b int) bool {
		if ua, ub := cands[a].reason == quality.OK, cands[b].reason == quality.OK; ua != ub {
			return ua
		}
		return cands[a].side > cands[b].side
	})
}

// rivalled reports whether a second face is prominent enough to make it
// unclear whose picture this is. A rival has to be a confident detection;
// its pose and position do not matter.
func rivalled(cands []candidate, p quality.Policy) bool {
	if p.MaxRival <= 0 || len(cands) < 2 || cands[0].reason != quality.OK {
		return false
	}
	minScore := max(p.MinScore, detect.DefaultThreshold)
	for _, c := range cands[1:] {
		if c.face.Score >= minScore && c.side > p.MaxRival*cands[0].side {
			return true
		}
	}
	return false
}

// detect runs the detector at growing input sizes until it has a face it
// can locate precisely.
func (e *Engine) detect(ctx context.Context, img *vision.Image) ([]detect.Face, int, error) {
	w, h := img.Size()
	var faces []detect.Face
	var used int
	for i, size := range e.sizes {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		got, err := e.det.Detect(img, detect.Options{Size: size})
		if err != nil {
			return nil, 0, fmt.Errorf("likeness: %w", err)
		}
		if len(got) > 0 || used == 0 {
			faces, used = got, size
		}
		if i == len(e.sizes)-1 || len(got) == 0 {
			continue
		}
		// picture pixels per detector pixel
		scale := float32(max(w, h)) / float32(size)
		biggest := float32(0)
		for _, f := range got {
			biggest = max(biggest, eyeDistance(f))
		}
		if biggest/scale >= minEyesInDetector {
			break
		}
	}
	return faces, used, nil
}

func (e *Engine) analyzeFace(img *vision.Image, c candidate, policy quality.Policy, embedIt bool, out *Face, t *Timing) error {
	f, m := c.face, c.measures
	out.Box = Box{f.Box[0], f.Box[1], f.Box[2], f.Box[3]}
	for i, p := range f.Landmarks {
		out.Landmarks[i] = Point{p.X, p.Y}
	}
	defer func() { out.Quality = Quality(m) }()
	if c.reason != quality.OK {
		out.Reject = Reason(c.reason)
		return nil
	}

	mark := time.Now()
	crop := e.crops.Get().(*[]float32)
	defer e.crops.Put(crop)
	if err := e.emb.Crop(img, f.Landmarks, *crop, embed.Options{}); err != nil {
		if errors.Is(err, align.ErrDegenerate) {
			out.Reject = LowScore // landmarks that do not span a face
			return nil
		}
		return fmt.Errorf("likeness: %w", err)
	}
	m.Pixels(*crop, e.emb.CropSize())
	t.Align += time.Since(mark)
	if r := policy.CheckPixels(m); r != quality.OK {
		out.Reject = Reason(r)
		return nil
	}
	if !embedIt {
		return nil
	}

	mark = time.Now()
	vec := make([]float32, e.emb.Dim())
	if err := e.emb.EmbedCrop(*crop, vec); err != nil {
		return fmt.Errorf("likeness: %w", err)
	}
	t.Embed += time.Since(mark)
	out.Embedding = vec
	return nil
}

func boxSide(f detect.Face) float32 {
	return float32(math.Sqrt(float64(max(f.Width(), 0) * max(f.Height(), 0))))
}

func eyeDistance(f detect.Face) float32 {
	return float32(math.Hypot(float64(f.Landmarks[1].X-f.Landmarks[0].X), float64(f.Landmarks[1].Y-f.Landmarks[0].Y)))
}
