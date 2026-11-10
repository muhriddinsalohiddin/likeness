package likeness

import "github.com/muhriddinsalohiddin/likeness/internal/quality"

// Quality describes how suitable one face is for recognition.
type Quality struct {
	// Score is the detector's confidence, 0..1.
	Score float32
	// EyeDistance is the distance between the eye centres in picture pixels.
	// A face is roughly 2.3 times as wide.
	EyeDistance float32
	// Yaw, Pitch and Roll are rough head angles in degrees, estimated from
	// five landmarks and a generic head shape; expect errors of ten degrees.
	// Yaw is positive when the face turns towards the right edge of the
	// picture, Pitch when it tilts down, Roll when it leans clockwise.
	// Pitch is the weakest of the three: a smile alone moves it by ten to
	// twenty degrees.
	Yaw, Pitch, Roll float32
	// InFrame is the fraction of the face box that lies inside the picture.
	InFrame float32
	// Brightness is the mean luma of the face area of the aligned crop,
	// 0..255. Zero when the face was turned down before it was cropped.
	Brightness float32
	// Sharpness is the variance of the Laplacian over the same area, taken
	// relative to its brightness (and scaled by 10^4) so that a dim picture
	// does not pass for a blurred one. A sharp close-up scores a few hundred;
	// below about 40 the fine detail is gone. Zero when the face was not
	// cropped.
	Sharpness float32
}

// Reason says why a face, or a whole picture, was turned down. The values
// are stable and meant to be passed on to API clients.
type Reason string

const (
	// Accepted is the zero Reason: nothing is wrong.
	Accepted Reason = ""
	// NoFace: the detector found no face in the picture.
	NoFace Reason = "no_face"
	// MultipleFaces: one face was expected and a second prominent one is there.
	MultipleFaces Reason = "multiple_faces"
	// LowScore: the detector is not confident this is a face.
	LowScore Reason = "low_det_score"
	// OutOfFrame: too much of the face is cut off by the picture's edge.
	OutOfFrame Reason = "out_of_frame"
	// FaceTooSmall: too few pixels between the eyes.
	FaceTooSmall Reason = "face_too_small"
	// BadPose: the head is turned, tilted or leaning too far.
	BadPose Reason = "bad_pose"
	// TooDark and TooBright: the face is under- or over-exposed.
	TooDark   Reason = "too_dark"
	TooBright Reason = "too_bright"
	// Blurry: the face lacks detail, from motion, focus or upscaling.
	Blurry Reason = "blurry"
)

// Policy holds the limits a face must meet to be embedded. A zero field
// sets no limit, so the zero Policy accepts every face the detector reports.
type Policy struct {
	MinScore       float32
	MinEyeDistance float32
	// MaxYaw, MaxPitch and MaxRoll bound the absolute angles, in degrees.
	MaxYaw, MaxPitch, MaxRoll float32
	MinInFrame                float32
	MinBrightness             float32
	MaxBrightness             float32
	MinSharpness              float32
	// MaxRival bounds how large a second face may be relative to the main
	// one (ratio of box sides) before the picture counts as showing several
	// people. Only consulted when exactly one face is expected.
	MaxRival float32
}

// SearchPolicy is lenient: it turns down only the faces that cannot give a
// meaningful vector, so that recognition works for someone glancing at a
// camera in a dim room.
//
// The limits come from measuring how far a face's vector drifts from that
// of the clean picture as the picture is degraded; each limit sits where
// the similarity has fallen to roughly 0.85.
func SearchPolicy() Policy {
	return Policy{
		MinScore:       0.5,
		MinEyeDistance: 20,
		MaxYaw:         55,
		MaxPitch:       55,
		MinInFrame:     0.6,
		MinBrightness:  15,
		MaxBrightness:  240,
		MinSharpness:   4,
	}
}

// EnrolmentPolicy is strict: a poor reference picture spoils every later
// comparison, so it is better to ask for another one. Its limits sit where
// a degraded picture's vector still has a similarity of about 0.97 with
// the clean one.
func EnrolmentPolicy() Policy {
	return Policy{
		MinScore:       0.7,
		MinEyeDistance: 40,
		MaxYaw:         30,
		MaxPitch:       40,
		MaxRoll:        25,
		MinInFrame:     0.95,
		MinBrightness:  50,
		MaxBrightness:  215,
		MinSharpness:   40,
		MaxRival:       0.5,
	}
}

// The public types mirror the internal ones field for field; these
// conversions stop compiling if they drift apart.
var (
	_ = quality.Measures(Quality{})
	_ = quality.Policy(Policy{})
)
