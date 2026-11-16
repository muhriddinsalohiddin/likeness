package client

import (
	"encoding/json"
	"math"
	"time"
)

// Reason says why a face, or a whole picture, was turned down. A server
// newer than this package may send reasons that are not listed here, and
// they arrive as they are.
type Reason string

const (
	// ReasonNoFace: no face was found in the picture.
	ReasonNoFace Reason = "no_face"
	// ReasonMultipleFaces: a reference picture shows a second face that
	// rivals the first in size.
	ReasonMultipleFaces Reason = "multiple_faces"
	// ReasonLowScore: the detector is not confident this is a face.
	ReasonLowScore Reason = "low_det_score"
	// ReasonOutOfFrame: too much of the face is cut off by the edge of the
	// picture.
	ReasonOutOfFrame Reason = "out_of_frame"
	// ReasonFaceTooSmall: too few pixels between the eyes.
	ReasonFaceTooSmall Reason = "face_too_small"
	// ReasonBadPose: the head is turned, tilted or leaning too far.
	ReasonBadPose Reason = "bad_pose"
	// ReasonTooDark and ReasonTooBright: the face is under- or
	// over-exposed.
	ReasonTooDark   Reason = "too_dark"
	ReasonTooBright Reason = "too_bright"
	// ReasonBlurry: the face lacks detail, from motion, focus or upscaling.
	ReasonBlurry Reason = "blurry"
	// ReasonSpoof: what was in front of the camera looks like a picture of
	// a face rather than a face.
	ReasonSpoof Reason = "spoof"
)

// Box is where a face is: the edges of a rectangle around it, in pixels of
// the upright picture, counted from its top left corner.
type Box struct {
	Left   float64 `json:"left"`
	Top    float64 `json:"top"`
	Right  float64 `json:"right"`
	Bottom float64 `json:"bottom"`
}

// Quality is what the server measured on a face.
type Quality struct {
	// Score is the detector's confidence that this is a face, 0 to 1.
	Score float64 `json:"score"`
	// EyeDistance is the number of pixels between the centres of the eyes;
	// a face is about 2.3 times as wide.
	EyeDistance float64 `json:"eye_distance"`
	// Yaw, Pitch and Roll are rough head angles in degrees.
	Yaw   float64 `json:"yaw"`
	Pitch float64 `json:"pitch"`
	Roll  float64 `json:"roll"`
	// InFrame is the share of the face box that lies inside the picture.
	InFrame float64 `json:"in_frame"`
	// Brightness is the mean brightness of the face, 0 to 255, and
	// Sharpness its fine detail: a sharp close-up scores a few hundred.
	// Both are zero for a face turned down for its size, position or pose,
	// before they were measured.
	Brightness float64 `json:"brightness,omitempty"`
	Sharpness  float64 `json:"sharpness,omitempty"`
}

// Face is a face as the server saw it in a picture.
type Face struct {
	// Box is where the face is and Quality what was measured on it.
	Box     Box     `json:"box"`
	Quality Quality `json:"quality"`
	// Liveness is the probability that this is a face and not a picture of
	// one; zero when the server does not judge it.
	Liveness float64 `json:"liveness,omitempty"`
	// Reject is set when the face was too poor to compare with anything.
	Reject Reason `json:"reject,omitempty"`
}

// Timing says where the server spent the time of a request with a picture.
type Timing struct {
	// Read is receiving the upload and Queue waiting for a free slot.
	Read, Queue time.Duration
	// Decode, Detect, Align and Embed are the work on the picture.
	Decode, Detect, Align, Embed time.Duration
	// Liveness is judging whether the face is live, on a server that does.
	Liveness time.Duration
	// Search, Compare and Store are what the endpoint did with the face: a
	// search has the first, a verification the second, an enrolment the
	// third.
	Search, Compare, Store time.Duration
	// Total is from the arrival of the request to its answer.
	Total time.Duration
	// Other holds stages this package does not know, under the names a
	// newer server gives them.
	Other map[string]time.Duration
}

// stage is one field of a Timing under its name in the API.
type stage struct {
	name string
	took *time.Duration
	// sometimes marks the stages only some endpoints go through.
	sometimes bool
}

func (t *Timing) stages() []stage {
	return []stage{
		{"read", &t.Read, false}, {"queue", &t.Queue, false},
		{"decode", &t.Decode, false}, {"detect", &t.Detect, false},
		{"align", &t.Align, false}, {"liveness", &t.Liveness, true}, {"embed", &t.Embed, false},
		{"search", &t.Search, true}, {"compare", &t.Compare, true}, {"store", &t.Store, true},
		{"total", &t.Total, false},
	}
}

// UnmarshalJSON reads the API's timing_ms object, which counts in
// milliseconds.
func (t *Timing) UnmarshalJSON(data []byte) error {
	var ms map[string]float64
	if err := json.Unmarshal(data, &ms); err != nil {
		return err
	}
	*t = Timing{}
	for _, st := range t.stages() {
		if took, ok := ms[st.name]; ok {
			*st.took = fromMillis(took)
			delete(ms, st.name)
		}
	}
	for name, took := range ms {
		if t.Other == nil {
			t.Other = make(map[string]time.Duration, len(ms))
		}
		t.Other[name] = fromMillis(took)
	}
	return nil
}

// MarshalJSON writes the timing as the API does, so that a result that is
// logged or passed on as JSON looks like the server's.
func (t Timing) MarshalJSON() ([]byte, error) {
	ms := make(map[string]float64, 8+len(t.Other))
	for name, took := range t.Other {
		ms[name] = millis(took)
	}
	for _, st := range t.stages() {
		if *st.took != 0 || !st.sometimes {
			ms[st.name] = millis(*st.took)
		}
	}
	return json.Marshal(ms)
}

func fromMillis(ms float64) time.Duration {
	return time.Duration(math.Round(ms * float64(time.Millisecond)))
}

func millis(d time.Duration) float64 {
	return float64(d) / float64(time.Millisecond)
}
