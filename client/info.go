package client

import (
	"context"
	"encoding/json"
	"net/http"
	"time"
)

// Info is what a server runs and holds.
type Info struct {
	// Version is the release of likeness the server runs.
	Version string `json:"version"`
	// Preset names the pair of models in use, Detector and Embedder the
	// models themselves.
	Preset   string `json:"preset"`
	Detector string `json:"detector"`
	Embedder string `json:"embedder"`
	// Fingerprint identifies the model that made the vectors in the
	// gallery, and Dim is their length.
	Fingerprint string `json:"fingerprint"`
	Dim         int    `json:"dim"`
	// Runtime is the version of ONNX Runtime, Threads how many processor
	// threads an analysis may use and Kernel the code that compares
	// vectors.
	Runtime string `json:"runtime"`
	Threads int    `json:"threads"`
	Kernel  string `json:"kernel"`
	// Thresholds are the similarities the server requires, Gallery
	// describes the people enrolled and Limits what a request may ask for.
	Thresholds Thresholds  `json:"thresholds"`
	Gallery    GalleryInfo `json:"gallery"`
	Limits     Limits      `json:"limits"`
	// RequestID is the id under which the server logged the request.
	RequestID string `json:"request_id,omitempty"`
}

// Thresholds are the similarities from which a face counts as a subject's:
// Search when it is looked up among everyone, Verify when it is checked
// against one claimed subject.
type Thresholds struct {
	Search float64 `json:"search"`
	Verify float64 `json:"verify"`
}

// GalleryInfo describes the enrolled people as a whole.
type GalleryInfo struct {
	// Subjects is how many people are enrolled and Templates how many
	// faces they have between them.
	Subjects  int `json:"subjects"`
	Templates int `json:"templates"`
	// Encrypted says whether the store is encrypted at rest and Created
	// when it was made.
	Encrypted bool      `json:"encrypted"`
	Created   time.Time `json:"created"`
}

// Limits are the bounds a server puts on its requests.
type Limits struct {
	// MaxBody is the largest request body, in bytes.
	MaxBody int64 `json:"max_body"`
	// MaxInflight is how many pictures are analysed at the same time,
	// Queue how many more requests may wait for a free slot and
	// QueueTimeout the longest such a wait may be. A request that would
	// exceed either is refused as overloaded.
	MaxInflight  int           `json:"max_inflight"`
	Queue        int           `json:"queue"`
	QueueTimeout time.Duration `json:"-"`
	// MaxTemplates is how many templates a subject may hold; 0 for no
	// bound.
	MaxTemplates int `json:"max_templates"`
	// SearchLimit is the most subjects a search names for a face.
	SearchLimit int `json:"search_limit"`
	// Candidates says whether a search may ask for subjects below the
	// threshold.
	Candidates bool `json:"candidates"`
}

// limitsWire is Limits as the API writes it, with the queue timeout in
// milliseconds.
type limitsWire struct {
	*plainLimits
	QueueTimeoutMS float64 `json:"queue_timeout_ms"`
}

// plainLimits is Limits without its methods, so that encoding it does not
// come back here.
type plainLimits Limits

// UnmarshalJSON reads the limits of the API, whose queue timeout counts in
// milliseconds.
func (l *Limits) UnmarshalJSON(data []byte) error {
	wire := limitsWire{plainLimits: (*plainLimits)(l)}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	l.QueueTimeout = fromMillis(wire.QueueTimeoutMS)
	return nil
}

// MarshalJSON writes the limits as the API does.
func (l Limits) MarshalJSON() ([]byte, error) {
	return json.Marshal(limitsWire{plainLimits: (*plainLimits)(&l), QueueTimeoutMS: millis(l.QueueTimeout)})
}

// Info asks the server what it runs and holds. It needs a search key.
func (c *Client) Info(ctx context.Context) (*Info, error) {
	res := new(Info)
	id, err := c.do(ctx, &call{method: http.MethodGet, path: "/v1/info", repeatable: true}, res)
	if err != nil {
		return nil, err
	}
	res.RequestID = id
	return res, nil
}

// Ready reports whether the server wants requests: nil if it does, an
// *Error with CodeShuttingDown once it has been told to stop, and the
// failure of the connection if it cannot be reached. It needs no key. As a
// probe it is sent once, whatever WithRetries says.
func (c *Client) Ready(ctx context.Context) error {
	_, err := c.do(ctx, &call{method: http.MethodGet, path: "/readyz", once: true}, nil)
	return err
}

// Alive reports whether the server process answers at all, ready or not.
// It needs no key and, like Ready, is sent once.
func (c *Client) Alive(ctx context.Context) error {
	_, err := c.do(ctx, &call{method: http.MethodGet, path: "/healthz", once: true}, nil)
	return err
}
