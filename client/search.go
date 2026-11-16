package client

import (
	"bytes"
	"context"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"strconv"
)

// pictureKind is what a picture is sent as. The server does not go by the
// type: it looks at the bytes.
const pictureKind = "application/octet-stream"

// SearchOptions adjust a search. The zero value asks for the server's
// defaults: the best subject for the largest face in the picture.
type SearchOptions struct {
	// Limit is how many subjects to name for each face, best first, up to
	// the SearchLimit in the server's Info. Zero leaves it to the server,
	// which names one.
	Limit int
	// Threshold is a similarity to require that is stricter than the
	// server's own; a lower one is refused, unless the server allows
	// candidates. Zero means the server's.
	Threshold float64
	// AllFaces looks at every face in the picture, up to 16, instead of the
	// largest alone.
	AllFaces bool
	// Candidates names the nearest subjects even below the threshold. Only
	// a server started with --allow-candidates does that; any other
	// answers CodeCandidatesDisabled.
	Candidates bool
}

// Match is a subject a face was found to be.
type Match struct {
	// Subject is the id the person was enrolled under.
	Subject string `json:"subject"`
	// Score is a cosine similarity: 1 for the same picture, around 0 for
	// two unrelated people.
	Score float64 `json:"score"`
	// Template is the one of the subject's templates that scored best.
	Template uint64 `json:"template"`
}

// SearchFace is a face in a searched picture and who it is.
type SearchFace struct {
	// Face is where the face is and what was measured on it.
	Face
	// Match says whether the face is somebody enrolled: Matches[0].
	Match bool `json:"match"`
	// Matches are the subjects whose similarity reaches the threshold, best
	// first, up to the limit of the search. It is empty when nobody does,
	// unless candidates were asked for.
	Matches []Match `json:"matches"`
	// Margin is how far the best subject is ahead of the next one. A match
	// with a small margin deserves less trust than its score suggests. It
	// is nil when fewer than two people are enrolled or nobody matched.
	Margin *float64 `json:"margin,omitempty"`
}

// SearchResult is the answer to a search. A picture in which nobody is
// recognised is a result, not an error.
type SearchResult struct {
	// Faces are the faces looked at, largest first: one, unless the search
	// asked for all. It is empty when no face was found.
	Faces []SearchFace `json:"faces"`
	// FacesFound is how many faces the detector saw in the picture.
	FacesFound int `json:"faces_found"`
	// Reject is set when the picture as a whole was turned down:
	// ReasonNoFace.
	Reject Reason `json:"reject,omitempty"`
	// Threshold is the similarity that was required of a match.
	Threshold float64 `json:"threshold"`
	// Width and Height are those of the upright picture, in pixels.
	Width  int `json:"width"`
	Height int `json:"height"`
	// Timing is where the server spent its time.
	Timing Timing `json:"timing_ms"`
	// RequestID is the id under which the server logged the request.
	RequestID string `json:"request_id,omitempty"`
}

// Best returns the strongest match in the picture, or nil when nobody in
// it is enrolled. A candidate below the threshold is not a match.
func (r *SearchResult) Best() *Match {
	if r == nil {
		return nil
	}
	var best *Match
	for i := range r.Faces {
		f := &r.Faces[i]
		if f.Match && len(f.Matches) > 0 && (best == nil || f.Matches[0].Score > best.Score) {
			best = &f.Matches[0]
		}
	}
	return best
}

// Search asks who is in the picture, a JPEG, PNG or WebP file. It needs a
// search key.
func (c *Client) Search(ctx context.Context, picture []byte, o SearchOptions) (*SearchResult, error) {
	query := url.Values{}
	if o.Limit != 0 {
		query.Set("limit", strconv.Itoa(o.Limit))
	}
	if o.Threshold != 0 {
		query.Set("threshold", number(o.Threshold))
	}
	if o.AllFaces {
		query.Set("faces", "all")
	}
	if o.Candidates {
		query.Set("candidates", "true")
	}
	rq := upload("/v1/search", query, picture)
	rq.repeatable = true
	res := new(SearchResult)
	id, err := c.do(ctx, rq, res)
	if err != nil {
		return nil, err
	}
	res.RequestID = id
	return res, nil
}

// VerifyOptions adjust a verification.
type VerifyOptions struct {
	// Threshold is a similarity to require that is stricter than the
	// server's own. Zero means the server's.
	Threshold float64
}

// Verification is the answer to the question whether a picture shows a
// given subject.
type Verification struct {
	// Subject is the id the picture was compared with.
	Subject string `json:"subject"`
	// Verified says whether the largest face in the picture is the
	// subject's.
	Verified bool `json:"verified"`
	// Score is the similarity to the subject's template that came closest,
	// and Template that template. Both are zero when the picture had no
	// usable face.
	Score    float64 `json:"score"`
	Template uint64  `json:"template,omitempty"`
	// Threshold is the similarity that was required.
	Threshold float64 `json:"threshold"`
	// Reject says why the picture had no usable face, if it had none.
	Reject Reason `json:"reject,omitempty"`
	// Face is the face that was compared, nil when none was found, and
	// Timing where the server spent its time.
	Face   *Face  `json:"face,omitempty"`
	Timing Timing `json:"timing_ms"`
	// RequestID is the id under which the server logged the request.
	RequestID string `json:"request_id,omitempty"`
}

// Verify asks whether the picture shows the subject. It needs a search
// key. A subject that is not enrolled is an error for which IsNotFound
// holds; a picture without a usable face is a Verification that is not
// verified and says why in Reject.
//
// The subject travels in the body of the request, not in its URL, and so
// stays out of the access logs of whatever stands between the caller and
// the server.
func (c *Client) Verify(ctx context.Context, subject string, picture []byte, o VerifyOptions) (*Verification, error) {
	if subject == "" {
		return nil, errNoSubject
	}
	fields := [][2]string{{"subject", subject}}
	if o.Threshold != 0 {
		fields = append(fields, [2]string{"threshold", number(o.Threshold)})
	}
	rq := form("/v1/verify", fields, picture)
	rq.repeatable = true
	res := new(Verification)
	id, err := c.do(ctx, rq, res)
	if err != nil {
		return nil, err
	}
	res.RequestID = id
	return res, nil
}

// upload describes a request that carries a picture as its body and its
// options in the query string.
func upload(path string, query url.Values, picture []byte) *call {
	return &call{
		method: http.MethodPost, path: path, query: query,
		kind: pictureKind, size: int64(len(picture)),
		body: func() io.Reader { return bytes.NewReader(picture) },
	}
}

// form describes a request that carries a picture and its options as a
// multipart form. The picture is not copied into the form: the body is
// read from what precedes it, the picture itself and what follows.
func form(path string, fields [][2]string, picture []byte) *call {
	var frame bytes.Buffer
	w := multipart.NewWriter(&frame)
	// Writing to a buffer cannot fail.
	for _, field := range fields {
		_ = w.WriteField(field[0], field[1])
	}
	_, _ = w.CreateFormFile("image", "picture")
	head := frame.Len()
	_ = w.Close()
	before, after := frame.Bytes()[:head], frame.Bytes()[head:]
	return &call{
		method: http.MethodPost, path: path,
		kind: w.FormDataContentType(), size: int64(len(before) + len(picture) + len(after)),
		body: func() io.Reader {
			return io.MultiReader(bytes.NewReader(before), bytes.NewReader(picture), bytes.NewReader(after))
		},
	}
}

// number writes a similarity the way the server reads it.
func number(v float64) string { return strconv.FormatFloat(v, 'f', -1, 64) }
