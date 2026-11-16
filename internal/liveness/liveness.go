// Package liveness tells a face in front of the camera from a picture of
// one: a print, or a photograph or video on a screen.
//
// It runs the Silent-Face models of Minivision (MiniFASNet). Each looks at
// the face with a different amount of its surroundings, where the edge of a
// sheet of paper or the bezel of a telephone shows, and says which of three
// things it sees; the second is a live face. Their answers are averaged.
//
// What they judge is one frame. They catch the casual attempt, a picture
// held up to the camera, and can be fooled by a good screen or a video fed
// to the browser in place of the camera.
package liveness

import (
	"errors"
	"fmt"
	"math"
	"path/filepath"

	"github.com/muhriddinsalohiddin/likeness/internal/align"
	"github.com/muhriddinsalohiddin/likeness/internal/models"
	"github.com/muhriddinsalohiddin/likeness/internal/ortx"
	"github.com/muhriddinsalohiddin/likeness/internal/vision"
)

// classes is what a model chooses among; live is the one that means a
// person is in front of the camera.
const (
	classes = 3
	live    = 1
)

// Checker is safe for concurrent use.
type Checker struct {
	nets []*net
}

type net struct {
	m    *models.Manifest
	sess *ortx.Session
	size int
	pool ortx.Keep[worker]
}

type worker struct {
	in  *ortx.Tensor
	out ortx.Outputs
}

// New loads the models described by ms from dir.
func New(rt *ortx.Runtime, ms []*models.Manifest, dir string) (*Checker, error) {
	if len(ms) == 0 {
		return nil, errors.New("liveness: no models")
	}
	c := &Checker{}
	for _, m := range ms {
		n, err := load(rt, m, dir)
		if err != nil {
			c.Close()
			return nil, err
		}
		c.nets = append(c.nets, n)
	}
	return c, nil
}

func load(rt *ortx.Runtime, m *models.Manifest, dir string) (*net, error) {
	if m.Kind != models.FASNet || m.Scale <= 0 {
		return nil, fmt.Errorf("liveness: %s is not an anti-spoofing model", m.Name)
	}
	sess, err := rt.NewSessionFromFile(filepath.Join(dir, m.File), ortx.SessionOptions{})
	if err != nil {
		return nil, err
	}
	in, out := sess.Inputs(), sess.Outputs()
	if len(in) != 1 || len(in[0].Dims) != 4 || in[0].Dims[1] != 3 || in[0].Dims[2] != int64(m.Input) || in[0].Dims[3] != int64(m.Input) ||
		len(out) != 1 || len(out[0].Dims) != 2 || out[0].Dims[1] != classes {
		sess.Close()
		return nil, fmt.Errorf("liveness: %s: expected a %dx%d picture in and %d scores out, got %+v and %+v", m.Name, m.Input, m.Input, classes, in, out)
	}
	return &net{m: m, sess: sess, size: m.Input}, nil
}

// Names lists the models in use.
func (c *Checker) Names() []string {
	names := make([]string, len(c.nets))
	for i, n := range c.nets {
		names[i] = n.m.Name
	}
	return names
}

// Close releases the models. It must not overlap with Real.
func (c *Checker) Close() {
	for _, n := range c.nets {
		n.pool.Drain(func(w *worker) { w.in.Close() })
		n.sess.Close()
	}
	c.nets = nil
}

// Real is the probability, from 0 to 1, that the face in the box (left,
// top, right, bottom, in picture pixels) is a live one.
func (c *Checker) Real(img *vision.Image, box [4]float32) (float32, error) {
	if img == nil {
		return 0, errors.New("liveness: nil image")
	}
	var sum float64
	for _, n := range c.nets {
		p, err := n.real(img, box)
		if err != nil {
			return 0, fmt.Errorf("liveness: %s: %w", n.m.Name, err)
		}
		sum += p
	}
	return float32(sum / float64(len(c.nets))), nil
}

func (n *net) real(img *vision.Image, box [4]float32) (float64, error) {
	w := n.pool.Get()
	if w == nil {
		t, err := ortx.NewFloat32(1, 3, int64(n.size), int64(n.size))
		if err != nil {
			return 0, err
		}
		w = &worker{in: t}
	}
	defer n.pool.Put(w)

	width, height := img.Size()
	x1, y1, x2, y2 := surroundings(box, width, height, n.m.Scale)
	// The models were trained on crops made with OpenCV's resize, which
	// samples without smoothing: they get the same.
	if err := img.Warp(w.in.Float32(), n.size, resize(x1, y1, x2, y2, n.size), n.m.Norm, vision.Options{NoAntiAlias: true}); err != nil {
		return 0, err
	}
	if err := n.sess.Run([]*ortx.Tensor{w.in}, &w.out); err != nil {
		return 0, err
	}
	if len(w.out.Data) != 1 || len(w.out.Data[0]) != classes {
		return 0, fmt.Errorf("unexpected output %v", w.out.Dims)
	}
	return softmax(w.out.Data[0])[live], nil
}

// surroundings is the part of the picture a model looks at: the face box
// grown about its centre by scale, moved back inside the picture where it
// sticks out, and grown less when the picture is too small for it. The
// corners are whole pixels, both included. This follows the reference
// implementation step by step, truncations and all: the models know no
// other framing.
func surroundings(box [4]float32, width, height int, scale float64) (x1, y1, x2, y2 int) {
	x, y := float64(int(box[0])), float64(int(box[1]))
	w, h := float64(int(box[2]-box[0])), float64(int(box[3]-box[1]))
	if w < 1 || h < 1 {
		return 0, 0, width - 1, height - 1
	}
	s := math.Min(float64(height-1)/h, math.Min(float64(width-1)/w, scale))
	nw, nh := w*s, h*s
	cx, cy := w/2+x, h/2+y
	left, top := cx-nw/2, cy-nh/2
	right, bottom := cx+nw/2, cy+nh/2
	if left < 0 {
		right -= left
		left = 0
	}
	if top < 0 {
		bottom -= top
		top = 0
	}
	if right > float64(width-1) {
		left -= right - float64(width) + 1
		right = float64(width - 1)
	}
	if bottom > float64(height-1) {
		top -= bottom - float64(height) + 1
		bottom = float64(height - 1)
	}
	return int(left), int(top), int(right), int(bottom)
}

// resize maps the pixels x1..x2 by y1..y2 of the picture onto a square of
// the given side, the centre of the first pixel onto the centre of the
// first, as an image resize does.
func resize(x1, y1, x2, y2, side int) align.Affine {
	sx := float32(side) / float32(x2-x1+1)
	sy := float32(side) / float32(y2-y1+1)
	return align.Affine{
		A: sx, C: (0.5-float32(x1))*sx - 0.5,
		E: sy, F: (0.5-float32(y1))*sy - 0.5,
	}
}

func softmax(logits []float32) [classes]float64 {
	top := float64(logits[0])
	for _, v := range logits[1:] {
		top = math.Max(top, float64(v))
	}
	var p [classes]float64
	var sum float64
	for i, v := range logits {
		p[i] = math.Exp(float64(v) - top)
		sum += p[i]
	}
	for i := range p {
		p[i] /= sum
	}
	return p
}
