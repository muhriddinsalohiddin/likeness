package detect

import (
	"math"
	"sort"

	"github.com/muhriddinsalohiddin/likeness/internal/align"
	"github.com/muhriddinsalohiddin/likeness/internal/ortx"
)

var strides = [3]int{8, 16, 32}

// decodeSCRFD reads the nine SCRFD outputs: scores, boxes and landmarks for
// strides 8, 16 and 32, two anchors per cell. The exporter names outputs by
// number, so they are told apart by shape: the last dimension is 1, 4 or 10
// and the anchor count halves with every stride.
func decodeSCRFD(out *ortx.Outputs, size int, thr float32, faces []Face) []Face {
	type head struct{ idx, n int }
	byWidth := map[int64][]head{}
	for i, dims := range out.Dims {
		if len(dims) == 0 {
			continue
		}
		last := dims[len(dims)-1]
		byWidth[last] = append(byWidth[last], head{i, len(out.Data[i]) / int(last)})
	}
	for _, g := range byWidth {
		sort.Slice(g, func(a, b int) bool { return g[a].n > g[b].n })
	}
	scores, boxes, kps := byWidth[1], byWidth[4], byWidth[10]
	if len(scores) != 3 || len(boxes) != 3 || len(kps) != 3 {
		return faces
	}
	for s, stride := range strides {
		cells := size / stride
		anchors := scores[s].n / (cells * cells)
		if anchors == 0 || scores[s].n != boxes[s].n || scores[s].n != kps[s].n {
			continue
		}
		sc, bx, kp := out.Data[scores[s].idx], out.Data[boxes[s].idx], out.Data[kps[s].idx]
		st := float32(stride)
		for i, score := range sc {
			if score < thr {
				continue
			}
			cell := i / anchors
			cx, cy := float32(cell%cells)*st, float32(cell/cells)*st
			f := Face{
				Box:   [4]float32{cx - bx[i*4]*st, cy - bx[i*4+1]*st, cx + bx[i*4+2]*st, cy + bx[i*4+3]*st},
				Score: score,
			}
			for k := range f.Landmarks {
				f.Landmarks[k] = align.Point{X: cx + kp[i*10+2*k]*st, Y: cy + kp[i*10+2*k+1]*st}
			}
			faces = append(faces, f)
		}
	}
	return faces
}

var yunetOutputs = []string{
	"cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32",
	"bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32",
}

// decodeYuNet reads the twelve YuNet outputs, one anchor per cell: the score
// is the geometric mean of class and objectness, boxes are centre offsets
// with log-scaled sizes, landmarks are cell offsets.
func decodeYuNet(out *ortx.Outputs, index map[string]int, size int, thr float32, faces []Face) []Face {
	for s, stride := range strides {
		cls, obj := out.Data[index[yunetOutputs[s]]], out.Data[index[yunetOutputs[3+s]]]
		bx, kp := out.Data[index[yunetOutputs[6+s]]], out.Data[index[yunetOutputs[9+s]]]
		cells := size / stride
		if len(cls) != cells*cells || len(obj) != len(cls) || len(bx) != 4*len(cls) || len(kp) != 10*len(cls) {
			continue
		}
		st := float32(stride)
		for i := range cls {
			score := float32(math.Sqrt(float64(clamp01(cls[i]) * clamp01(obj[i]))))
			if score < thr {
				continue
			}
			col, row := float32(i%cells), float32(i/cells)
			cx, cy := (col+bx[i*4])*st, (row+bx[i*4+1])*st
			w, h := float32(math.Exp(float64(bx[i*4+2])))*st, float32(math.Exp(float64(bx[i*4+3])))*st
			f := Face{Box: [4]float32{cx - w/2, cy - h/2, cx + w/2, cy + h/2}, Score: score}
			for k := range f.Landmarks {
				f.Landmarks[k] = align.Point{X: (kp[i*10+2*k] + col) * st, Y: (kp[i*10+2*k+1] + row) * st}
			}
			faces = append(faces, f)
		}
	}
	return faces
}

func clamp01(v float32) float32 { return min(max(v, 0), 1) }

// nms keeps the best-scoring face of every group of overlapping boxes. The
// result is ordered by score and reuses the input slice.
func nms(faces []Face, iou float32) []Face {
	sort.Slice(faces, func(a, b int) bool { return faces[a].Score > faces[b].Score })
	keep := faces[:0]
	for _, f := range faces {
		suppressed := false
		for _, k := range keep {
			if overlap(f.Box, k.Box) > iou {
				suppressed = true
				break
			}
		}
		if !suppressed {
			keep = append(keep, f)
		}
	}
	return keep
}

// overlap is the intersection over union of two boxes.
func overlap(a, b [4]float32) float32 {
	w := min(a[2], b[2]) - max(a[0], b[0])
	h := min(a[3], b[3]) - max(a[1], b[1])
	if w <= 0 || h <= 0 {
		return 0
	}
	inter := w * h
	union := (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}
