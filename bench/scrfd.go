package main

import "sort"

type Face struct {
	X1, Y1, X2, Y2 float32
	Score          float32
	Kps            [5][2]float32
}

type scrfdHead struct {
	stride           int
	score, bbox, kps int // output indices
}

// scrfdLayout groups the 9 SCRFD outputs (score/bbox/kps x stride 8/16/32) by
// their last dimension, largest anchor count first.
func scrfdLayout(m *Model) []scrfdHead {
	type out struct{ idx, n int }
	groups := map[int64][]out{}
	for i, s := range m.outShapes {
		last := s[len(s)-1]
		total := int64(1)
		for _, d := range s {
			total *= d
		}
		groups[last] = append(groups[last], out{i, int(total / last)})
	}
	for _, g := range groups {
		sort.Slice(g, func(a, b int) bool { return g[a].n > g[b].n })
	}
	strides := []int{8, 16, 32}
	var heads []scrfdHead
	for i := 0; i < len(strides) && i < len(groups[1]); i++ {
		heads = append(heads, scrfdHead{strides[i], groups[1][i].idx, groups[4][i].idx, groups[10][i].idx})
	}
	return heads
}

// decodeSCRFD turns raw head outputs into faces in input-tensor coordinates.
func decodeSCRFD(b *Binding, heads []scrfdHead, thr, iou float32) []Face {
	size := b.m.Spec.Size
	var faces []Face
	for _, h := range heads {
		fw := size / h.stride
		scores, boxes, kps := b.Out[h.score], b.Out[h.bbox], b.Out[h.kps]
		st := float32(h.stride)
		for i, sc := range scores {
			if sc < thr {
				continue
			}
			cell := i / 2 // two anchors per cell
			cx, cy := float32(cell%fw)*st, float32(cell/fw)*st
			f := Face{
				X1: cx - boxes[i*4]*st, Y1: cy - boxes[i*4+1]*st,
				X2: cx + boxes[i*4+2]*st, Y2: cy + boxes[i*4+3]*st,
				Score: sc,
			}
			for k := 0; k < 5; k++ {
				f.Kps[k] = [2]float32{cx + kps[i*10+2*k]*st, cy + kps[i*10+2*k+1]*st}
			}
			faces = append(faces, f)
		}
	}
	sort.Slice(faces, func(a, c int) bool { return faces[a].Score > faces[c].Score })
	keep := faces[:0]
	for _, f := range faces {
		ok := true
		for _, k := range keep {
			if overlap(f, k) > iou {
				ok = false
				break
			}
		}
		if ok {
			keep = append(keep, f)
		}
	}
	return keep
}

func overlap(a, b Face) float32 {
	w := min(a.X2, b.X2) - max(a.X1, b.X1)
	h := min(a.Y2, b.Y2) - max(a.Y1, b.Y1)
	if w <= 0 || h <= 0 {
		return 0
	}
	inter := w * h
	return inter / ((a.X2-a.X1)*(a.Y2-a.Y1) + (b.X2-b.X1)*(b.Y2-b.Y1) - inter)
}
