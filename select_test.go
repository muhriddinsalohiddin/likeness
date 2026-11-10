package likeness

import (
	"testing"

	"github.com/muhriddinsalohiddin/likeness/internal/detect"
	"github.com/muhriddinsalohiddin/likeness/internal/quality"
)

func cand(side, score float32, reason quality.Reason) candidate {
	return candidate{face: detect.Face{Score: score}, reason: reason, side: side}
}

func TestRankPutsUsableFacesFirst(t *testing.T) {
	cands := []candidate{
		cand(90, 0.9, quality.BadPose), // the biggest, but unusable
		cand(40, 0.9, quality.OK),
		cand(70, 0.9, quality.OK),
		cand(95, 0.6, quality.TooSmall),
		cand(70, 0.8, quality.OK), // same size as an earlier one: order kept
	}
	rank(cands)
	wantSide := []float32{70, 70, 40, 95, 90}
	wantScore := []float32{0.9, 0.8, 0.9, 0.6, 0.9}
	for i := range cands {
		if cands[i].side != wantSide[i] || cands[i].face.Score != wantScore[i] {
			t.Fatalf("position %d: side %v score %v, want side %v score %v", i, cands[i].side, cands[i].face.Score, wantSide[i], wantScore[i])
		}
	}
	rank(nil) // nothing to do, nothing to break
}

func TestRivalled(t *testing.T) {
	p := quality.Policy{MinScore: 0.7, MaxRival: 0.5}
	main := cand(100, 0.95, quality.OK)
	for name, c := range map[string]struct {
		cands []candidate
		p     quality.Policy
		want  bool
	}{
		"alone":                         {[]candidate{main}, p, false},
		"small bystander":               {[]candidate{main, cand(40, 0.9, quality.OK)}, p, false},
		"second person":                 {[]candidate{main, cand(60, 0.9, quality.OK)}, p, true},
		"second person in profile":      {[]candidate{main, cand(80, 0.9, quality.BadPose)}, p, true},
		"large but doubtful detection":  {[]candidate{main, cand(90, 0.6, quality.OK)}, p, false},
		"rival behind a bystander":      {[]candidate{main, cand(30, 0.9, quality.OK), cand(55, 0.9, quality.OK)}, p, true},
		"no limit set":                  {[]candidate{main, cand(99, 0.9, quality.OK)}, quality.Policy{MinScore: 0.7}, false},
		"main face unusable":            {[]candidate{cand(100, 0.95, quality.TooSmall), cand(90, 0.9, quality.TooSmall)}, p, false},
		"low policy score, weak rival":  {[]candidate{main, cand(90, 0.45, quality.OK)}, quality.Policy{MaxRival: 0.5}, false},
		"low policy score, solid rival": {[]candidate{main, cand(90, 0.55, quality.OK)}, quality.Policy{MaxRival: 0.5}, true},
	} {
		if got := rivalled(c.cands, c.p); got != c.want {
			t.Errorf("%s: got %v, want %v", name, got, c.want)
		}
	}
}

func TestBest(t *testing.T) {
	good := Face{Embedding: []float32{1}}
	dim := Face{Reject: TooDark}
	for name, c := range map[string]struct {
		res    Result
		face   bool
		reason Reason
	}{
		"no faces, reason given":    {Result{Reject: NoFace}, false, NoFace},
		"no faces, no reason":       {Result{}, false, NoFace},
		"usable face":               {Result{Faces: []Face{good, dim}}, true, Accepted},
		"unusable face":             {Result{Faces: []Face{dim}}, true, TooDark},
		"picture turned down":       {Result{Faces: []Face{good}, Reject: MultipleFaces}, true, MultipleFaces},
		"picture outranks the face": {Result{Faces: []Face{dim}, Reject: MultipleFaces}, true, MultipleFaces},
	} {
		f, r := c.res.Best()
		if (f != nil) != c.face || r != c.reason {
			t.Errorf("%s: face=%v reason=%q, want face=%v reason=%q", name, f != nil, r, c.face, c.reason)
		}
	}
	if b := (Box{Left: 10, Top: 20, Right: 50, Bottom: 80}); b.Width() != 40 || b.Height() != 60 {
		t.Errorf("box size %v x %v", b.Width(), b.Height())
	}
}

func TestPoliciesAreOrdered(t *testing.T) {
	s, e := SearchPolicy(), EnrolmentPolicy()
	// enrolment must never accept what search would turn down
	if e.MinScore < s.MinScore || e.MinEyeDistance < s.MinEyeDistance || e.MinInFrame < s.MinInFrame ||
		e.MinBrightness < s.MinBrightness || e.MaxBrightness > s.MaxBrightness || e.MinSharpness < s.MinSharpness ||
		e.MaxYaw > s.MaxYaw || e.MaxPitch > s.MaxPitch {
		t.Fatalf("the enrolment policy is looser than the search policy somewhere:\n enrol  %+v\n search %+v", e, s)
	}
	if e.MaxRival <= 0 || e.MaxRoll <= 0 {
		t.Fatal("enrolment should limit rivals and roll")
	}
	if o := ForEnrolment(); o.Faces != Single || o.Policy != e {
		t.Fatalf("ForEnrolment() = %+v", o)
	}
	if o := ForSearch(); o.Faces != Largest || o.Policy != s {
		t.Fatalf("ForSearch() = %+v", o)
	}
}
