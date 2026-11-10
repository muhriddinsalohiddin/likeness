package likeness

import (
	"context"
	"testing"

	"github.com/muhriddinsalohiddin/likeness/internal/fixtures"
)

// One search request, start to finish, for someone at a webcam: the picture
// arrives as a JPEG and leaves as a vector.
func BenchmarkAnalyze(b *testing.B) {
	near, err := fixtures.Closeup("hathaway")
	if err != nil {
		b.Fatal(err)
	}
	// the same person in a 1280x720 frame, as a webcam would deliver it
	frame := encodeJPEG(b, onCanvas(portrait(b), 1, 1280, 720), 90)

	for _, preset := range presets {
		for _, pic := range []struct {
			name string
			data []byte
		}{{"closeup448", near}, {"frame720p", frame}} {
			b.Run(preset+"/"+pic.name, func(b *testing.B) {
				e := open(b, preset)
				var t Timing
				b.SetBytes(int64(len(pic.data)))
				b.ResetTimer()
				for b.Loop() {
					res, err := e.Analyze(context.Background(), pic.data, ForSearch())
					if err != nil {
						b.Fatal(err)
					}
					if _, reason := res.Best(); reason != Accepted {
						b.Fatal(reason)
					}
					t.Decode += res.Timing.Decode
					t.Detect += res.Timing.Detect
					t.Align += res.Timing.Align
					t.Embed += res.Timing.Embed
				}
				n := float64(b.N)
				b.ReportMetric(float64(t.Decode.Microseconds())/n/1000, "decode-ms")
				b.ReportMetric(float64(t.Detect.Microseconds())/n/1000, "detect-ms")
				b.ReportMetric(float64(t.Align.Microseconds())/n/1000, "align-ms")
				b.ReportMetric(float64(t.Embed.Microseconds())/n/1000, "embed-ms")
			})
		}
	}
}
