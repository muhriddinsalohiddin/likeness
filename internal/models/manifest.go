// Package models describes the neural networks likeness can run and fetches
// them. No weights ship with the source: a manifest names a file, where to
// get it, its checksum and how the engine must feed it.
package models

import (
	"fmt"
	"runtime"

	"github.com/muhriddinsalohiddin/likeness/internal/vision"
)

// Kind tells the engine which decoder and preprocessing a model needs.
type Kind string

const (
	// SCRFD is the InsightFace detector family: three strides, two anchors
	// per cell, distance-coded boxes and five landmarks.
	SCRFD Kind = "scrfd"
	// YuNet is the OpenCV Zoo detector: three strides, one anchor per cell,
	// separate objectness and class scores.
	YuNet Kind = "yunet"
	// ArcFace is an InsightFace-style embedder: 112x112 RGB crop normalised
	// to [-1, 1].
	ArcFace Kind = "arcface"
	// SFace is the OpenCV Zoo embedder: 112x112 BGR crop, raw 0..255 values.
	SFace Kind = "sface"
)

// Detector reports whether models of this kind find faces (as opposed to
// describing them).
func (k Kind) Detector() bool { return k == SCRFD || k == YuNet }

// Artifact is something to download: a plain file, or one member of an
// archive.
type Artifact struct {
	URL    string
	SHA256 string // of the file that ends up on disk
	Size   int64  // of that file, in bytes
	// Member is the path inside the archive at URL when the artifact is
	// packed; empty means URL is the file itself.
	Member string
}

// Manifest describes one model.
type Manifest struct {
	// Name identifies the model in configuration and in fingerprints.
	Name string
	Kind Kind
	// File is the name the model is stored under in the models directory.
	File     string
	Artifact Artifact
	// Input is the side of the square network input. Detectors with dynamic
	// shapes accept other sizes; Dynamic says so.
	Input   int
	Dynamic bool
	Norm    vision.Norm
	// Embedding is the vector length of an embedder, 0 for detectors.
	Embedding int
	// License is the SPDX identifier or a short name of the weights' licence.
	License string
	// NonCommercial marks weights that may only be used for research; the
	// tool asks for explicit consent before fetching them.
	NonCommercial bool
	Source        string // where the weights came from, for the docs
}

// Fingerprint identifies the weights, so that embeddings computed by one
// model are never compared with another's.
func (m *Manifest) Fingerprint() string {
	return fmt.Sprintf("%s:%s:%s", m.Kind, m.Name, m.Artifact.SHA256[:16])
}

// Preset is a detector and an embedder that work well together.
type Preset struct {
	Name        string
	Description string
	Detector    *Manifest
	Embedder    *Manifest
}

// NonCommercial reports whether either model carries a research-only licence.
func (p Preset) NonCommercial() bool {
	return p.Detector.NonCommercial || p.Embedder.NonCommercial
}

var (
	yunet = &Manifest{
		Name: "yunet_2023mar", Kind: YuNet,
		File: "face_detection_yunet_2023mar.onnx",
		Artifact: Artifact{
			URL:    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
			SHA256: "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
			Size:   232589,
		},
		Input: 640, Norm: vision.Uniform(0, 1, true),
		License: "MIT", Source: "OpenCV Zoo (Shiqi Yu et al.)",
	}
	sface = &Manifest{
		Name: "sface_2021dec", Kind: SFace,
		File: "face_recognition_sface_2021dec.onnx",
		Artifact: Artifact{
			URL:    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
			SHA256: "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
			Size:   38696353,
		},
		// OpenCV's FaceRecognizerSF swaps the channels of its BGR frames, so
		// the network sees RGB; the detector, by contrast, is fed BGR as is.
		Input: 112, Norm: vision.Uniform(0, 1, false), Embedding: 128,
		License: "Apache-2.0", Source: "OpenCV Zoo (Yaoyao Zhong et al.)",
	}
	det500m = &Manifest{
		Name: "det_500m", Kind: SCRFD, File: "det_500m.onnx",
		Artifact: Artifact{
			URL:    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
			Member: "det_500m.onnx",
			SHA256: "5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a",
			Size:   2524817,
		},
		Input: 640, Dynamic: true, Norm: vision.Uniform(127.5, 128, false),
		License: "non-commercial research", NonCommercial: true, Source: "InsightFace buffalo_s",
	}
	det10g = &Manifest{
		Name: "det_10g", Kind: SCRFD, File: "det_10g.onnx",
		Artifact: Artifact{
			URL:    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
			Member: "det_10g.onnx",
			SHA256: "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
			Size:   16923827,
		},
		Input: 640, Dynamic: true, Norm: vision.Uniform(127.5, 128, false),
		License: "non-commercial research", NonCommercial: true, Source: "InsightFace buffalo_l",
	}
	w600kMBF = &Manifest{
		Name: "w600k_mbf", Kind: ArcFace, File: "w600k_mbf.onnx",
		Artifact: Artifact{
			URL:    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
			Member: "w600k_mbf.onnx",
			SHA256: "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
			Size:   13616099,
		},
		Input: 112, Norm: vision.Uniform(127.5, 127.5, false), Embedding: 512,
		License: "non-commercial research", NonCommercial: true, Source: "InsightFace buffalo_s",
	}
	w600kR50 = &Manifest{
		Name: "w600k_r50", Kind: ArcFace, File: "w600k_r50.onnx",
		Artifact: Artifact{
			URL:    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
			Member: "w600k_r50.onnx",
			SHA256: "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
			Size:   174383860,
		},
		Input: 112, Norm: vision.Uniform(127.5, 127.5, false), Embedding: 512,
		License: "non-commercial research", NonCommercial: true, Source: "InsightFace buffalo_l",
	}
)

// All lists every known model.
var All = []*Manifest{yunet, sface, det500m, det10g, w600kMBF, w600kR50}

// Presets lists the model pairs the tool knows how to set up.
var Presets = []Preset{
	{
		Name:        "default",
		Description: "YuNet detector and SFace embedder from the OpenCV Zoo (MIT and Apache-2.0 weights)",
		Detector:    yunet, Embedder: sface,
	},
	{
		Name:        "buffalo_s",
		Description: "InsightFace SCRFD-500M detector and MobileFaceNet embedder (research-only weights)",
		Detector:    det500m, Embedder: w600kMBF,
	},
	{
		Name:        "buffalo_l",
		Description: "InsightFace SCRFD-10G detector and ResNet50 embedder (research-only weights; needs a GPU to stay under 100 ms)",
		Detector:    det10g, Embedder: w600kR50,
	},
}

// Lookup finds a model by name.
func Lookup(name string) (*Manifest, bool) {
	for _, m := range All {
		if m.Name == name {
			return m, true
		}
	}
	return nil, false
}

// LookupPreset finds a preset by name.
func LookupPreset(name string) (Preset, bool) {
	for _, p := range Presets {
		if p.Name == name {
			return p, true
		}
	}
	return Preset{}, false
}

// RuntimeVersion is the ONNX Runtime release the tool installs.
const RuntimeVersion = "1.29.0"

var runtimes = map[string]Artifact{
	"linux/amd64": {
		URL:    "https://github.com/microsoft/onnxruntime/releases/download/v1.29.0/onnxruntime-linux-x64-1.29.0.tgz",
		Member: "onnxruntime-linux-x64-1.29.0/lib/libonnxruntime.so.1.29.0",
		SHA256: "5715f06d8992ca8eeeddcce43df3a7d38f97d537052126f558e912cb312460ca",
		Size:   28497752,
	},
	"linux/arm64": {
		URL:    "https://github.com/microsoft/onnxruntime/releases/download/v1.29.0/onnxruntime-linux-aarch64-1.29.0.tgz",
		Member: "onnxruntime-linux-aarch64-1.29.0/lib/libonnxruntime.so.1.29.0",
		SHA256: "a27d21126db312aa8f02f3d5eaebe466e991f51f469882e6d0407d5a8b64afda",
		Size:   24538024,
	},
	"darwin/arm64": {
		URL:    "https://github.com/microsoft/onnxruntime/releases/download/v1.29.0/onnxruntime-osx-arm64-1.29.0.tgz",
		Member: "onnxruntime-osx-arm64-1.29.0/lib/libonnxruntime.1.29.0.dylib",
		SHA256: "68f6e54e695583adc371aef610ec4abb1ffaa3df656582922de7690f7e2000eb",
		Size:   43184400,
	},
	"windows/amd64": {
		URL:    "https://github.com/microsoft/onnxruntime/releases/download/v1.29.0/onnxruntime-win-x64-1.29.0.zip",
		Member: "onnxruntime-win-x64-1.29.0/lib/onnxruntime.dll",
		SHA256: "69d8e6d3879a3b4001cdc74c8ed9ccc7e7f799a5b847059738323404519ec471",
		Size:   16149344,
	},
}

// Runtime returns the ONNX Runtime library artifact for a platform, named
// as Go does: "linux/amd64".
func Runtime(platform string) (Artifact, bool) {
	a, ok := runtimes[platform]
	return a, ok
}

// HostRuntime returns the runtime artifact for the machine this code runs on.
func HostRuntime() (Artifact, bool) {
	return Runtime(runtime.GOOS + "/" + runtime.GOARCH)
}
