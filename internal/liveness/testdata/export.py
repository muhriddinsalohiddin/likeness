# Converts the Silent-Face anti-spoofing models to ONNX with the code of
# their own repository, and records what that code makes of a few pictures:
# the reference our implementation is tested against.
import io, json, os, sys, tarfile, urllib.request, hashlib
import numpy as np, cv2, torch

OUT, FIX, REAL, YUNET = "/out", "/fixtures", "/real", "/models/face_detection_yunet_2023mar.onnx"
REPO = "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/archive/refs/heads/master.tar.gz"

def fetch(url):
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:
            print("retry", attempt, e, flush=True)
    raise SystemExit("cannot fetch " + url)

tar = tarfile.open(fileobj=io.BytesIO(fetch(REPO)))
tar.extractall("/work")
repo = "/work/Silent-Face-Anti-Spoofing-master"
sys.path.insert(0, repo)
from src.model_lib.MiniFASNet import MiniFASNetV2, MiniFASNetV1SE
from src.utility import get_kernel, parse_model_name
from src.generate_patches import CropImage

KINDS = {"2.7_80x80_MiniFASNetV2.pth": ("minifasnet_v2", MiniFASNetV2),
         "4_0_0_80x80_MiniFASNetV1SE.pth": ("minifasnet_v1se", MiniFASNetV1SE)}
nets = {}
for file, (name, cls) in KINDS.items():
    h, w, kind, scale = parse_model_name(file)
    net = cls(conv6_kernel=get_kernel(h, w))
    state = torch.load(os.path.join(repo, "resources/anti_spoof_models", file), map_location="cpu")
    net.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in state.items()})
    net.eval()
    path = os.path.join(OUT, name + ".onnx")
    torch.onnx.export(net, torch.zeros(1, 3, h, w), path, input_names=["input"], output_names=["logits"],
                      opset_version=13, do_constant_folding=True)
    data = open(path, "rb").read()
    print(name, "scale", scale, "size", len(data), "sha256", hashlib.sha256(data).hexdigest(), flush=True)
    nets[name] = (net, scale, h, w)

# their ToTensor: channels first, float, and no division by 255
cropper = CropImage()
def to_tensor(crop):
    return torch.from_numpy(crop.transpose((2, 0, 1))).float()

def score(img, box, keep=None):
    """box is x, y, w, h. Returns per model the logits, and the mean probability of 'real'."""
    out, total = {}, np.zeros(3)
    for name, (net, scale, h, w) in nets.items():
        crop = cropper.crop(org_img=img, bbox=box, scale=scale, out_w=w, out_h=h, crop=True)
        if keep is not None:
            cv2.imwrite(os.path.join(OUT, "%s_%s.png" % (keep, name)), crop)
        with torch.no_grad():
            logits = net(to_tensor(crop).unsqueeze(0)).numpy()[0]
        e = np.exp(logits - logits.max()); p = e / e.sum()
        out[name] = [float(v) for v in logits]
        total += p
    return out, float(total[1] / len(nets))

# the two close-ups of the test fixtures, with boxes our own detector gives
ref = []
for name, box in {"hathaway": [151, 116, 145, 211], "cassada": [146, 125, 144, 199]}.items():
    img = cv2.imread(os.path.join(FIX, "closeups", name + ".jpg"))
    logits, real = score(img, box, keep=name)
    ref.append({"picture": name, "box": box, "logits": logits, "real": real})
    print("fixture", name, "real %.4f" % real, flush=True)
json.dump(ref, open(os.path.join(OUT, "liveness.json"), "w"), indent=1)

# what the old service's chain (YuNet box, then these models) says of real webcam frames
if os.path.isdir(REAL):
    det = cv2.FaceDetectorYN.create(YUNET, "", (0, 0), 0.9, 0.3, 5000)
    for person in sorted(os.listdir(REAL)):
        vals = []
        for f in sorted(os.listdir(os.path.join(REAL, person))):
            img = cv2.imread(os.path.join(REAL, person, f))
            det.setInputSize((img.shape[1], img.shape[0]))
            _, faces = det.detect(img)
            if faces is None:
                vals.append(None); continue
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])[:4]
            _, real = score(img, [int(x), int(y), int(w), int(h)])
            vals.append(round(real, 3))
        print("real frames", person, vals, flush=True)
