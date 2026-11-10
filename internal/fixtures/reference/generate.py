#!/usr/bin/env python3
"""Computes reference results for the fixture pictures with implementations
that share no code with likeness: OpenCV's FaceDetectorYN and FaceRecognizerSF
for the OpenCV Zoo models, and a NumPy port of InsightFace's SCRFD and ArcFace
pipelines for the InsightFace models.

    python generate.py --models ~/.cache/likeness/models

Needs numpy, opencv-python (>= 4.8) and onnxruntime. Writes one JSON file per
picture next to this script.
"""
import argparse
import json
import os

import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
FACES = os.path.join(HERE, "..", "faces")

ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def letterbox(img, size):
    """Scale to fit a size x size canvas, anchored top-left, zero padding.

    Returns the canvas and the affine that maps canvas coordinates back to
    picture coordinates (pixel-centre convention, as cv2.resize uses)."""
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    cw, ch = min(max(int(round(w * s)), 1), size), min(max(int(round(h * s)), 1), size)
    resized = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:ch, :cw] = resized
    inv_x, inv_y = w / cw, h / ch
    return canvas, (inv_x, 0.5 * inv_x - 0.5, inv_y, 0.5 * inv_y - 0.5)


def to_picture(pts, m):
    a, c, e, f = m
    pts = np.asarray(pts, dtype=np.float64)
    out = pts.copy()
    out[..., 0] = pts[..., 0] * a + c
    out[..., 1] = pts[..., 1] * e + f
    return out


def face_record(box, score, lm):
    return {
        "box": [round(float(v), 3) for v in box],
        "score": round(float(score), 5),
        "landmarks": [[round(float(x), 3), round(float(y), 3)] for x, y in lm],
    }


def yunet(model, img, size, thr=0.5, nms=0.3):
    det = cv2.FaceDetectorYN.create(model, "", (size, size), thr, nms, 5000)
    canvas, m = letterbox(img, size)
    det.setInputSize((size, size))
    _, faces = det.detect(canvas)
    out = []
    if faces is None:
        return out
    for f in faces:
        x, y, w, h = f[:4]
        box = to_picture([[x, y], [x + w, y + h]], m).reshape(4)
        lm = to_picture(f[4:14].reshape(5, 2), m)
        out.append(face_record(box, f[14], lm))
    return out


def scrfd(model, img, size, thr=0.5, nms=0.4):
    """NumPy port of insightface.model_zoo.scrfd.SCRFD.detect."""
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    canvas, m = letterbox(img, size)
    blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (size, size), (127.5, 127.5, 127.5), swapRB=True)
    outs = sess.run(None, {sess.get_inputs()[0].name: blob})
    boxes, scores, kpss = [], [], []
    for i, stride in enumerate((8, 16, 32)):
        sc = outs[i].reshape(-1)
        bb = outs[i + 3].reshape(-1, 4) * stride
        kp = outs[i + 6].reshape(-1, 10) * stride
        n = size // stride
        centres = np.stack(np.mgrid[:n, :n][::-1], axis=-1).astype(np.float32).reshape(-1, 2) * stride
        centres = np.stack([centres] * 2, axis=1).reshape(-1, 2)
        pos = np.where(sc >= thr)[0]
        for p in pos:
            cx, cy = centres[p]
            boxes.append([cx - bb[p, 0], cy - bb[p, 1], cx + bb[p, 2], cy + bb[p, 3]])
            scores.append(sc[p])
            kpss.append([[cx + kp[p, 2 * k], cy + kp[p, 2 * k + 1]] for k in range(5)])
    if not boxes:
        return []
    boxes, scores, kpss = np.array(boxes), np.array(scores), np.array(kpss)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        iou = inter / (area(boxes[i]) + area(boxes[order[1:]]) - inter)
        order = order[1:][iou <= nms]
    return [face_record(to_picture(boxes[i].reshape(2, 2), m).reshape(4), scores[i], to_picture(kpss[i], m)) for i in keep]


def umeyama(src, dst):
    """skimage.transform._geometric._umeyama with estimate_scale=True."""
    num, dim = src.shape
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_demean, dst_demean = src - src_mean, dst - dst_mean
    A = dst_demean.T @ src_demean / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1, dtype=np.float64)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V
    scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale
    return T


def arcface(model, img, landmarks):
    """insightface.utils.face_align.norm_crop followed by ArcFaceONNX.get_feat."""
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    M = umeyama(np.asarray(landmarks, dtype=np.float64), ARCFACE_DST.astype(np.float64))[0:2, :]
    aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
    blob = cv2.dnn.blobFromImages([aligned], 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
    feat = sess.run(None, {sess.get_inputs()[0].name: blob})[0].reshape(-1)
    return feat / np.linalg.norm(feat)


def sface(model, img, landmarks):
    rec = cv2.FaceRecognizerSF.create(model, "")
    lm = np.asarray(landmarks, dtype=np.float32).reshape(-1)
    row = np.concatenate([[0, 0, 1, 1], lm, [1.0]]).astype(np.float32)  # box is unused by alignCrop
    aligned = rec.alignCrop(img, row)
    feat = rec.feature(aligned).reshape(-1)
    return feat / np.linalg.norm(feat)


def embed_records(fn, model, img, faces):
    out = []
    for f in faces:
        vec = fn(model, img, f["landmarks"])
        out.append({"landmarks": f["landmarks"], "vector": [round(float(v), 6) for v in vec]})
    return out


JPEG_QUALITY = 85


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()
    path = lambda f: os.path.join(args.models, f)
    have = lambda f: os.path.exists(path(f))

    for name in sorted(os.listdir(FACES)):
        if not name.endswith(".png"):
            continue
        img = cv2.imread(os.path.join(FACES, name), cv2.IMREAD_COLOR)
        ref = {"picture": name[:-4], "detections": {}, "embeddings": {}}
        if have("face_detection_yunet_2023mar.onnx"):
            faces = yunet(path("face_detection_yunet_2023mar.onnx"), img, 640)
            ref["detections"]["yunet_2023mar"] = {"input_size": 640, "faces": faces}
            if have("face_recognition_sface_2021dec.onnx"):
                ref["embeddings"]["sface_2021dec"] = embed_records(sface, path("face_recognition_sface_2021dec.onnx"), img, faces)
        for det in ("det_500m", "det_10g"):
            if not have(det + ".onnx"):
                continue
            for size in (640, 320):
                faces = scrfd(path(det + ".onnx"), img, size)
                ref["detections"][f"{det}@{size}"] = {"input_size": size, "faces": faces}
                if det == "det_500m" and size == 640:
                    for emb in ("w600k_mbf", "w600k_r50"):
                        if have(emb + ".onnx"):
                            ref["embeddings"][emb] = embed_records(arcface, path(emb + ".onnx"), img, faces)
        # The same picture as a JPEG, written next to the PNG, so that the
        # production decode path can be checked against OpenCV's on identical
        # bytes. Embeddings from it use the PNG landmarks.
        ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        assert ok
        with open(os.path.join(FACES, ref["picture"] + ".jpg"), "wb") as f:
            f.write(jpeg.tobytes())
        decoded = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        ref["jpeg_embeddings"] = {}
        if have("face_recognition_sface_2021dec.onnx") and "sface_2021dec" in ref["embeddings"]:
            faces = [{"landmarks": e["landmarks"]} for e in ref["embeddings"]["sface_2021dec"]]
            ref["jpeg_embeddings"]["sface_2021dec"] = embed_records(sface, path("face_recognition_sface_2021dec.onnx"), decoded, faces)
        if have("w600k_mbf.onnx") and "w600k_mbf" in ref["embeddings"]:
            faces = [{"landmarks": e["landmarks"]} for e in ref["embeddings"]["w600k_mbf"]]
            ref["jpeg_embeddings"]["w600k_mbf"] = embed_records(arcface, path("w600k_mbf.onnx"), decoded, faces)
        with open(os.path.join(args.out, ref["picture"] + ".json"), "w") as f:
            json.dump(ref, f, indent=1)
        print(name, {k: len(v["faces"]) for k, v in ref["detections"].items()})


if __name__ == "__main__":
    main()
