# Benchmark image for linux/amd64. Contains InsightFace weights (non-commercial
# research licence): for internal measurement only, do not publish this image.
FROM golang:1.26-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
COPY vendor ./vendor
COPY *.go ./
RUN CGO_ENABLED=1 go build -mod=vendor -trimpath -o /out/facebench .

FROM debian:bookworm-slim
WORKDIR /app
COPY lib/onnxruntime-linux-x64-1.29.0/lib/libonnxruntime.so.1.29.0 /app/lib/libonnxruntime.so
COPY models/det_500m.onnx models/w600k_mbf.onnx models/w600k_r50.onnx models/yunet_2023mar.onnx models/sface_2021dec.onnx /app/models/
COPY --from=build /out/facebench /app/facebench
COPY run.sh /app/run.sh
ENTRYPOINT ["/app/run.sh"]
