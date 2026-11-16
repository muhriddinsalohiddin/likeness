# An image with the server, the runtime library and the default models in
# it: built where there is internet access, run where there is none.
#
#   docker build -t likeness .
#   docker run -p 8080:8080 -v likeness-data:/data \
#     -e LIKENESS_ADMIN_KEYS=... -e LIKENESS_SEARCH_KEYS=... likeness

FROM golang:1.26-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=1 go build -trimpath -ldflags "-s -w" -o /out/likeness ./cmd/likeness

# The default preset only: its models may be used by anyone. Fetching here
# also loads them once, so an image that builds is an image that works.
RUN /out/likeness setup --preset default --home /opt/likeness

FROM debian:bookworm-slim
RUN useradd --system --uid 10001 --home-dir /data --create-home likeness
COPY --from=build /out/likeness /usr/local/bin/likeness
COPY --from=build /opt/likeness /opt/likeness
ENV LIKENESS_HOME=/opt/likeness
USER likeness
VOLUME /data
EXPOSE 8080
ENTRYPOINT ["likeness"]
CMD ["serve", "--store", "/data/faces.db", "--listen", "0.0.0.0:8080"]
