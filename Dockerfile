# syntax=docker/dockerfile:1.7

ARG ZIG_VERSION=0.16.0
ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=13.0.1
ARG GLIBC_RUNTIME_IMAGE=cgr.dev/chainguard/glibc-dynamic:latest

FROM ubuntu:${UBUNTU_VERSION} AS toolchain
ARG TARGETARCH
ARG ZIG_VERSION
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates curl xz-utils \
    && rm -rf /var/lib/apt/lists/*
RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64) zig_arch="x86_64"; zig_sha256="70e49664a74374b48b51e6f3fdfbf437f6395d42509050588bd49abe52ba3d00" ;; \
        arm64) zig_arch="aarch64"; zig_sha256="ea4b09bfb22ec6f6c6ceac57ab63efb6b46e17ab08d21f69f3a48b38e1534f17" ;; \
        *) echo "unsupported container architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    archive="zig-${zig_arch}-linux-${ZIG_VERSION}.tar.xz"; \
    curl --fail --location --show-error --silent \
        "https://ziglang.org/download/${ZIG_VERSION}/${archive}" \
        --output "/tmp/${archive}"; \
    printf '%s  %s\n' "${zig_sha256}" "/tmp/${archive}" | sha256sum --check --strict; \
    mkdir --parents /opt/zig; \
    tar --extract --xz --file "/tmp/${archive}" --directory /opt/zig --strip-components=1; \
    rm "/tmp/${archive}"
ENV PATH="/opt/zig:${PATH}"

FROM toolchain AS cpu-builder
ARG TARGETARCH
WORKDIR /src
COPY build.zig build.zig.zon ./
COPY benchmarks/ benchmarks/
COPY experiments/ experiments/
COPY src/ src/
RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64) zig_target="x86_64" ;; \
        arm64) zig_target="aarch64" ;; \
        *) echo "unsupported container architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    zig build benchmark-exe \
        -Dgpu=none \
        -Dtarget="${zig_target}-linux-musl" \
        --prefix /opt/zig-nn

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS cuda-builder
COPY --from=toolchain /opt/zig /opt/zig
ENV PATH="/opt/zig:${PATH}"
WORKDIR /src
COPY build.zig build.zig.zon ./
COPY benchmarks/ benchmarks/
COPY experiments/ experiments/
COPY src/ src/
RUN zig build benchmark-exe -Dgpu=cuda --prefix /opt/zig-nn \
    && mkdir --parents /opt/zig-nn/cuda-libs \
    && cp --archive \
        /usr/local/cuda/lib64/libcublas.so.* \
        /usr/local/cuda/lib64/libcublasLt.so.* \
        /usr/local/cuda/lib64/libnvrtc.so.* \
        /usr/local/cuda/lib64/libnvrtc-builtins.so.* \
        /opt/zig-nn/cuda-libs/

FROM ${GLIBC_RUNTIME_IMAGE} AS cuda
LABEL org.opencontainers.image.title="zig-nn CUDA benchmark" \
      org.opencontainers.image.description="ReleaseFast zig-nn benchmark runner with CUDA support" \
      org.opencontainers.image.source="https://github.com/thevilledev/zig-nn" \
      org.opencontainers.image.licenses="MIT"
ENV NVIDIA_VISIBLE_DEVICES="all" \
    NVIDIA_DRIVER_CAPABILITIES="compute,utility" \
    NVIDIA_REQUIRE_CUDA="cuda>=13.0" \
    LD_LIBRARY_PATH="/usr/local/lib/zig-nn:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
COPY --from=cuda-builder --chown=65532:65532 \
    /opt/zig-nn/bin/zig_nn_benchmark /usr/local/bin/zig-nn-benchmark
COPY --from=cuda-builder /opt/zig-nn/cuda-libs/ /usr/local/lib/zig-nn/
COPY --from=cuda-builder /NGC-DL-CONTAINER-LICENSE /licenses/NVIDIA-CUDA-LICENSE
COPY --chown=65532:65532 LICENSE /licenses/LICENSE
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/zig-nn-benchmark"]
CMD ["--quick"]

FROM scratch AS cpu
LABEL org.opencontainers.image.title="zig-nn CPU benchmark" \
      org.opencontainers.image.description="Statically linked ReleaseFast zig-nn benchmark runner" \
      org.opencontainers.image.source="https://github.com/thevilledev/zig-nn" \
      org.opencontainers.image.licenses="MIT"
COPY --from=cpu-builder --chown=65532:65532 \
    /opt/zig-nn/bin/zig_nn_benchmark /zig-nn-benchmark
COPY --chown=65532:65532 LICENSE /licenses/LICENSE
USER 65532:65532
ENTRYPOINT ["/zig-nn-benchmark"]
CMD ["--quick"]
