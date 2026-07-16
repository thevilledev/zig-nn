# Containers

The root [`Dockerfile`](../Dockerfile) packages the ReleaseFast benchmark as a
finite container job. It uses separate build and runtime stages, keeps the Zig
compiler and source out of the result, runs as an unprivileged numeric user,
and supports both `linux/amd64` and `linux/arm64`.

The default `cpu` target is a statically linked musl binary in a `scratch`
image. Build and run its smoke-sized benchmark with:

```bash
docker build --target cpu --tag zig-nn-benchmark:cpu .
docker run --rm zig-nn-benchmark:cpu
```

Arguments after the image override the default `--quick` argument while keeping
the benchmark entrypoint:

```bash
docker run --rm zig-nn-benchmark:cpu --filter tiny_gpt
```

## CUDA

The `cuda` target compiles against the pinned CUDA development image. Its final
stage uses Chainguard's distroless `glibc-dynamic` runtime and adds only the
benchmark plus its NVRTC, cuBLAS, and cuBLASLt shared libraries. This keeps
scanner-visible package metadata for glibc, libgcc, and libstdc++ without
retaining Ubuntu. The compiler, shell, package manager, CUDA headers, Nsight
tools, and unrelated math libraries do not enter the final image. The CUDA
executable targets the baseline CPU feature set so it does not inherit AVX-512
or other optional instructions from the native image builder.

Build an amd64 image on a native amd64 builder for a Verda GPU worker, then
test it on a host with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
docker build \
  --platform linux/amd64 \
  --target cuda \
  --tag zig-nn-benchmark:cuda .

docker run --rm --gpus all zig-nn-benchmark:cuda
docker run --rm --gpus all zig-nn-benchmark:cuda --filter gpu_peak
```

The CUDA image intentionally relies on the host runtime to inject
`libcuda.so.1`; it will not start without an NVIDIA driver and container
runtime. Because the image has no shell or package manager, inspect it through
its build metadata or export its filesystem instead of trying to use
`docker exec`. Cross-building the amd64 CUDA image through Rosetta on Apple
Silicon can fail in Zig's `translate-c` helper, so use a native amd64 CI runner
or builder for the Verda artifact.

The CUDA version defaults to `13.0.1` to match the current benchmark worker
toolchain. Override both CUDA stages together when the deployment environment
moves to another available NVIDIA image version:

```bash
docker build \
  --platform linux/amd64 \
  --target cuda \
  --build-arg CUDA_VERSION=13.0.1 \
  --tag registry.example/zig-nn-benchmark:cuda-13.0.1 .
```

## GitHub Container Registry

The `Container` GitHub Actions workflow runs only when a Git tag is pushed. It
builds native `linux/amd64` images and publishes them to
`ghcr.io/thevilledev/zig-nn`. The Git tag is preserved in each image tag, with
a suffix distinguishing the two runtime targets:

- Git tag `v0.2.0` publishes `v0.2.0-cpu`
- Git tag `v0.2.0` publishes `v0.2.0-cuda`

Each published digest includes BuildKit provenance and an SPDX SBOM. The
workflow adds a keyless Cosign image signature, signs SLSA provenance and the
SPDX SBOM with GitHub's Sigstore-backed artifact attestations, then pushes the
signature and both attestations to GHCR. Verify a published image and its
keyless signature with:

```bash
TAG=v0.2.0
cosign verify \
  --certificate-identity \
    "https://github.com/thevilledev/zig-nn/.github/workflows/container.yml@refs/tags/${TAG}" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  "ghcr.io/thevilledev/zig-nn:${TAG}-cuda"
```

Verify its provenance and SBOM attestations against this repository with:

```bash
TAG=v0.2.0
gh attestation verify \
  "oci://ghcr.io/thevilledev/zig-nn:${TAG}-cuda" \
  --repo thevilledev/zig-nn
```

Tags are convenient selectors; use the verified `sha256` digest returned by
the registry when a Verda job must be reproducible.

## Verda Serverless Jobs

Push the selected image to an OCI-compatible registry. On a command-driven
finite job runner, its default command runs the quick benchmark; the full
executable command is:

```text
/usr/local/bin/zig-nn-benchmark --quick
```

For the CPU image the executable path is `/zig-nn-benchmark`. The process emits
CSV to its logs and exits with the benchmark status. Use an immutable image tag
or digest for repeatable benchmark results.

Verda currently also documents an HTTP-oriented
[Batch Jobs deployment](https://docs.verda.com/containers/batch-jobs/) with a
different contract: the application must expose a health endpoint, accept an
asynchronous request, return its result, and then terminate. The current images
are one-shot workers and are not that HTTP service. An inference batch adapter
should be added as a separate container target instead of wrapping the
benchmark image.
