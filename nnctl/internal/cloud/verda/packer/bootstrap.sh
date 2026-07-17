#!/bin/bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

BOOTSTRAP_TMP="$(/usr/bin/mktemp -d /tmp/zig-nn-bootstrap.XXXXXX)"
trap '/bin/rm -rf "${BOOTSTRAP_TMP}"' EXIT

/usr/bin/apt-get update
/usr/bin/apt-get install -y \
  ca-certificates \
  curl \
  xz-utils \
  rsync \
  build-essential \
  pkg-config \
  gnupg

# CUDA toolkit repo for Ubuntu 24.04. Idempotent if already installed.
if [ ! -f /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list ]; then
  CUDA_KEYRING="${BOOTSTRAP_TMP}/cuda-keyring_1.1-1_all.deb"
  CUDA_KEYRING_SHA256="d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba"
  /usr/bin/curl -fsSL \
    --proto '=https' \
    --tlsv1.2 \
    --max-filesize 1048576 \
    --retry 3 \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    -o "${CUDA_KEYRING}"
  printf '%s  %s\n' "${CUDA_KEYRING_SHA256}" "${CUDA_KEYRING}" | /usr/bin/sha256sum --check --strict -
  /usr/bin/dpkg -i "${CUDA_KEYRING}"
fi

/usr/bin/apt-get update

if [ ! -f /usr/local/cuda/include/cuda.h ]; then
  /usr/bin/apt-get install -y cuda-toolkit-13-0
fi

# Ensure /usr/local/cuda exists if alternatives did not create it.
if [ ! -e /usr/local/cuda ] && [ -d /usr/local/cuda-13.0 ]; then
  /bin/ln -s /usr/local/cuda-13.0 /usr/local/cuda
fi

# Zig 0.16.0.
ZIG_VERSION="0.16.0"
ZIG_DIR="/opt/zig/${ZIG_VERSION}"
ZIG_TARBALL="${BOOTSTRAP_TMP}/zig-x86_64-linux-${ZIG_VERSION}.tar.xz"
ZIG_TARBALL_SHA256="70e49664a74374b48b51e6f3fdfbf437f6395d42509050588bd49abe52ba3d00"

if [ ! -x "${ZIG_DIR}/zig" ]; then
  /usr/bin/install -d /opt/zig
  /usr/bin/curl -fsSL \
    --proto '=https' \
    --tlsv1.2 \
    --max-filesize 62914560 \
    --retry 3 \
    "https://ziglang.org/download/${ZIG_VERSION}/zig-x86_64-linux-${ZIG_VERSION}.tar.xz" \
    -o "${ZIG_TARBALL}"
  printf '%s  %s\n' "${ZIG_TARBALL_SHA256}" "${ZIG_TARBALL}" | /usr/bin/sha256sum --check --strict -
  /usr/bin/tar -C /opt/zig -xf "${ZIG_TARBALL}"
  /bin/rm -rf "${ZIG_DIR}"
  /bin/mv "/opt/zig/zig-x86_64-linux-${ZIG_VERSION}" "${ZIG_DIR}"
fi

/bin/ln -sf "${ZIG_DIR}/zig" /usr/local/bin/zig

# Runtime linker paths for CUDA/NVRTC.
cat >/etc/ld.so.conf.d/zig-nn-cuda.conf <<'EOF'
/usr/local/cuda/lib64
/usr/local/cuda/targets/x86_64-linux/lib
/lib/x86_64-linux-gnu
/usr/lib/x86_64-linux-gnu
EOF
/sbin/ldconfig

# Helpful for interactive shells, but scripts should still use full paths.
cat >/etc/profile.d/zig-nn.sh <<'EOF'
export PATH="/usr/local/bin:/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
EOF

# Keep GPU awake if nvidia-smi is present.
/usr/bin/nvidia-smi -pm 1 || true

/usr/local/bin/zig version
/usr/bin/test -f /usr/local/cuda/include/cuda.h
/usr/bin/find -L /usr/local/cuda -maxdepth 3 \( -name cuda.h -o -name 'libnvrtc.so*' -o -name 'libcuda.so*' \) -print
