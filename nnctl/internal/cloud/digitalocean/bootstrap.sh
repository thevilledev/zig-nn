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
  pkg-config

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

cat >/etc/profile.d/zig-nn.sh <<'EOF'
export PATH="/usr/local/bin:/usr/local/cuda/bin:/opt/rocm/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/opt/rocm/lib:/opt/rocm/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
EOF

/usr/local/bin/zig version
