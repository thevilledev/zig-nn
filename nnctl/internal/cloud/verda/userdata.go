package verdacloud

// DefaultUserDataScript is intentionally hardcoded so benchmark worker setup can
// live with the nnctl deploy policy. Replace this script as the agent bootstrap
// becomes concrete.
const DefaultUserDataScript = `#!/usr/bin/env bash
set -euo pipefail

mkdir -p /opt/nnctl-agent
cat >/opt/nnctl-agent/README <<'EOF'
This instance was created by nnctl for zig-nn GPU benchmark work.
Replace nnctl/internal/cloud/verda/userdata.go with the real agent bootstrap.
EOF
`
