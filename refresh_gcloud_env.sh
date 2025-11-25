#!/usr/bin/env bash
# Convenience helper so a new shell has gcloud + gsutil without re-running bootstrap.

set -e

if [ -f "/workspace/code/google-cloud-sdk/path.bash.inc" ]; then
  # shellcheck source=/dev/null
  source /workspace/code/google-cloud-sdk/path.bash.inc
  echo "gcloud version: $(gcloud --version | head -n 1 || echo 'unknown')"
  echo "gsutil version: $(gsutil --version 2>/dev/null || echo 'unknown')"
else
  echo "google-cloud-sdk not found under /workspace/code."
  echo "Run ./basi_pod_bootstrap_v2.sh first."
fi

