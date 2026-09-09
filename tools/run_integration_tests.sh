#!/usr/bin/env bash

set -euo pipefail

uv sync --all-packages --locked --extra uk
uv run --no-sync pytest \
  packages/microcosm-build/tests/test_uk_staging_integration.py \
  -q -s -p no:cacheprovider

if [[ -z "${HF_STAGING_READ_TOKEN:-}" ]]; then
  echo \
    "HF_STAGING_READ_TOKEN is unavailable; skipping the optional private repository access check."
  exit 0
fi

HF_TOKEN="$HF_STAGING_READ_TOKEN" \
  uv run --no-sync python tools/provision_uk_staging_repository.py --verify-access
