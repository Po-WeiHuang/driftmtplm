#!/usr/bin/env bash
# Applies local patches (kept in third_party/mtp-lm-patch/) on top of the
# pinned mtp-lm submodule commit. Safe to re-run: skips a patch that is
# already applied.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${REPO}/mtp-lm"
PATCH_DIR="${REPO}/mtp-lm-patch"

cd "${TARGET}"
for patch in "${PATCH_DIR}"/*.patch; do
  [ -e "${patch}" ] || continue
  if git apply --reverse --check "${patch}" 2>/dev/null; then
    echo "Already applied: $(basename "${patch}")"
    continue
  fi
  echo "Applying: $(basename "${patch}")"
  git apply "${patch}"
done
