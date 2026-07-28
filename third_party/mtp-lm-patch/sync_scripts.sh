#!/usr/bin/env bash
# Copies local standalone scripts (kept in third_party/mtp-lm-patch/) into the
# mtp-lm submodule root, so they survive a fresh submodule checkout elsewhere.
# Safe to re-run: skips files already in sync.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${REPO}/mtp-lm"

rsync -a "${REPO}/mtp-lm-patch/dwload_traindst_cmd_qwen.sh" "${TARGET}/"
rsync -a "${REPO}/mtp-lm-patch/dwload_traindst_cmd_llama.sh" "${TARGET}/"