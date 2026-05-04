#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

mkdir -p "$REPO_ROOT/containers"

singularity build --force --ignore-fakeroot-command "$REPO_ROOT/containers/contradiction_pipeline.sif" \
    "$SCRIPT_DIR/contradiction_pipeline.def"

# Optional rebuild if your cluster supports fakeroot locally:
# singularity build --fakeroot --force "$REPO_ROOT/containers/contradiction_pipeline.sif" \
#     "$SCRIPT_DIR/contradiction_pipeline.def"
