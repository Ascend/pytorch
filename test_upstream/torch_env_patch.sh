#!/bin/bash
# torch_env_patch.sh - Apply patches to installed torch package in Python environment
#
# This script applies patches from test_upstream/torch/ directory to the
# torch package installed in the Python environment (e.g., site-packages/torch).
#
# Usage:
#   ./torch_env_patch.sh [--python=<version>]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
PYTHON="python3"
PATCH_DIR="$SCRIPT_DIR/torch"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --python=*)
            PYTHON="python${1#*=}"
            shift
            ;;
        --python)
            PYTHON="python$2"
            shift 2
            ;;
        -v|--verbose)
            # Accepted for backward compatibility, no special behavior
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Verify Python is available
if ! command -v "$PYTHON" &> /dev/null; then
    echo "ERROR: Python executable '$PYTHON' not found"
    exit 1
fi

echo "Using Python: $($PYTHON --version 2>&1)"

# Find torch package installation location
TORCH_PATH=$($PYTHON -c "import torch; print(torch.__path__[0])" 2>/dev/null || echo "")
if [ -z "$TORCH_PATH" ]; then
    echo "ERROR: torch package not found in Python environment"
    exit 1
fi

echo "Torch package location: $TORCH_PATH"

# Verify patch directory exists
if [ ! -d "$PATCH_DIR" ]; then
    echo "ERROR: Patch directory not found: $PATCH_DIR"
    exit 1
fi

echo "Patch directory: $PATCH_DIR"

# Find all patch files
PATCH_FILES=$(find "$PATCH_DIR" -type f \( -name "*.patch" -o -name "*.diff" \) | sort)
if [ -z "$PATCH_FILES" ]; then
    echo "No patch files found in $PATCH_DIR"
    exit 0
fi

PATCH_COUNT=$(echo "$PATCH_FILES" | wc -l)
echo "Found $PATCH_COUNT patch files"

# Change to site-packages (parent of torch package)
# Patch files use paths like torch/_inductor/graph.py, with -p1 this resolves correctly
TORCH_PARENT_DIR=$(dirname "$TORCH_PATH")
echo "Working directory: $TORCH_PARENT_DIR"
cd "$TORCH_PARENT_DIR"

# Apply patches (patch command natively handles both LF and CRLF line endings)
echo ""
echo "========================================"
echo "Applying torch environment patches..."
echo "========================================"

count=0
fail=0
for patch in $PATCH_FILES; do
    count=$((count+1))
    patch_rel=$(realpath --relative-to="$SCRIPT_DIR" "$patch" 2>/dev/null || basename "$patch")
    echo "[$count/$PATCH_COUNT] $patch_rel"

    if patch -p1 --no-backup-if-mismatch -f < "$patch" > /tmp/torch_patch_output.log 2>&1; then
        :
    else
        echo "  FAILED: $(cat /tmp/torch_patch_output.log)"
        fail=$((fail+1))
        exit 1
    fi
done

echo ""
echo "========================================"
echo "All $count patches applied successfully"
echo "========================================"
