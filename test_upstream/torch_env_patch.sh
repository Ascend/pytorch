#!/bin/bash
# torch_env_patch.sh - Apply patches to installed torch package in Python environment
#
# This script applies patches from test_upstream/torch/ directory to the
# torch package installed in the Python environment (e.g., site-packages/torch).
#
# Usage:
#   ./torch_env_patch.sh [--python=<version>] [--patch-dir=<path>] [--dry-run]
#
# Options:
#   --python=<version>   Python version to use (e.g., 3.11). Default: auto-detect
#   --patch-dir=<path>   Directory containing torch patches. Default: test_upstream/torch
#   --dry-run            Only check what patches would be applied, don't actually apply
#   -v, --verbose        Show verbose output
#
# Environment variables:
#   PYTHON_VERSION       Python version (alternative to --python flag)
#   TORCH_PATCH_DIR      Patch directory (alternative to --patch-dir flag)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# Default values
PYTHON_VERSION=""
PATCH_DIR=""
DRY_RUN=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --python=*)
            PYTHON_VERSION="${1#*=}"
            shift
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --patch-dir=*)
            PATCH_DIR="${1#*=}"
            shift
            ;;
        --patch-dir)
            PATCH_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $SCRIPT_NAME [options]"
            echo ""
            echo "Apply patches from test_upstream/torch/ to installed torch package."
            echo ""
            echo "Options:"
            echo "  --python=<version>   Python version (e.g., 3.11)"
            echo "  --patch-dir=<path>   Patch directory (default: test_upstream/torch)"
            echo "  --dry-run            Check only, don't apply patches"
            echo "  -v, --verbose        Show verbose output"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Apply environment variables if not set via arguments
PYTHON_VERSION="${PYTHON_VERSION:-${PYTHON_VERSION:-}}"
PATCH_DIR="${PATCH_DIR:-${TORCH_PATCH_DIR:-$SCRIPT_DIR/test_upstream/torch}}"

# Resolve Python executable
if [ -n "$PYTHON_VERSION" ]; then
    PYTHON="python${PYTHON_VERSION}"
    PIP="pip${PYTHON_VERSION}"
else
    # Auto-detect Python version
    PYTHON="python3"
    PIP="pip3"
fi

# Verify Python is available
if ! command -v "$PYTHON" &> /dev/null; then
    echo "ERROR: Python executable '$PYTHON' not found"
    exit 1
fi

PYTHON_VER_FULL=$($PYTHON --version 2>&1)
echo "Using Python: $PYTHON_VER_FULL"

# Find torch package installation location
TORCH_PATH=$($PYTHON -c "import torch; print(torch.__path__[0])" 2>/dev/null || echo "")

if [ -z "$TORCH_PATH" ]; then
    echo "ERROR: torch package not found in Python environment"
    echo "Please install torch first: $PIP install torch"
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

# Statistics
SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

# Apply patches
echo ""
echo "========================================"
echo "Applying torch environment patches..."
echo "========================================"

# Change to torch package directory (patch files use paths like torch/testing/_internal/...)
cd "$TORCH_PATH"

for patch_file in $PATCH_FILES; do
    # Get relative patch name for display
    patch_rel=$(realpath --relative-to="$SCRIPT_DIR" "$patch_file" 2>/dev/null || basename "$patch_file")

    if $VERBOSE; then
        echo ""
        echo "Processing: $patch_rel"
    fi

    if $DRY_RUN; then
        # Dry run: check if patch can be applied
        if patch -p1 --dry-run --no-backup-if-mismatch -f < "$patch_file" > /dev/null 2>&1; then
            echo "[OK] $patch_rel (dry-run: can apply)"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            # Check if already applied
            if patch -p1 --dry-run --reverse --no-backup-if-mismatch -f < "$patch_file" > /dev/null 2>&1; then
                echo "[SKIP] $patch_rel (already applied)"
                SKIP_COUNT=$((SKIP_COUNT + 1))
            else
                echo "[FAIL] $patch_rel (dry-run: cannot apply)"
                FAIL_COUNT=$((FAIL_COUNT + 1))
            fi
        fi
    else
        # Actually apply the patch
        # Use --no-backup-if-mismatch to avoid creating .orig files
        # Use -f to force apply without prompts

        # First check if already applied (reverse test)
        if patch -p1 --dry-run --reverse --no-backup-if-mismatch -f < "$patch_file" > /dev/null 2>&1; then
            echo "[SKIP] $patch_rel (already applied)"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi

        # Try to apply
        if patch -p1 --no-backup-if-mismatch -f < "$patch_file" > /tmp/torch_patch_output.log 2>&1; then
            echo "[OK] $patch_rel"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

            if $VERBOSE; then
                cat /tmp/torch_patch_output.log
            fi
        else
            echo "[FAIL] $patch_rel"
            FAIL_COUNT=$((FAIL_COUNT + 1))

            if $VERBOSE; then
                echo "--- Patch output ---"
                cat /tmp/torch_patch_output.log
                echo "--- End output ---"
            fi

            # Continue with other patches instead of failing immediately
            # This allows partial application which may be useful for debugging
        fi
    fi
done

# Summary
echo ""
echo "========================================"
echo "Patch Application Summary"
echo "========================================"
echo "Total patches:    $PATCH_COUNT"
echo "Successfully:     $SUCCESS_COUNT"
echo "Skipped (applied): $SKIP_COUNT"
echo "Failed:           $FAIL_COUNT"
echo ""

if $DRY_RUN; then
    echo "(Dry run mode - no patches were actually applied)"
else
    if [ $FAIL_COUNT -gt 0 ]; then
        echo "WARNING: Some patches failed to apply"
        echo "This may indicate version mismatch or already modified files"
        exit 1
    else
        echo "All patches applied successfully!"
    fi
fi

exit 0