#!/bin/bash

set -e

CUR_DIR=$(dirname $(readlink -f $0))
SUPPORTED_PY_VERSION=(3.9 3.10 3.11 3.12 3.13)
SUPPORTED_TORCH_VERSION=(2.10.0 2.11.0 2.12.0)
# Default supported python version is 3.9
PY_VERSION="3.9"
# Torch version to validate against installed PyTorch (empty = skip check)
# Also written to version.txt before building
TORCH_VERSION=""

# Parse arguments inside script
function parse_script_args() {
    local args_num=0
    if [[ "x${1}" = "x" ]]; then
        # default: bash build.sh (python3.9)
        return 0
    fi

    args_num=$#

    while true; do
        case "${1}" in
        --python=*)
            PY_VERSION=$(echo "${1}"|cut -d"=" -f2)
            args_num=$((args_num-1))
            shift
            ;;
        --torch=*)
            TORCH_VERSION=$(echo "${1}"|cut -d"=" -f2)
            args_num=$((args_num-1))
            shift
            ;;
        --disable_torchair)
            export DISABLE_INSTALL_TORCHAIR=TRUE
            args_num=$((args_num-1))
            shift
            ;;
        --disable_rpc)
            export DISABLE_RPC_FRAMEWORK=TRUE
            args_num=$((args_num-1))
            shift
            ;;
        --enable_lto)
            export ENABLE_LTO=TRUE
            args_num=$((args_num-1))
            shift
            ;;
        --enable_pgo=*)
            pgo_mode=$(echo "${1}"|cut -d"=" -f2)
            case $pgo_mode in
            1)
                export PGO_MODE=1
                args_num=$((args_num-1))
                ;;
            2)  
                export PGO_MODE=2
                args_num=$((args_num-1))
                ;;
            *)
                echo "ERROR Unsupported parameters: ${1}"
                return 1
                ;;
            esac
            shift
            ;;
        -*)
            echo "ERROR Unsupported parameters: ${1}"
            return 1
            ;;
        *)
            if [ "x${1}" != "x" ]; then
                echo "ERROR Unsupported parameters: ${1}"
                return 1
            fi
            break
            ;;
        esac
    done

    # if some "--param=value" are not parsed correctly, throw an error.
    if [[ ${args_num} -ne 0 ]]; then
        return 1
    fi
}

function check_python_version() {
    matched_py_version="false"
    for ver in ${SUPPORTED_PY_VERSION[*]}; do
        if [ "${PY_VERSION}" = "${ver}" ]; then
            matched_py_version="true"
            return 0
        fi
    done
    if [ "${matched_py_version}" = "false" ]; then
        echo "${PY_VERSION} is an unsupported python version, we suggest ${SUPPORTED_PY_VERSION[*]}"
        exit 1
    fi
}

function check_torch_version() {
    if [ -z "${TORCH_VERSION}" ]; then
        return 0
    fi
    local matched="false"
    for ver in ${SUPPORTED_TORCH_VERSION[*]}; do
        if [ "${TORCH_VERSION}" = "${ver}" ]; then
            matched="true"
            break
        fi
    done
    if [ "${matched}" = "false" ]; then
        echo "${TORCH_VERSION} is an unsupported torch version, we suggest ${SUPPORTED_TORCH_VERSION[*]}"
        exit 1
    fi
}

function check_torch_installed() {
    local installed
    installed=$(python"${PY_VERSION}" -c "import torch; print(torch.__version__)" 2>/dev/null)
    if [ -z "${installed}" ]; then
        echo "PyTorch is not installed for python${PY_VERSION}. Please install it before building."
        exit 1
    fi
    # Strip local tag (e.g. 2.11.0+cpu -> 2.11.0)
    local installed_base="${installed%%+*}"
    if [ -n "${TORCH_VERSION}" ]; then
        # Compare major.minor only (ignore patch and local tag)
        local requested_mm installed_mm
        requested_mm=$(echo "${TORCH_VERSION}" | cut -d. -f1,2)
        installed_mm=$(echo "${installed_base}" | cut -d. -f1,2)
        if [ "${installed_mm}" != "${requested_mm}" ]; then
            echo "PyTorch version mismatch: requested ${TORCH_VERSION}, but ${installed} is installed."
            exit 1
        fi
    fi
    echo "Using PyTorch ${installed}"
}

function main()
{
    if ! parse_script_args "$@"; then
        echo "Failed to parse script args. Please check your inputs."
        exit 1
    fi
    check_python_version
    check_torch_version
    check_torch_installed

    cd ${CUR_DIR}/..

    if [ -n "${TORCH_VERSION}" ]; then
        export TORCH_VERSION
        echo "${TORCH_VERSION}" > version.txt
        echo "Set package version to ${TORCH_VERSION}"
    fi
    # if you add or delete file/files in the project, you need to remove the following comment
    # make clean

    python"${PY_VERSION}" setup.py build bdist_wheel
    if [ $? != 0 ]; then
        echo "Failed to compile the wheel file. Please check the source code by yourself."
        exit 1
    fi

    exit 0
}

main "$@"
