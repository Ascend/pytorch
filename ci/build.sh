#!/bin/bash

set -e

CUR_DIR=$(dirname $(readlink -f $0))
SUPPORTED_PY_VERSION=(3.9 3.10 3.11)
# Default supported python version is 3.9
PY_VERSION="3.9"

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

function main()
{
    if ! parse_script_args "$@"; then
        echo "Failed to parse script args. Please check your inputs."
        exit 1
    fi
    check_python_version

    cd ${CUR_DIR}/..
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
