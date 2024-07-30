#!/bin/bash

set -e

CUR_DIR=$(dirname $(readlink -f $0))
SUPPORTED_PY_VERSION=(3.8 3.9 3.10 3.11)
PY_VERSION='3.8'                     # Default supported python version is 3.8
DEFAULT_SCRIPT_ARGS_NUM=1            # Default supported input parameters
export DISABLE_RPC_FRAMEWORK=FALSE

# Parse arguments inside script
function parse_script_args() {
    local args_num=0
    if [[ "x${1}" = "x" ]]; then
        # default: bash build.sh (python3.8)
        return 0
    fi

    while true; do
        if [[ "x${1}" = "x" ]]; then
            break
        fi
        if [[ "$(echo "${1}"|cut -b1-|cut -b-2)" == "--" ]]; then
            args_num=$((args_num+1))
        fi
        if [[ ${args_num} -eq ${DEFAULT_SCRIPT_ARGS_NUM} ]]; then
            break
        fi
        shift
    done

    # if num of args are not fully parsed, throw an error.
    if [[ ${args_num} -lt ${DEFAULT_SCRIPT_ARGS_NUM} ]]; then
        return 1
    fi

    while true; do
        case "${1}" in
        --python=*)
            PY_VERSION=$(echo "${1}"|cut -d"=" -f2)
            args_num=$((args_num-1))
            shift
            ;;
        --disable_torchair)
            export DISABLE_INSTALL_TORCHAIR=TRUE
            shift
            ;;
        --disable_rpc)
            export DISABLE_RPC_FRAMEWORK=TRUE
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
    matched_py_version='false'
    for ver in ${SUPPORTED_PY_VERSION[*]}; do
        if [ "${PY_VERSION}" = "${ver}" ]; then
            matched_py_version='true'
            return 0
        fi
    done
    if [ "${matched_py_version}" = 'false' ]; then
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