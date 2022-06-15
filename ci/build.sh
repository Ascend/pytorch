#!/bin/bash

# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CUR_DIR=$(dirname $(readlink -f $0))
SUPPORTED_PY_VERSION=(3.7 3.8 3.9)
PY_VERSION='3.7'                     # Default supported python version is 3.7
DEFAULT_SCRIPT_ARGS_NUM=1            # Default supported input parameters

# Parse arguments inside script
function parse_script_args() {
    local args_num=0
    if [[ "x${1}" = "x" ]]; then
        # default: bash build.sh (python3.7)
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
        --tocpu=*)
            export 'NPU_TOCPU'=${1:8}
            args_num=$((args_num-1))
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

    # Find matched dependent Python libraries to current Python version in HCCL compiling
    hccl_file=${CUR_DIR}/../third_party/acl/libs/build_stub.sh
    cmake_file=${CUR_DIR}/../CMakeLists.txt
    if [[ ${PY_VERSION} = '3.7' ]]; then
        dst_py_ver='3.7m'
    else
        dst_py_ver=${PY_VERSION}
    fi
    for src_py_ver in ${SUPPORTED_PY_VERSION[*]}; do
        if [[ ${src_py_ver} = '3.7' ]]; then
            src_py_ver='3.7m'
        fi
        if [[ $(grep -c "${src_py_ver}" ${hccl_file}) -ne 0 && ${src_py_ver} != ${dst_py_ver} ]]; then
            sed -i "s/libpython${src_py_ver}/libpython${dst_py_ver}/g" ${hccl_file}
        fi
        if [[ $(grep -c "${src_py_ver}" ${cmake_file}) -ne 0 && ${src_py_ver} != ${dst_py_ver} ]]; then
            sed -i "s/python${src_py_ver}/python${dst_py_ver}/g" ${cmake_file}
        fi
    done

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