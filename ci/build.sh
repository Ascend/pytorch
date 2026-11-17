#!/bin/bash

set -e

CUR_DIR=$(dirname $(readlink -f $0))
SUPPORTED_PY_VERSION=(3.10 3.11 3.12 3.13 3.14 3.14t)
# Default supported python version is 3.10
PY_VERSION="3.10"
# Torch version to validate against installed PyTorch (empty = skip check)
# When set, exported as TORCH_VERSION so setup.py builds the matching package
# version (e.g. 2.13.0.post1). When unset, setup.py uses the installed
# PyTorch version. The requested version is validated against version.txt.
TORCH_VERSION=""
# Build method: legacy setuptools+cmake by default (transition phase);
# --scikit-build-core opts into the new PEP 517 build path.
USE_SCIKIT_BUILD_CORE="FALSE"

# Parse arguments inside script
function parse_script_args() {
    local args_num=0
    if [[ "x${1}" = "x" ]]; then
        # default: bash build.sh (python3.10)
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
        --scikit-build-core)
            USE_SCIKIT_BUILD_CORE="TRUE"
            args_num=$((args_num-1))
            shift
            ;;
        --disable_torchair)
            export DISABLE_INSTALL_TORCHAIR=TRUE
            args_num=$((args_num-1))
            shift
            ;;
        --disable_fxrt)
            export DISABLE_INSTALL_FXRT=TRUE
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
    # Simple match against the versions listed in version.txt. The requested
    # version need not include the .postN suffix: a listed version such as
    # 2.13.0.post1 matches a request for its base 2.13.0.
    local version_file="${CUR_DIR}/../version.txt"
    if grep -v '^#' "${version_file}" | grep -Fq -- "${TORCH_VERSION}"; then
        return 0
    fi
    echo "${TORCH_VERSION} is an unsupported torch version, supported versions in version.txt:"
    grep -v '^#' "${version_file}" | sed '/^[[:space:]]*$/d'
    exit 1
}

function check_torch_installed() {
    local installed
    # Disable torch_npu autoload so an installed torch_npu doesn't sideload the
    # in-tree package and confuse the version probe.
    if ! installed=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
            python"${PY_VERSION}" -c "import torch; print(torch.__version__)" 2>/dev/null) \
        || [ -z "${installed}" ]; then
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
        echo "Building torch_npu for PyTorch ${TORCH_VERSION}"
    fi
    # if you add or delete file/files in the project, you need to remove the following comment
    # make clean

    if [ "${USE_SCIKIT_BUILD_CORE}" = "TRUE" ]; then
        # New build: scikit-build-core (PEP 517) via pip's built-in frontend.
        echo "Building torch_npu with scikit-build-core (PEP 517)..."
        # Build into a fresh temp dir so stale wheels in dist/ (or a parallel
        # build) can never be mistaken for this build's output.
        wheel_tmpdir=$(mktemp -d)
        python"${PY_VERSION}" -m pip wheel -v . --no-build-isolation --no-deps -w "${wheel_tmpdir}"
        if [ $? != 0 ]; then
            echo "Failed to compile the wheel file. Please check the source code by yourself."
            rm -rf "${wheel_tmpdir}"
            exit 1
        fi

        # pip wheel must have produced exactly one wheel.
        wheel_count=$(ls "${wheel_tmpdir}"/*.whl 2>/dev/null | wc -l)
        if [ "${wheel_count}" -ne 1 ]; then
            echo "Expected exactly one wheel, found ${wheel_count}."
            rm -rf "${wheel_tmpdir}"
            exit 1
        fi
        wheel_file=$(ls "${wheel_tmpdir}"/*.whl 2>/dev/null)

        # Manylinux repair: when AUDITWHEEL_PLAT names a manylinux tag, relabel
        # the wheel and bundle external libs with auditwheel, matching the
        # legacy BdistWheelBuild behavior. patchelf/eu-strip are not needed
        # here: the new build no longer links libgomp and Release has no debug
        # sections.
        if [[ "${AUDITWHEEL_PLAT}" == manylinux* ]]; then
            # External libs that must not be bundled (provided by PyTorch/CANN
            # at runtime); same list as legacy setup.py.
            deps=(libc10.so libtorch.so libtorch_cpu.so libtorch_python.so
                  libhccl.so libascendcl.so libacl_op_compiler.so libge_runner.so
                  libgraph.so libacl_tdt_channel.so libfmk_parser.so libascend_protobuf.so
                  libascend_ml.so libtorch_npu.so libnpu_profiler.so libgomp.so.1 libatb.so)
            exclude_args=()
            for dep in "${deps[@]}"; do
                exclude_args+=("--exclude" "${dep}")
            done
            auditwheel -v repair --plat "${AUDITWHEEL_PLAT}" -w dist "${wheel_file}" "${exclude_args[@]}"
            if [ $? != 0 ]; then
                echo "Failed to run auditwheel repair."
                rm -rf "${wheel_tmpdir}"
                exit 1
            fi
        else
            cp "${wheel_file}" dist/
        fi
        rm -rf "${wheel_tmpdir}"
    else
        # Legacy build: setuptools + cmake (default during the transition
        # phase). auditwheel/patchelf/eu-strip are handled inside setup.py's
        # BdistWheelBuild.
        echo "Building torch_npu with legacy setuptools + cmake..."
        python"${PY_VERSION}" setup.py build bdist_wheel
        if [ $? != 0 ]; then
            echo "Failed to compile the wheel file. Please check the source code by yourself."
            exit 1
        fi
    fi

    exit 0
}

main "$@"
