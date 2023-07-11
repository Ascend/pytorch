#!/bin/bash

set -e
set -o pipefail

SCRIPT_BASE=$(dirname $(readlink -f $0))
TORCHAIR_BASE=${SCRIPT_BASE}/torchair

cd ${TORCHAIR_BASE} && chmod +x build_and_install.sh && ./build_and_install.sh $1
