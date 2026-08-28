#!/bin/bash

set -e

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

cd ${CDIR}

gcc -fPIC -shared -o libhccl.so -I./ hccl.cpp

ACL_INCLUDE_DIR="${1:-../inc}"

gcc -fPIC -shared -o libascendcl.so -I"${ACL_INCLUDE_DIR}" acl.cpp

gcc -fPIC -shared -o libacl_op_compiler.so -I"${ACL_INCLUDE_DIR}" acl_op_compiler.cpp

gcc -fPIC -shared -o libge_runner.so -I"${ACL_INCLUDE_DIR}" ge_runner.cpp ge_api.cpp

gcc -fPIC -shared -o libgraph.so -I"${ACL_INCLUDE_DIR}" graph.cpp operator_factory.cpp operator.cpp tensor.cpp

gcc -fPIC -shared -o libacl_tdt_channel.so -I"${ACL_INCLUDE_DIR}" acl_tdt.cpp

gcc -fPIC -shared -o libascend_ml.so -I"${ACL_INCLUDE_DIR}" aml_fwk_detect.cpp
