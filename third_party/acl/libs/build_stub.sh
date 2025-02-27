#!/bin/bash

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

cd ${CDIR}

gcc -fPIC -shared -o libhccl.so -I./ hccl.cpp

gcc -fPIC -shared -o libascendcl.so -I../inc acl.cpp

gcc -fPIC -shared -o libacl_op_compiler.so -I../inc acl_op_compiler.cpp

gcc -fPIC -shared -o libge_runner.so -I../inc ge_runner.cpp ge_api.cpp

gcc -fPIC -shared -o libgraph.so -I../inc graph.cpp operator_factory.cpp operator.cpp tensor.cpp

gcc -fPIC -shared -o libacl_tdt_channel.so -I../inc acl_tdt.cpp

gcc -fPIC -shared -o libascend_ml.so -I../inc aml_fwk_detect.cpp

