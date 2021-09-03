#!/bin/bash

gcc -fPIC -shared -o libtsdclient.so -I./ tdtclient.cpp

gcc -fPIC -shared -o libhccl.so -I./ hccl.cpp

gcc -fPIC -shared -o libpython3.7m.so -I./ python.cpp

gcc -fPIC -shared -o libascendcl.so -I../inc acl.cpp

gcc -fPIC -shared -o libacl_op_compiler.so -I../inc acl_op_compiler.cpp