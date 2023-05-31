# Copyright (c) 2020 Huawei Technologies Co., Ltd
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
#!/bin/bash

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

cd ${CDIR}

gcc -fPIC -shared -o libhccl.so -I./ hccl.cpp

gcc -fPIC -shared -o libpython3.7m.so -I./ python.cpp

gcc -fPIC -shared -o libascendcl.so -I../inc acl.cpp

gcc -fPIC -shared -o libacl_op_compiler.so -I../inc acl_op_compiler.cpp

gcc -fPIC -shared -o libge_runner.so -I../inc ge_runner.cpp ge_api.cpp

gcc -fPIC -shared -o libgraph.so -I../inc graph.cpp operator_factory.cpp operator.cpp tensor.cpp

gcc -fPIC -shared -o libacl_tdt_channel.so -I../inc acl_tdt.cpp

