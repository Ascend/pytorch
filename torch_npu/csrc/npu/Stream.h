// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THNP_STREAM_INC
#define THNP_STREAM_INC

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <torch/csrc/Stream.h>
#include <torch/csrc/python_headers.h>

struct THNPStream : THPStream {
  c10_npu::NPUStream npu_stream;
};
extern PyObject *THNPStreamClass;

void THNPStream_init(PyObject *module);

inline bool THNPStream_Check(PyObject* obj) {
  return THNPStreamClass && PyObject_IsInstance(obj, THNPStreamClass);
}

#endif // THNP_STREAM_INC