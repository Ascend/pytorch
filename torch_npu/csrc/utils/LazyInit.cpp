// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include <mutex>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/Exceptions.h>
#include "torch_npu/csrc/utils/LazyInit.h"

namespace torch_npu {
namespace utils {


static bool npu_run_yet = false;

void npu_lazy_init() {
  AutoGIL g;
  // Protected by the GIL.  We don't use call_once because under ASAN it
  // has a buggy implementation that deadlocks if an instance throws an
  // exception.  In any case, call_once isn't necessary, because we
  // have taken a lock.
  if (!npu_run_yet) {
    auto module = THPObjectPtr(PyImport_ImportModule("torch_npu.npu"));
    if (!module) throw python_error();
    auto res = THPObjectPtr(PyObject_CallMethod(module.get(), "_lazy_init", ""));
    if (!res) throw python_error();
    npu_run_yet = true;
  }
}

void npu_set_run_yet_variable_to_false() {
  npu_run_yet = false;
}

}
}