# PyTorch毕昇C++算子开发指南
-   [简介](#简介md)
-   [环境准备](#环境准备md)
-   [C++ Extension方式开发算子](#Extension方式开发算子md)
    -   [环境准备](#环境准备md)
    -   [示例：add算子实现](#add算子md)
    -   [示例：add算子编译](#add算子安装代码md)
    -   [示例：add算子测试](#add算子测试md)
-   [框架联合编译方式开发算子](#框架联合编译方式开发算子md)
-   [FAQ](#FAQmd)
    -   [示例add算子编译失败](#python-setup_py-install失败md)
    -   [示例add算子测试失败](#python-test_add_bisheng_py失败md)
<h2 id="简介md">简介</h2>
毕昇C++提供了面向昇腾芯片的通用编程模型抽象，让开发者可以使用标准C++语法编写适用于昇腾芯片的算子。

昇腾PyTorch支持C++ Extension和框架联合编译两种方式开发毕昇C++算子，该文档介绍了怎么在PyTorch中使用毕昇C++开发算子。
<h2 id="环境准备md">环境准备</h2>
在PyTorch框架中开发毕昇C++算子环境准备包括两部分内容：一是CANN和毕昇C++的安装，二是PyTorch框架安装：

- CANN和毕昇C++的安装：当前毕昇C++内置于CANN 6.3.T100版本，安装CANN时同时安装毕昇C++，使用CANN内置脚本配置CANN环境变量时会同时配置毕昇C++环境变量。
  - CANN 6.3.T100下载地址及毕昇C++文档请参考：[CANN 6.3.T100（毕昇C++）安装包及文档](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/259218207?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373)；
  - CANN安装请参考昇腾社区文档中环境安装环节[《CANN软件安装》](https://www.hiascend.com/document?tag=community-developer)
  - 安装后运行CANN安装目录下的`set_env.sh`自动配置CANN和毕昇C++相关环境变量。

安装CANN并完成环境变量配置后可以查看CANN和毕昇C++的环境变量：
```
echo $BISHENG_CPP_HOME
echo $ASCEND_HOME_PATH
```

- 参考README里编译安装原生PyTorch和昇腾PyTorch

<h2 id="Extension方式开发算子md">C++ Extension方式开发算子</h2>
<h3 id="环境准备md">环境准备</h3>

参考前文安装CANN、毕昇C++和PyTorch，并配置环境变量

<h3 id="add算子md">示例：add算子实现</h3>
add_bisheng.cpp

```
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

#include <sycl/sycl.hpp>
#include <acl/acl.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

using namespace at;
using namespace sycl;
using namespace sycl::access;
template <typename T>
using LocalAccessor = accessor<T, 1, mode::read_write, target::local>;

template <typename T, int group_count, typename kernel_name>
void add_bisheng_kernel(queue &sycl_queue, const T *in1, const T *in2, T *out, size_t length) {
  sycl_queue.submit([&](handler &cgh)
                    {
        LocalAccessor<T> local_in1(length, cgh);
        LocalAccessor<T> local_in2(length, cgh);
        LocalAccessor<T> local_out(length, cgh);
        cgh.parallel_for_work_group<kernel_name>(sycl::range<1> { group_count }, [=](group<1> grp) {
            const auto gid = grp.get_id(0);
            const auto idx_base = gid * length;
            [[loop::parallel]] for (size_t i = 0; i < length; ++i) {
                local_in1[i] = in1[i + idx_base];
                local_in2[i] = in2[i + idx_base];
                local_out[i] = local_in1[i] + local_in2[i];
                out[i + idx_base] = local_out[i];
            }
        }); });
  sycl_queue.wait();
}

template <class T, class kernel_name>
Tensor add_bisheng_launch(const Tensor &self, const Tensor &other) {
  Tensor result = at::empty(self.sizes(), self.options());
  const T *self_ptr = static_cast<T *>(self.storage().data_ptr().get());
  const T *other_ptr = static_cast<T *>(other.storage().data_ptr().get());
  T *result_ptr = static_cast<T *>(result.storage().data_ptr().get());

  aclrtContext acl_context;
  aclrtGetCurrentContext(&acl_context);
  sycl::context sycl_context = sycl::make_context<sycl::backend::cce>(acl_context);
  int device_id = 0;
  aclrtGetDevice(&device_id);
  auto npu_stream = c10_npu::getCurrentNPUStream(device_id);
  auto acl_stream = npu_stream.stream();
  queue sycl_queue = sycl::make_queue<sycl::backend::cce>(acl_stream, sycl_context);

  const T *input1 = self_ptr;
  const T *input2 = other_ptr;
  constexpr size_t group_count = 1;
  size_t len = result.numel() / group_count;
  add_bisheng_kernel<T, group_count, kernel_name>(sycl_queue, input1, input2, result_ptr, len);

  return result;
}

Tensor add_bisheng(const Tensor &self, const Tensor &other) {
  if (self.scalar_type() == at::kHalf) {
    return add_bisheng_launch<half, class addhalf>(self, other);
  } else if (self.scalar_type() == at::kFloat) {
    return add_bisheng_launch<float, class addfloat>(self, other);
  } else {
    return add_bisheng_launch<int64_t, class addint>(self, other);
  }
}

PYBIND11_MODULE(add_bisheng, m) {
  m.def("add", &add_bisheng, "x + y");
}

```

<h3 id="add算子安装代码md">示例：add算子编译</h3>
python setup.py install

```
# Copyright (c) 2023 Huawei Technologies Co., Ltd
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

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

from torch_npu.utils.cpp_extension import BiShengExtension


setup(
    name='add_bisheng',
    ext_modules=[
        BiShengExtension(
            name='add_bisheng',
            sources=['add_bisheng.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

```
<h3 id="add算子测试md">示例：add算子测试</h3>
python test_add_bisheng.py

```
# Copyright (c) 2023 Huawei Technologies Co., Ltd
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

import os
import time
import torch

import torch_npu


JIT_COMPILE_EXTENSION = True
if JIT_COMPILE_EXTENSION:
    from torch_npu.utils.cpp_extension import load
    module = load(name="add_bisheng", sources=["./add_bisheng.cpp"], verbose=1)
else:
    # python setup.py install
    import add_bisheng as module

torch.manual_seed(0)
npu_device = os.environ.get('SET_NPU_DEVICE')
if npu_device is None:
    torch.npu.set_device("npu:0")
else:
    torch.npu.set_device(f"npu:{npu_device}")

a = torch.rand(2, 10).npu()
b = torch.rand(2, 10).npu()
print("A:  " + str(a))
print("B:  " + str(b))
add_result = module.add(a, b)
print("add_result:  " + str(add_result))

print("\n")
a = torch.randint(0, 10, (2, 10)).npu()
b = torch.randint(0, 10, (2, 10)).npu()
print("A int:  " + str(a))
print("B int:  " + str(b))
add_result = module.add(a, b)
print("add_result int:  " + str(add_result))

print("\n")
a = torch.rand(2, 10).half()
b = torch.rand(2, 10).half()
a = a.npu()
b = b.npu()
print("A half:  " + str(a))
print("B half:  " + str(b))
add_result = module.add(a, b)
print("add_result half:  " + str(add_result))

```

<h2 id="框架联合编译方式开发算子md">框架联合编译方式开发算子</h2>

-  在torch_npu/csrc/aten/npu_native_functions.yaml增加接口定义, 参考：bscpp_add
```
  - func: bscpp_add(Tensor self, Tensor other) -> Tensor
    bscpp_op: True
```
-  在torch_npu/csrc/aten/bscpp_ops目录下增加算子实现CPP文件，参考：AddKernel.cpp

-  编译框架：
```
# BSCPP_SYCL_TARGET设置请参考毕昇C++文档
BSCPP_OPS_ENABLE=1 BSCPP_SYCL_TARGET=ascend_910-cce bash ci/build.sh
```

-  安装框架: 
```
pip install --upgrade --force-reinstall dist/*.whl
```

<h2 id="FAQmd">FAQ</h2>

-   **[示例add算子编译失败](#python-setup_py-install失败md)**  
-   **[示例add算子测试失败](#python-test_add_bisheng_py失败md)**  

<h3 id="python-setup_py-install失败md">示例add算子编译失败</h3>

#### 现象描述

/opt/BiShengCPP/bin/clang: error while loading shared libraries: libtinfo.so.6: cannot open shared object file: No such file or directory


#### 处理方法

复制libtinfo.so.6到 ${BISHENTCPP_ROOT}/lib

```
(***111) root@ubuntu:/home/***/add_bisheng# find / -name "libtinfo.so.6"
/root/miniconda3/pkgs/ncurses-6.3-h2f4d8fa_2/lib/libtinfo.so.6
/root/miniconda3/pkgs/ncurses-6.3-h998d150_3/lib/libtinfo.so.6
/root/miniconda3/envs/pt181/lib/libtinfo.so.6
/root/miniconda3/envs/pt111/lib/libtinfo.so.6
/root/miniconda3/envs/***111/lib/libtinfo.so.6
/root/miniconda3/lib/libtinfo.so.6
/home/syf/miniconda3/pkgs/ncurses-6.3-h2f4d8fa_2/lib/libtinfo.so.6
/home/syf/miniconda3/pkgs/ncurses-6.3-h998d150_3/lib/libtinfo.so.6
/home/syf/miniconda3/lib/libtinfo.so.6
(***111) root@ubuntu:/home/***/add_bisheng# cp /root/miniconda3/lib/libtinfo.so.6 ${BISHENTCPP_ROOT}/lib

```

<h3 id="python-test_add_bisheng_py失败md">示例add算子测试失败</h3>

#### 现象描述

```
ERROR: setUpClass (__main__.test_add_bisheng)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/root/miniconda3/envs/***111/lib/python3.7/site-packages/torch_npu/testing/testcase.py", line 71, in setUpClass
    self.npu_device = set_npu_device()
  File "/root/miniconda3/envs/***111/lib/python3.7/site-packages/torch_npu/testing/common_utils.py", line 83, in set_npu_device
    torch.npu.set_device(npu_device)
  File "/root/miniconda3/envs/***111/lib/python3.7/site-packages/torch_npu/npu/utils.py", line 149, in set_device
    torch_npu._C._npu_setDevice(torch.device(device).index)
RuntimeError: Initialize:/home/***/PyTorch/torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp:97 NPU error, error code is 507033
EL0003: The argument is invalid.
        Solution: Try again with a valid argument.
        TraceBack (most recent call last):
        Failed to open device, retCode=0x7020014, deviceId=0.[FUNC:Init][FILE:device.cc][LINE:227]
        Check param failed, dev can not be NULL![FUNC:DeviceRetain][FILE:runtime.cc][LINE:2133]
        Check param failed, dev can not be NULL![FUNC:PrimaryContextRetain][FILE:runtime.cc][LINE:1946]
        Check param failed, ctx can not be NULL![FUNC:PrimaryContextRetain][FILE:runtime.cc][LINE:1973]
        Check param failed, context can not be null.[FUNC:NewDevice][FILE:api_impl.cc][LINE:1315]
        new device failed, retCode=0x7010006[FUNC:SetDevice][FILE:api_impl.cc][LINE:1336]
        rtSetDevice execute failed, reason=[device retain error][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:49]
        open device 0 failed, runtime result = 507033.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]
        ctx is NULL![FUNC:GetDevErrMsg][FILE:api_impl.cc][LINE:3490]
        rtGetDevMsg execute failed, reason=[context pointer null][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:49]
```


#### 处理方法

export SET_NPU_DEVICE=1