// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/framework/interface/GeHelper.h"

namespace at_npu
{
  namespace native
  {

    ge::TensorDesc GeHelper::Convert(const aclTensorDesc *desc)
    {
      ge::Shape shape(GetTensorDescDims(desc));
      auto format = Convert(aclGetTensorDescFormat(desc));
      auto dataType = Convert(aclGetTensorDescType(desc));
      ge::TensorDesc tensorDesc(shape, format, dataType);
      return tensorDesc;
    }

    std::vector<int64_t> GeHelper::GetTensorDescDims(const aclTensorDesc *desc)
    {
      auto size = aclGetTensorDescNumDims(desc);
      std::vector<int64_t> dims;
      dims.resize(size);
      for (int i = 0; i < size; i++)
      {
        dims[i] = aclGetTensorDescDim(desc, i);
      }
      return dims;
    }

    ge::DataType GeHelper::Convert(aclDataType dataType)
    {
      return (ge::DataType)dataType;
    }

    ge::Format GeHelper::Convert(aclFormat format)
    {
      return (ge::Format)format;
    }

  } // namespace native
} // namespace at_npu
