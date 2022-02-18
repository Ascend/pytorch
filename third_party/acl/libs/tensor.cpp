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

#include "graph/tensor.h"

namespace ge {
Tensor::Tensor() {}

Tensor tensor;

Tensor::Tensor(const TensorDesc& tensorDesc) {}

Tensor::Tensor(const TensorDesc& tensorDesc, const uint8_t* data, size_t size) {}

graphStatus Tensor::SetData(
    uint8_t* data,
    size_t size,
    const Tensor::DeleteFunc& deleter_func) {
  return GRAPH_SUCCESS;
}

Shape::Shape() {}

std::vector<int64_t> Shape::GetDims() const {
  return std::vector<int64_t>{};
}

size_t Shape::GetDimNum() const {
  return 0;
}

TensorDesc::TensorDesc() {}
void TensorDesc::SetFormat(Format format) {}
void TensorDesc::SetOriginFormat(Format originFormat) {}
void TensorDesc::SetShape(const Shape& shape) {}
void TensorDesc::SetOriginShape(const Shape& originShape) {}
void TensorDesc::SetDataType(DataType dt) {}
void TensorDesc::SetPlacement(Placement placement) {}
Format TensorDesc::GetFormat() const {return FORMAT_NCHW;}
Format TensorDesc::GetOriginFormat() const {return FORMAT_NCHW;}
Shape TensorDesc::GetShape() const {return Shape();};
Shape TensorDesc::GetOriginShape() const {return Shape();}
DataType TensorDesc::GetDataType() const {return DT_INT64;}
} // namespace ge