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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

tuple<int64_t, int64_t, int64_t> get_definedTensor_num(const TensorList& indices) {
  int64_t definedMaxNum = 1;
  int64_t definedNum = 0;
  int64_t lastdefinedTensor = 0;
  int64_t firstdefinedTensor = 0;

  for (int64_t i = 0; i < indices.size(); i++) {
    if (indices[i].defined()) {
      definedNum = definedNum + 1;
      if (definedNum == 1) {
        firstdefinedTensor = i;
      }
      if (indices[i].scalar_type() == at::kBool) {
        int64_t boolNum = indices[i].sum().item().toLong();
        if (boolNum > definedMaxNum) {
          definedMaxNum = boolNum;
        }
      } else {
        if (indices[i].numel() > definedMaxNum) {
          definedMaxNum = indices[i].numel();
        }
      }
      lastdefinedTensor = i;
    }
  }
  return std::tuple<int64_t, int64_t, int64_t>(definedMaxNum, lastdefinedTensor, firstdefinedTensor);
}

Tensor convert_bools_to_tensor(const Tensor& indice) {
  Tensor indice_contiguous = indice.to(Device(at::kCPU), at::kBool).contiguous();
  SmallVector<int64_t, N> boolTensorlist;
  bool* indicePtr = indice_contiguous.data_ptr<bool>();
  for (int64_t i = 0; i < indice.size(0); i++) {
    if (indicePtr[i]) {
      boolTensorlist.emplace_back(i);
    }
  }

  Tensor indiceT = at::tensor(IntArrayRef(boolTensorlist), indice.options().dtype(at::kLong));
  return indiceT;
}

tuple<SmallVector<Tensor, N>, SmallVector<int64_t, N>> get_indices_list(const Tensor& self,
                                                                        const TensorList& indices,
                                                                        int64_t definedMaxNum) {
  Tensor indexFlatten;
  SmallVector<int64_t, N> isDefinedTensor;
  SmallVector<Tensor, N> indicesTensorList;
  for (int64_t i = 0; i < indices.size(); i++) {
    if (indices[i].defined()) {
      if (indices[i].scalar_type() == at::kBool) {
        Tensor temp = convert_bools_to_tensor(indices[i]);
        indexFlatten = temp.reshape(temp.numel());
      } else{
        if (indices[i].numel() != definedMaxNum) {
          indexFlatten = at::native::repeat(indices[i].reshape(indices[i].numel()),
                                            {definedMaxNum / indices[i].numel()});
        } else {
          indexFlatten = indices[i].reshape(indices[i].numel());
        }
      }
      isDefinedTensor.emplace_back(1);
    } else {
      indexFlatten = at::arange(self.size(i), self.options().dtype(at::kLong));
      isDefinedTensor.emplace_back(0);
    }
    indicesTensorList.emplace_back(indexFlatten);
  }
  if (indices.size() < self.dim()) {
    for (int64_t i = indices.size(); i < self.dim(); i++) {
      indexFlatten = at::arange(self.size(i), self.options().dtype(at::kLong));
      indicesTensorList.emplace_back(indexFlatten);
      isDefinedTensor.emplace_back(0);
    }
  }
  return std::tuple<SmallVector<Tensor, N>, SmallVector<int64_t, N>>(indicesTensorList, isDefinedTensor);
}

SmallVector<Tensor, N> contiguous_indices_repeat(SmallVector<Tensor, N> indicesTensorList,
                                                 SmallVector<int64_t, N> isDefinedTensor,
                                                 int64_t lastdefinedTensor,
                                                 int64_t broadNum) {
  int64_t sumCo = 1;
  int64_t sumCo1 = 1;
  Tensor tempIndex;
  Tensor indexTransp;
  Tensor indexReshape;
  SmallVector<Tensor, N> contiguousList;

  for (int64_t i = indicesTensorList.size() - 1; i >= 0; i--) {
    if (i == indicesTensorList.size() - 1) {
      contiguousList.emplace_back(at::native::repeat(indicesTensorList[i], {broadNum / indicesTensorList[i].numel()}));
      sumCo = sumCo * indicesTensorList[i].numel();
    } else {
      if (isDefinedTensor[i] == 1) {
        if (lastdefinedTensor == indicesTensorList.size() - 1) {
          contiguousList.emplace_back(at::native::repeat(indicesTensorList[i], {broadNum / indicesTensorList[i].numel()}));
        } else {
          if (i == lastdefinedTensor) {
            sumCo1 = sumCo;
            tempIndex = at::native::repeat(indicesTensorList[i], {sumCo});
            indexTransp = tempIndex.reshape({sumCo, indicesTensorList[i].numel()}).transpose(1, 0);
            sumCo = sumCo * indicesTensorList[i].numel();
          } else {
            tempIndex = at::native::repeat(indicesTensorList[i], {sumCo1});
            indexTransp = tempIndex.reshape({sumCo1, indicesTensorList[i].numel()}).transpose(1, 0);
          }
          indexReshape = indexTransp.reshape(indexTransp.numel());
          contiguousList.emplace_back(at::native::repeat(indexReshape, broadNum / indexReshape.numel()));
        }
      } else {
        tempIndex = at::native::repeat(indicesTensorList[i], {sumCo});
        indexTransp = tempIndex.reshape({sumCo, indicesTensorList[i].numel()}).transpose(1, 0);
        indexReshape = indexTransp.reshape(indexTransp.numel());
        contiguousList.emplace_back(at::native::repeat(indexReshape, broadNum / indexReshape.numel()));
        sumCo = sumCo * indicesTensorList[i].numel();
      }
    }
  }
  return contiguousList;
}

SmallVector<Tensor, N> discontiguous_indices_repeat(SmallVector<Tensor, N> indicesTensorList,
                                                    SmallVector<int64_t, N> isDefinedTensor,
                                                    int64_t broadNum) {
  Tensor tempIndex;
  Tensor indexTransp;
  Tensor indexReshape;
  int64_t sumCo = 1;
  SmallVector<Tensor, N> discontiguousList;
  for (int64_t i = indicesTensorList.size() - 1; i >= 0; i--) {
    if (isDefinedTensor[i] == 0) {
      tempIndex = at::native::repeat(indicesTensorList[i], {sumCo});
      indexTransp = tempIndex.reshape({sumCo, indicesTensorList[i].numel()}).transpose(1, 0);
      indexReshape = indexTransp.reshape(indexTransp.numel());
      discontiguousList.emplace_back(at::native::repeat(indexReshape, broadNum / indexReshape.numel()));
      sumCo = sumCo * indicesTensorList[i].numel();
    } else {
      tempIndex = at::native::repeat(indicesTensorList[i], {broadNum / indicesTensorList[i].numel()});
      indexTransp = tempIndex.reshape({broadNum / indicesTensorList[i].numel(), indicesTensorList[i].numel()}).transpose(1, 0);
      indexReshape = indexTransp.reshape(indexTransp.numel());
      discontiguousList.emplace_back(indexReshape);
    }
  }
  return discontiguousList;
}

Tensor check_indices_dim(const Tensor& self, Tensor stacklist1) {
  int64_t dim = self.dim();
  std::vector<int64_t> stridelist(dim, 1);
  stridelist[dim - 1] = 1;
  for (int64_t j = self.dim() - 1; j > 0; j--) {
    stridelist[j - 1] = stridelist[j] * self.size(j);
  }
  SmallVector<int64_t, N> indicesTensorList3;
  Tensor indices1;
  Tensor stacklist_tensor = stacklist1.to(at::Device(at::kCPU), at::kLong).contiguous();
  IntArrayRef stacklist_list(stacklist_tensor.data_ptr<int64_t>(), stacklist_tensor.numel());
  for (int64_t i = 0; i < stacklist_list.size(); i += dim) {
    int64_t stride = 0;
    for (int64_t j = 0; j < stacklist1.size(1); j++) {
      stride += (stridelist[j] * stacklist_list[j + i]);
    }
    indicesTensorList3.emplace_back(stride);
  }
  indices1 = at::tensor(IntArrayRef(indicesTensorList3), self.options().dtype(at::kLong));
  return indices1;
}

Tensor& index_put_nocheck(
    Tensor& result,
    const Tensor& self,
    const TensorList& indices,
    const Tensor& value,
    bool accumulate) {
  Tensor selfCp = self.clone();
  if (value.numel() == 0) {
    return selfCp;
  }
  auto definedNum = get_definedTensor_num(indices);
  int64_t definedMaxNum = std::get<0>(definedNum);
  int64_t lastdefinedTensor = std::get<1>(definedNum);
  int64_t firstdefinedTensor = std::get<2>(definedNum);

  auto indicesList = get_indices_list(self, indices, definedMaxNum);
  SmallVector<Tensor, N> indicesTensorList = std::get<0>(indicesList);
  SmallVector<int64_t, N> isDefinedTensor = std::get<1>(indicesList);

  int64_t undefinedMaxNum = 1;
  for (int64_t i = 0; i < indicesTensorList.size(); i++) {
    if (isDefinedTensor[i] == 0) {
      undefinedMaxNum = undefinedMaxNum * self.size(i);
    }
  }
  int64_t broadNum = definedMaxNum * undefinedMaxNum;
  bool isContiguous = true;
  for (int64_t i = lastdefinedTensor; i >= firstdefinedTensor; i--) {
    if (isDefinedTensor[i] == 0) {
      isContiguous = false;
      break;
    }
  }
  SmallVector<Tensor, N> indicesTensorList1;
  SmallVector<Tensor, N> indicesTensorList2;

  if (isContiguous) {
    indicesTensorList1 = contiguous_indices_repeat(indicesTensorList, isDefinedTensor, lastdefinedTensor, broadNum);
  } else {
    indicesTensorList1 = discontiguous_indices_repeat(indicesTensorList, isDefinedTensor, broadNum);
  }
  for (int64_t i = indicesTensorList1.size() - 1; i >= 0; i--) {
    indicesTensorList2.emplace_back(indicesTensorList1[i]);
  }
  Tensor stacklist1 = at::stack(indicesTensorList2, 1);
  Tensor lastIndex;
  Tensor valueReshapelast = value.reshape(value.numel());
  Tensor valuecast = valueReshapelast.repeat({broadNum / valueReshapelast.numel()});
  if (self.dim() > 5) {
    result = selfCp.reshape(-1);
    lastIndex = check_indices_dim(self, stacklist1);
    return result.put_(lastIndex, valuecast, accumulate);
  } else {
    OpCommand cmd;
    accumulate ? cmd.Name("ScatterNdAdd") : cmd.Name("ScatterNdUpdate");
    cmd.Input(self)
        .Input(stacklist1)
        .Input(valuecast)
        .Output(result)
        .Attr("use_locking", false)
        .Run();
    return result;
  }
}

Tensor index_put_npu(
    const Tensor& self,
    TensorList indices,
    const Tensor& value,
    bool accumulate) {
  return self.clone(at::MemoryFormat::Contiguous)
      .index_put_(indices, value, accumulate);
}

Tensor& index_put_npu_(
    Tensor& self,
    TensorList indices,
    const Tensor& value,
    const bool accumulate) {
  return at::_index_put_impl_(
      self, indices, value, accumulate, false);
}

Tensor& _index_put_impl_npu_(
    Tensor& self,
    TensorList indices,
    const Tensor& value,
    const bool accumulate,
    const bool unsafe) {
  OpPreparation::CheckMemory({self}, {self});
  OpPreparation::CastBackToOriFormat(self);

  Tensor valueCopy = value;
  Tensor selfCopy = self;
  OpPreparation::CastBackToOriFormat(valueCopy);

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(selfCopy);
    Tensor result = index_put_nocheck(
        contiguousSelf, contiguousSelf, indices, valueCopy, accumulate);
    self.copy_(result);
  } else {
    index_put_nocheck(selfCopy, selfCopy, indices, valueCopy, accumulate);
    self.copy_(selfCopy);
  }
  return self;
}
} // namespace native
} // namespace at_npu