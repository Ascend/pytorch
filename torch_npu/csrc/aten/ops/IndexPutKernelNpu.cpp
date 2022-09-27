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

#include <ATen/native/IndexingUtils.h>

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

tuple<int64_t, int64_t, int64_t> get_definedTensor_num(const at::TensorList& indices) {
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
      if (indices[i].numel() > definedMaxNum) {
        definedMaxNum = indices[i].numel();
      }
      lastdefinedTensor = i;
    }
  }
  return std::tuple<int64_t, int64_t, int64_t>(definedMaxNum, lastdefinedTensor, firstdefinedTensor);
}

tuple<c10::SmallVector<at::Tensor, N>, c10::SmallVector<int64_t, N>> get_indices_list(const at::Tensor& self,
                                                                                      const at::TensorList& indices,
                                                                                      int64_t definedMaxNum) {
  at::Tensor indexFlatten;
  c10::SmallVector<int64_t, N> isDefinedTensor;
  c10::SmallVector<at::Tensor, N> indicesTensorList;
  for (int64_t i = 0; i < indices.size(); i++) {
    if (indices[i].defined()) {
      if (indices[i].numel() != definedMaxNum) {
        indexFlatten = NPUNativeFunctions::repeat(indices[i].reshape(indices[i].numel()),
                                                  {definedMaxNum / indices[i].numel()});
      } else {
        indexFlatten = indices[i].reshape(indices[i].numel());
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
  return std::tuple<c10::SmallVector<at::Tensor, N>, c10::SmallVector<int64_t, N>>(indicesTensorList, isDefinedTensor);
}

c10::SmallVector<at::Tensor, N> contiguous_indices_repeat(c10::SmallVector<at::Tensor, N> indicesTensorList,
                                                          c10::SmallVector<int64_t, N> isDefinedTensor,
                                                          int64_t lastdefinedTensor,
                                                          int64_t broadNum) {
  int64_t sumCo = 1;
  int64_t sumCo1 = 1;
  at::Tensor tempIndex;
  at::Tensor indexTransp;
  at::Tensor indexReshape;
  c10::SmallVector<at::Tensor, N> contiguousList;

  for (int64_t i = indicesTensorList.size() - 1; i >= 0; i--) {
    if (i == indicesTensorList.size() - 1) {
      contiguousList.emplace_back(NPUNativeFunctions::repeat(indicesTensorList[i], {broadNum / indicesTensorList[i].numel()}));
      sumCo = sumCo * indicesTensorList[i].numel();
    } else {
      if (isDefinedTensor[i] == 1) {
        if (lastdefinedTensor == indicesTensorList.size() - 1) {
          contiguousList.emplace_back(NPUNativeFunctions::repeat(indicesTensorList[i], {broadNum / indicesTensorList[i].numel()}));
        } else {
          if (i == lastdefinedTensor) {
            sumCo1 = sumCo;
            tempIndex = NPUNativeFunctions::repeat(indicesTensorList[i], {sumCo});
            indexTransp = tempIndex.reshape({sumCo, indicesTensorList[i].numel()}).transpose(1, 0);
            sumCo = sumCo * indicesTensorList[i].numel();
          } else {
            tempIndex = NPUNativeFunctions::repeat(indicesTensorList[i], {sumCo1});
            indexTransp = tempIndex.reshape({sumCo1, indicesTensorList[i].numel()}).transpose(1, 0);
          }
          indexReshape = indexTransp.reshape(indexTransp.numel());
          contiguousList.emplace_back(NPUNativeFunctions::repeat(indexReshape, broadNum / indexReshape.numel()));
        }
      } else {
        tempIndex = NPUNativeFunctions::repeat(indicesTensorList[i], {sumCo});
        indexTransp = tempIndex.reshape({sumCo, indicesTensorList[i].numel()}).transpose(1, 0);
        indexReshape = indexTransp.reshape(indexTransp.numel());
        contiguousList.emplace_back(NPUNativeFunctions::repeat(indexReshape, broadNum / indexReshape.numel()));
        sumCo = sumCo * indicesTensorList[i].numel();
      }
    }
  }
  return contiguousList;
}

c10::SmallVector<at::Tensor, N> discontiguous_indices_repeat(c10::SmallVector<at::Tensor, N> indicesTensorList,
                                                             c10::SmallVector<int64_t, N> isDefinedTensor,
                                                             int64_t broadNum) {
  at::Tensor tempIndex;
  at::Tensor indexTransp;
  at::Tensor indexReshape;
  int64_t sumCo = 1;
  c10::SmallVector<at::Tensor, N> discontiguousList;
  for (int64_t i = indicesTensorList.size() - 1; i >= 0; i--) {
    if (isDefinedTensor[i] == 0) {
      tempIndex = NPUNativeFunctions::repeat(indicesTensorList[i], {sumCo});
      indexTransp = tempIndex.reshape({sumCo, indicesTensorList[i].numel()}).transpose(1, 0);
      indexReshape = indexTransp.reshape(indexTransp.numel());
      discontiguousList.emplace_back(NPUNativeFunctions::repeat(indexReshape, broadNum / indexReshape.numel()));
      sumCo = sumCo * indicesTensorList[i].numel();
    } else {
      tempIndex = NPUNativeFunctions::repeat(indicesTensorList[i], {broadNum / indicesTensorList[i].numel()});
      indexTransp = tempIndex.reshape({broadNum / indicesTensorList[i].numel(), indicesTensorList[i].numel()}).transpose(1, 0);
      indexReshape = indexTransp.reshape(indexTransp.numel());
      discontiguousList.emplace_back(indexReshape);
    }
  }
  return discontiguousList;
}

at::Tensor check_indices_dim(const at::Tensor& self, at::Tensor stacklist1) {
  int64_t dim = self.dim();
  std::vector<int64_t> stridelist(dim, 1);
  stridelist[dim - 1] = 1;
  for (int64_t j = self.dim() - 1; j > 0; j--) {
    stridelist[j - 1] = stridelist[j] * self.size(j);
  }
  c10::SmallVector<int64_t, N> indicesTensorList3;
  at::Tensor indices1;
  at::Tensor stacklist_tensor = stacklist1.to(at::Device(at::kCPU), at::kLong).contiguous();
  at::IntArrayRef stacklist_list(stacklist_tensor.data_ptr<int64_t>(), stacklist_tensor.numel());
  for (int64_t i = 0; i < stacklist_list.size(); i += dim) {
    int64_t stride = 0;
    for (int64_t j = 0; j < stacklist1.size(1); j++) {
      stride += (stridelist[j] * stacklist_list[j + i]);
    }
    indicesTensorList3.emplace_back(stride);
  }
  indices1 = at::tensor(at::IntArrayRef(indicesTensorList3), self.options().dtype(at::kLong));
  return indices1;
}

at::Tensor& index_put_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::TensorList& indices,
    const at::Tensor& value,
    bool accumulate) {
  at::Tensor selfCp = self.clone();
  if (value.numel() == 0) {
    return selfCp;
  }
  auto definedNum = get_definedTensor_num(indices);
  int64_t definedMaxNum = std::get<0>(definedNum);
  int64_t lastdefinedTensor = std::get<1>(definedNum);
  int64_t firstdefinedTensor = std::get<2>(definedNum);

  auto indicesList = get_indices_list(self, indices, definedMaxNum);
  c10::SmallVector<at::Tensor, N> indicesTensorList = std::get<0>(indicesList);
  c10::SmallVector<int64_t, N> isDefinedTensor = std::get<1>(indicesList);

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

  c10::SmallVector<at::Tensor, N> indicesTensorList1;
  c10::SmallVector<at::Tensor, N> indicesTensorList2;

  if (isContiguous) {
    indicesTensorList1 = contiguous_indices_repeat(indicesTensorList, isDefinedTensor, lastdefinedTensor, broadNum);
  } else {
    indicesTensorList1 = discontiguous_indices_repeat(indicesTensorList, isDefinedTensor, broadNum);
  }

  for (int64_t i = indicesTensorList1.size() - 1; i >= 0 ; i--) {
    indicesTensorList2.emplace_back(indicesTensorList1[i]);
  }
  at::Tensor stacklist1 = at::stack(indicesTensorList2, 1);

  at::Tensor lastIndex;
  at::Tensor valueReshapelast = value.reshape(value.numel());
  at::Tensor valuecast = NPUNativeFunctions::repeat(valueReshapelast, {broadNum / valueReshapelast.numel()});

  if (self.dim() > 5) {
    selfCp = result.reshape(-1);
    lastIndex = check_indices_dim(self, stacklist1);
    return NPUNativeFunctions::put_(selfCp, lastIndex, valuecast, accumulate);
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

at::Tensor NPUNativeFunctions::index_put(
    const at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>> & indices,
    const at::Tensor& value,
    bool accumulate) {
  return self.clone(at::MemoryFormat::Contiguous).index_put_(indices, value, accumulate);
}

at::Tensor& NPUNativeFunctions::index_put_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>> & indices,
    const at::Tensor& value,
    const bool accumulate) {
  return at::_index_put_impl_(self, indices, value, accumulate, false);
}

at::Tensor& NPUNativeFunctions::_index_put_impl_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>> & indices,
    const at::Tensor& value,
    const bool accumulate,
    const bool unsafe) {
  at::native::checkIndexTensorTypes(indices);
  auto indices_cast = at::native::expandTensors(self, indices);

  OpPreparation::CastBackToOriFormat(self);
  at::Tensor valueCopy = value;
  OpPreparation::CastBackToOriFormat(valueCopy);

  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    index_put_nocheck(contiguousSelf, self, indices_cast, valueCopy, accumulate);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    index_put_nocheck(self, self, indices_cast, valueCopy, accumulate);
  }
  return self;
}
} // namespace native
} // namespace at_npu