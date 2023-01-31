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
#include <thread>
#include <mutex>

#include <ATen/core/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/Tensor.h>
#include <c10/core/CPUAllocator.h> 

#include "TdtChannelForPrint.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"

namespace at_npu {
namespace native {
namespace {
const int32_t kChannelTimeOut = 500;
const int32_t kChannelCapacity = 3;
}
using namespace c10_npu;
bool TdtChannelForPrint::Init() {
  std::lock_guard<std::mutex> lock(channel_mutex_);
  if (channel_ == nullptr) {
    channel_ = new NpuTdtChannel(kChannelTimeOut, kChannelCapacity, "TDTChannelForPrint");
  }
  TORCH_CHECK(channel_ != nullptr, "Channel is none during Init TdtChannelForPrint");
  return channel_->Init();
}

TdtChannelForPrint& TdtChannelForPrint::GetInstance() {
  static TdtChannelForPrint channel_for_print;
  return channel_for_print;
}

std::shared_ptr<TdtDataSet> TdtChannelForPrint::GetNextDatasetToPrint() {
  std::lock_guard<std::mutex> lock(channel_mutex_);
  if (channel_ == nullptr) {
    return nullptr;
  }
  return channel_->Dequeue();
}

TupleToPrint TdtChannelForPrint::GetTupleToPrint() {
  auto tdt_data_set = this->GetNextDatasetToPrint();
  if (tdt_data_set == nullptr) {
    TupleToPrint tuple_to_print;
    return tuple_to_print;
  }
  auto data_set = tdt_data_set.get()->GetPtr();
  TORCH_CHECK(data_set != nullptr, "Get item to be printed failed");
  auto data_size = acl_tdt::AcltdtGetDatasetSize(data_set.get());
  std::vector<at::Tensor> tensor_to_print;
  for (size_t i = 0UL; i < data_size; i++) {
    auto data_item = acl_tdt::AcltdtGetDataItem(data_set.get(), i);
    void* data_addr = acl_tdt::AcltdtGetDataAddrFromItem(data_item);
    size_t dim_size = acl_tdt::AcltdtGetDimNumFromItem(data_item);
    int64_t dims[dim_size];
    acl_tdt::AcltdtGetDimsFromItem(data_item, dims, dim_size);

    c10::SmallVector<int64_t, 5> sizes;
    for (auto& j : dims) {
      sizes.emplace_back(j);
    }

    auto data_type = acl_tdt::AcltdtGetDataTypeFromItem(data_item);
    auto at_data_type = CalcuOpUtil::ConvertToATDataType(data_type);

    auto options = c10::TensorOptions().dtype(at_data_type);
    at::Tensor tensor = at::empty(sizes, options);
    (void)memcpy(tensor.data_ptr(), data_addr, tensor.numel() * tensor.itemsize());
    tensor_to_print.emplace_back(std::move(tensor));
    
  }
  const char* desc_name = acl_tdt::AcltdtGetDatasetName(data_set.get());
  const std::string format_string(desc_name);

  TupleToPrint tuple_to_print;

  std::get<0>(tuple_to_print) = tensor_to_print;
  std::get<1>(tuple_to_print) = format_string;

  return tuple_to_print;
}
} // namespace native
} // namespace at_npu

