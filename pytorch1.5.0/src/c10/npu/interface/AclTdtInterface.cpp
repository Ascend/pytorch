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

#include "AclTdtInterface.h"
#include <c10/npu/register/FunctionLoader.h>
#include <c10/util/Exception.h>

namespace c10 {
namespace npu {
namespace acl_tdt {
#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libacl_tdt_channel, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName)              \
  GET_FUNCTION(libacl_tdt_channel, funcName)

REGISTER_LIBRARY(libacl_tdt_channel)
LOAD_FUNCTION(acltdtCreateChannelWithCapacity)
LOAD_FUNCTION(acltdtDestroyChannel)
LOAD_FUNCTION(acltdtReceiveTensor)
LOAD_FUNCTION(acltdtCreateDataset)
LOAD_FUNCTION(acltdtDestroyDataset)
LOAD_FUNCTION(acltdtGetDataItem)
LOAD_FUNCTION(acltdtGetDataTypeFromItem)
LOAD_FUNCTION(acltdtGetDataAddrFromItem)
LOAD_FUNCTION(acltdtGetDimNumFromItem)
LOAD_FUNCTION(acltdtGetDimsFromItem)
LOAD_FUNCTION(acltdtDestroyDataItem)
LOAD_FUNCTION(acltdtGetDatasetSize)
LOAD_FUNCTION(acltdtGetDatasetName)

acltdtChannelHandle* AcltdtCreateChannelWithCapacity(uint32_t deviceId,
                                                     const char* name,
                                                     size_t capacity) {
  typedef acltdtChannelHandle* (*AcltdtCreateChannelWithCapacityFunc)
          (uint32_t, const char*, size_t);
  static AcltdtCreateChannelWithCapacityFunc func = nullptr;
  if (func == nullptr) {
    func = (AcltdtCreateChannelWithCapacityFunc)GET_FUNC(acltdtCreateChannelWithCapacity);
  }
  TORCH_CHECK(func, "Failed to find function ", "acltdtCreateChannelWithCapacity");
  return func(deviceId, name, capacity);
}

aclError AcltdtDestroyChannel(acltdtChannelHandle* handle) {
  typedef aclError (*AcltdtDestroyChannelFunc)(acltdtChannelHandle*);
  static AcltdtDestroyChannelFunc func = nullptr;
  if (func == nullptr) {
    func = (AcltdtDestroyChannelFunc)GET_FUNC(acltdtDestroyChannel);
  }
  TORCH_CHECK(func, "Failed to find function ", "acltdtDestroyChannel");
  return func(handle);
}

aclError AcltdtReceiveTensor(const acltdtChannelHandle* handle,
                             acltdtDataset* dataset,
                             int32_t timeout) {
  typedef aclError (*AcltdtReceiveTensorFunc)
          (const acltdtChannelHandle*, acltdtDataset*, int32_t);
  static AcltdtReceiveTensorFunc func = nullptr;
  if (func == nullptr) {
    func = (AcltdtReceiveTensorFunc)GET_FUNC(acltdtReceiveTensor);
  }
  TORCH_CHECK(func, "Failed to find function ", "acltdtReceiveTensor");
  return func(handle, dataset, timeout);
}

acltdtDataset* AcltdtCreateDataset() {
  typedef acltdtDataset* (*AcltdtCreateDatasetFunc)();
  static AcltdtCreateDatasetFunc func = nullptr;
  if (func == nullptr) {
    func = (AcltdtCreateDatasetFunc)GET_FUNC(acltdtCreateDataset);
  }
  TORCH_CHECK(func, "Failed to find function ", "acltdtCreateDataset");
  return func();
}

aclError AcltdtDestroyDataset(acltdtDataset* dataset) {
  typedef aclError (*AcltdtDestroyDatasetFunc)(acltdtDataset*);
  static AcltdtDestroyDatasetFunc func = nullptr;
  if (func == nullptr) {
    func = (AcltdtDestroyDatasetFunc)GET_FUNC(acltdtDestroyDataset);
  }
  TORCH_CHECK(func, "Failed to find function ", "acltdtDestroyDataset");
  return func(dataset);
}

acltdtDataItem* AcltdtGetDataItem(const acltdtDataset* dataset, size_t index) {
  typedef acltdtDataItem* (*AcltdtGetDataItemFunc)(const acltdtDataset*, size_t);
  static AcltdtGetDataItemFunc func = nullptr;
  if (func == nullptr) {
    func = (AcltdtGetDataItemFunc)GET_FUNC(acltdtGetDataItem);
  }
  TORCH_CHECK(func, "Failed to find function ", "acltdtGetDataItem");
  return func(dataset, index);
}

aclDataType AcltdtGetDataTypeFromItem(const acltdtDataItem* dataItem) {
  typedef aclDataType (*AcltdtGetDataTypeFromItemFunc)(const acltdtDataItem*);
  static AcltdtGetDataTypeFromItemFunc func = nullptr;
  if (func == nullptr) {
    func = (AcltdtGetDataTypeFromItemFunc)GET_FUNC(acltdtGetDataTypeFromItem);
  }
  TORCH_CHECK(func, "Failed to find function ", "acltdtGetDataTypeFromItem");
  return func(dataItem);
}

void* AcltdtGetDataAddrFromItem(const acltdtDataItem* dataItem) {
  typedef void* (*AcltdtGetDataAddrFromItemFunc)(const acltdtDataItem*);
  static AcltdtGetDataAddrFromItemFunc func = nullptr;
  if (func == nullptr) {
    func = (AcltdtGetDataAddrFromItemFunc)GET_FUNC(acltdtGetDataAddrFromItem);
  }
  TORCH_CHECK(func, "Failed to find function ", "acltdtGetDataAddrFromItem");
  return func(dataItem);
}

size_t AcltdtGetDimNumFromItem(const acltdtDataItem* dataItem) {
  typedef size_t (*AcltdtGetDimNumFromItemFunc)(const acltdtDataItem*);
  static AcltdtGetDimNumFromItemFunc func = nullptr;
  if (func == nullptr) {
    func = (AcltdtGetDimNumFromItemFunc)GET_FUNC(acltdtGetDimNumFromItem);
  }
  TORCH_CHECK(func, "Failed to find function ", "acltdtGetDimNumFromItem");
  return func(dataItem);
}

aclError AcltdtGetDimsFromItem(const acltdtDataItem* dataItem, int64_t* dims, size_t dimNum) {
  typedef aclError (*AcltdtGetDimsFromItemFunc)(const acltdtDataItem*, int64_t*, size_t);
  static AcltdtGetDimsFromItemFunc func = nullptr;
  if (func == nullptr) {
    func = (AcltdtGetDimsFromItemFunc)GET_FUNC(acltdtGetDimsFromItem);
  }
  TORCH_CHECK(func, "Failed to find function ", "acltdtGetDimsFromItem");
  return func(dataItem, dims, dimNum);
}

aclError AcltdtDestroyDataItem(acltdtDataItem* dataItem) {
  typedef aclError (*AcltdtDestroyDataItemFunc)(const acltdtDataItem*);
  static AcltdtDestroyDataItemFunc func = nullptr;
  if (func == nullptr) {
    func = (AcltdtDestroyDataItemFunc)GET_FUNC(acltdtDestroyDataItem);
  }
  TORCH_CHECK(func, "Failed to find function ", "acltdtDestroyDataItem");
  return func(dataItem);
}

size_t AcltdtGetDatasetSize(const acltdtDataset* dataset) {
  typedef size_t (*AcltdtGetDatasetSizeFunc)(const acltdtDataset*);
  static AcltdtGetDatasetSizeFunc func = nullptr;
  if (func == nullptr) {
    func = (AcltdtGetDatasetSizeFunc)GET_FUNC(acltdtGetDatasetSize);
  }
  TORCH_CHECK(func, "Failed to find function ", "acltdtGetDatasetSize");
  return func(dataset);
}

const char* AcltdtGetDatasetName(const acltdtDataset* dataset) {
  typedef char* (*AcltdtGetDatasetNameFunc)(const acltdtDataset*);
  static AcltdtGetDatasetNameFunc func = nullptr;
  if (func == nullptr) {
    func = (AcltdtGetDatasetNameFunc)GET_FUNC(acltdtGetDatasetName);
  }
  TORCH_CHECK(func, "Failed to find function ", "acltdtGetDatasetName");
  return func(dataset);
}
}
}
}