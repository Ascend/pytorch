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

#ifndef __C10_NPU_INTERFACE_ACLTDTINTERFACE__
#define __C10_NPU_INTERFACE_ACLTDTINTERFACE__
#include <third_party/acl/inc/acl/acl_tdt.h>

namespace c10_npu {
namespace acl_tdt {
acltdtChannelHandle* AcltdtCreateChannelWithCapacity(uint32_t deviceId,
                                                     const char* name,
                                                     size_t capacity);

aclError AcltdtDestroyChannel(acltdtChannelHandle* handle);

aclError AcltdtReceiveTensor(const acltdtChannelHandle* handle,
                             acltdtDataset* dataset,
                             int32_t timeout);

acltdtDataset* AcltdtCreateDataset();

aclError AcltdtDestroyDataset(acltdtDataset* dataset);

acltdtDataItem* AcltdtGetDataItem(const acltdtDataset* dataset, size_t index);

aclDataType AcltdtGetDataTypeFromItem(const acltdtDataItem* dataItem);

void* AcltdtGetDataAddrFromItem(const acltdtDataItem* dataItem);

size_t AcltdtGetDimNumFromItem(const acltdtDataItem* dataItem);

aclError AcltdtGetDimsFromItem(const acltdtDataItem* dataItem, int64_t* dims, size_t dimNum);

aclError AcltdtDestroyDataItem(acltdtDataItem* dataItem);

size_t AcltdtGetDatasetSize(const acltdtDataset* dataset);

const char* AcltdtGetDatasetName(const acltdtDataset* dataset);
} // namespace acl_tdt
} // namespace c10_npu
#endif

