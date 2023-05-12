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

