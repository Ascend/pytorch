#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "acl/acl_base.h"
#include "acl/acl_mdl.h"
#include "acl/acl_tdt.h"
extern "C" {
acltdtChannelHandle* acltdtCreateChannelWithCapacity(uint32_t deviceId,
                                                     const char* name,
                                                     size_t capacity) {return nullptr;}

aclError acltdtDestroyChannel(acltdtChannelHandle* handle) {return 0;}

aclError acltdtReceiveTensor(const acltdtChannelHandle* handle,
                             acltdtDataset* dataset,
                             int32_t timeout) {return 0;}

acltdtDataset* acltdtCreateDataset() {return nullptr;}

aclError acltdtDestroyDataset(acltdtDataset* dataset) {return 0;}

acltdtDataItem* acltdtGetDataItem(const acltdtDataset* dataset, size_t index) {return nullptr;}

aclDataType acltdtGetDataTypeFromItem(const acltdtDataItem* dataItem) {return ACL_DT_UNDEFINED;}

void* acltdtGetDataAddrFromItem(const acltdtDataItem* dataItem) {return nullptr;}

size_t acltdtGetDimNumFromItem(const acltdtDataItem* dataItem) {return 0;}

aclError acltdtGetDimsFromItem(const acltdtDataItem* dataItem, int64_t* dims, size_t dimNum) {return 0;}

aclError acltdtDestroyDataItem(acltdtDataItem* dataItem) {return 0;}

size_t acltdtGetDatasetSize(const acltdtDataset* dataset) {return 0;}

const char *acltdtGetDatasetName(const acltdtDataset *dataset) {return nullptr;}
}