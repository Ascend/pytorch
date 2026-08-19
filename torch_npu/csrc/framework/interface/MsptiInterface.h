#ifndef __TORCH_NPU_MSPTIINTERFACE__
#define __TORCH_NPU_MSPTIINTERFACE__

#include <third_party/mspti/mspti_activity.h>
#include <third_party/mspti/mspti_callback.h>

namespace at_npu {
namespace native {

bool IsSupportMsptiFunc();

bool MsptiActivityIsEnabled(msptiActivityKind kind);

bool MsptiActivityEnable(msptiActivityKind kind);
bool MsptiActivityDisable(msptiActivityKind kind);
bool MsptiActivityRegisterCallbacks(
    msptiBuffersCallbackRequestFunc funcBufferRequested,
    msptiBuffersCallbackCompleteFunc funcBufferCompleted);
bool MsptiActivityFlushAll(uint32_t flag);
bool MsptiActivityGetNextRecord(uint8_t* buffer, size_t validBufferSizeBytes, msptiActivity** record);

bool MsptiActivityPushExternalCorrelationId(msptiExternalCorrelationKind kind, uint64_t id);
bool MsptiActivityPopExternalCorrelationId(msptiExternalCorrelationKind kind, uint64_t* lastId);

bool MsptiSubscribe(msptiSubscriberHandle* subscriber);
bool MsptiUnsubscribe(msptiSubscriberHandle subscriber);

} // namespace native
} // namespace at_npu

#endif
