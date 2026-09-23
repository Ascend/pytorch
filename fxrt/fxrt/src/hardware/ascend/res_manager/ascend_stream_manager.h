#ifndef FXRT_SRC_HARDWARE_ASCEND_ASCEND_STREAM_MANAGER_H_
#define FXRT_SRC_HARDWARE_ASCEND_ASCEND_STREAM_MANAGER_H_

#include <memory>
#include <vector>
#include <set>
#include <mutex>

#include "acl/acl_rt.h"
#include "common/visible.h"

namespace fxrt {
namespace device {
namespace ascend {
class FXRT_EXPORT AscendStreamMng {
 public:
  static AscendStreamMng& GetInstance();

  ~AscendStreamMng() = default;
  void Clear() {}

  void ResetResource() {
    curStreamNum_ = 0;
    curEventNum_ = 0;
  }

  uint32_t ApplyNewStream() {
    return curStreamNum_++;
  }

  uint32_t ApplyNewEvent() {
    return curEventNum_++;
  }

  aclrtEvent ApplyRtEvent();
  aclrtEvent ApplyRtEventWithFlag(uint32_t flag);
  uint32_t GetRtEventId(const aclrtEvent& event) const;
  void DestroyAllRtEvents();

  void DeleteEvent();

  void DeleteStream();

  uint32_t GetCurAllocStreamId() const;

  uint32_t cur_stream_num() const {
    return curStreamNum_;
  }

  uint32_t cur_event_num() const {
    return curEventNum_;
  }

  void CreateStream(aclrtStream* stream, int32_t priority = 0);
  void CreateStream(size_t* streamId, int32_t priority = 0);
  void RegCallback(aclrtStream stream);
  void UnRegCallback(aclrtStream stream, bool delete_item = true);
  void CreateStreamWithFlags(aclrtStream* stream, uint32_t flags, int32_t priority = 0);
  void CreateStreamWithFlags(size_t* streamId, uint32_t flags, int32_t priority = 0);
  bool DestroyStream(size_t streamId);
  bool DestroyAllStreams();
  bool ForceDestroyAllStreams();
  aclrtStream GetStream(size_t streamId) const;
  void SetCurrentStream(void* currentStream);
  void* GetCurrentStream() const;
  bool SyncStream(size_t streamId) const;
  bool SyncStream(aclrtStream stream) const;
  // 'syncDevice' means whether calling 'aclrtSynchronizeDeviceWithTimeout' or 'aclrtSynchronizeStreamWithTimeout'.
  bool SyncAllStreams(bool syncDevice = true) const;
  bool SyncNotDefaultStreams() const;
  // Sync all streams except the streams in exceptStreams.
  bool SyncExceptStreamsInList(const std::set<aclrtStream>& exceptStreams) const;
  size_t QueryStreamSize() const;
  bool QueryStream(size_t streamId);
  size_t GetStreamId(void* streamPtr);
  std::vector<uint32_t> GetStreamIds() const;
  void SetBusyStreamNum(uint32_t stream_num) {
    busyStreamNum_ = stream_num;
  }
  uint32_t GetBusyStreamNum() const {
    return busyStreamNum_;
  }
  void SetCopyInStream(aclrtStream stream) {
    copyInStream_ = stream;
  }
  void SetCopyOutStream(aclrtStream stream) {
    copyOutStream_ = stream;
  }
  void SetForwardSendStream(aclrtStream stream) {
    forwardSendStream_ = stream;
  }
  void SetBackwardSendStream(aclrtStream stream) {
    backwardSendStream_ = stream;
  }
  void SetForwardRecvStream(aclrtStream stream) {
    forwardRecvStream_ = stream;
  }
  void SetBackwardRecvStream(aclrtStream stream) {
    backwardRecvStream_ = stream;
  }
  aclrtStream GetCopyInStream() const {
    return copyInStream_;
  }
  aclrtStream GetCopyOutStream() const {
    return copyOutStream_;
  }
  aclrtStream GetForwardSendStream() const {
    return forwardSendStream_;
  }
  aclrtStream GetBackwardSendStream() const {
    return backwardSendStream_;
  }
  aclrtStream GetForwardRecvStream() const {
    return forwardRecvStream_;
  }
  aclrtStream GetBackwardRecvStream() const {
    return backwardRecvStream_;
  }

  void set_current_stream(size_t streamId) {
    currentStreamId_ = streamId;
  }
  size_t current_stream() const {
    return currentStreamId_;
  }

  void CreateDefaultStream();
  size_t default_stream_id() const;
  size_t communication_stream_id() const;
  aclrtStream default_stream() const;
  aclrtStream communication_stream() const;

  bool singleOpMultiStreamEnable() const {
    return singleOpMultiStreamEnable_;
  }
  void set_single_op_multi_stream_enable(bool singleOpMultiStreamEnable) {
    singleOpMultiStreamEnable_ = singleOpMultiStreamEnable;
  }

 private:
  // Count streams and events number in task sink scenario
  uint32_t curStreamNum_{0};
  uint32_t curEventNum_{0};

  // The max stream num on device ar a time
  uint32_t busyStreamNum_{0};

  // Ensure the thread safety for creating and destroying stream.
  std::mutex streamMutex_;
  aclrtStream copyInStream_{nullptr};
  aclrtStream copyOutStream_{nullptr};
  aclrtStream forwardSendStream_{nullptr};
  aclrtStream backwardSendStream_{nullptr};
  aclrtStream forwardRecvStream_{nullptr};
  aclrtStream backwardRecvStream_{nullptr};

  // all gpu CUDA streams including defaultStream_.
  std::vector<void*> streams_;
  std::vector<aclrtEvent> events_{};

  // Currently using stream id.
  size_t currentStreamId_{0};

  // Default stream. We consider the first stream created as default stream.
  aclrtStream defaultStream_{nullptr};
  size_t defaultStreamId_{0};
  aclrtStream communicationStream_{nullptr};
  size_t communicationStreamId_{0};
  void* currentStream_{nullptr};

  bool singleOpMultiStreamEnable_{false};
};
} // namespace ascend
} // namespace device
} // namespace fxrt

#endif // FXRT_SRC_HARDWARE_ASCEND_ASCEND_STREAM_MANAGER_H_
