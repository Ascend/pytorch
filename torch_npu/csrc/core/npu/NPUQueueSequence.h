#pragma once

#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>

#include <c10/core/Device.h>
#include "torch_npu/csrc/logging/LogContext.h"
#include "torch_npu/csrc/core/npu/ExperimentalSequenceMpscQueueCore.h"
#include <acl/acl_op.h>

namespace c10_npu {

inline std::shared_ptr<npu_logging::Logger>& GetQueueLogger()
{
    static std::shared_ptr<npu_logging::Logger> logger = npu_logging::logging().getLogger("torch_npu.dispatch");
    return logger;
}

#define TORCH_NPU_QUEUE_LOGD(format, ...)                                      \
    do {                                                                       \
        TORCH_NPU_LOGD(c10_npu::GetQueueLogger(), format, ##__VA_ARGS__);      \
        ASCEND_LOGD(format, ##__VA_ARGS__);                                    \
    } while (0);

#define TORCH_NPU_QUEUE_LOGI(format, ...)                                      \
    do {                                                                       \
        TORCH_NPU_LOGI(c10_npu::GetQueueLogger(), format, ##__VA_ARGS__);      \
        ASCEND_LOGI(format, ##__VA_ARGS__);                                    \
    } while (0);

enum RepoStatus {
  INIT = 0,
  RUN = 1,
  NEED_EXIT = 2,
  CAN_EXIT = 3,
  ERROR_EXIT = 4,
  UCE_EXIT = 5,
  STOP_EXIT = 6,
  HBM_ECC_EXIT = 7,
  SUSPECT_MEM_EXIT = 8,
  HCCS_LINK_EXIT = 9,
  HCCL_OP_RETRY_EXIT = 10,
  SUSPECT_REMOTE_EXIT = 11,
};

enum class Lifecycle {
  Uninitialized,
  Initializing,
  Running,
  Stopping,
  Stopped,
};

struct ErrorRecord {
  std::string repository;
  std::string acl;
  std::string queue;
  std::string current_parameter;
};

// c10::SmallVector max size
const int N = 32;

class ReleaseQueue {
public:
  ReleaseQueue() = default;
  ~ReleaseQueue();
  void PushToReleaseQueue(void* cur_paras);
  void PopFromReleaseQueue();
  void InitReleaseQueue(c10::DeviceIndex device_id);
  void Shutdown();
  RepoStatus GetStatus() const;
  c10::DeviceIndex GetDeviceID() const;

private:
  friend void StartRelease(ReleaseQueue* releaseQue);
  void SetStatus(RepoStatus desired);
  void ChangeStatus(RepoStatus expected, RepoStatus desired);

private:
  void* data = nullptr;
  std::thread releaser;
  c10::DeviceIndex device_idx;

private:
  using QueueCore = queue::ExperimentalSequenceMpscQueueCore;
  QueueCore queue_core_{8192};
  std::atomic<RepoStatus> repo_status{RepoStatus::INIT};
  std::atomic<Lifecycle> lifecycle_{Lifecycle::Uninitialized};
  std::mutex lifecycle_mu_;
};

class NPUQueueBase {
public:
  virtual ~NPUQueueBase() {}
  virtual RepoStatus GetStatus() const = 0;
  virtual void SetStatus(RepoStatus desired) = 0;
  virtual void ChangeStatus(RepoStatus expected, RepoStatus desired) = 0;
  virtual void Enqueue(void* cur_paras) = 0;
  virtual void Dequeue() = 0;
  virtual NPUStatus MakeSureQueueEmpty(bool check_error = true) = 0;
  virtual void InitRepo(c10::DeviceIndex device_id) = 0;
  virtual bool CheckInit() const = 0;
  virtual std::string GetPara() = 0;
  virtual void ClearQueue() = 0;
  virtual void SetQueueErrMsg(const char* errmsg) = 0;
  virtual std::string GetQueueErrMsg() = 0;
  virtual bool IsEmptyRepo() = 0;
};

class NPUQueueFactoryBase {
public:
  virtual NPUQueueBase* create() = 0;
  virtual ~NPUQueueFactoryBase() {}
};

class Repository : public NPUQueueBase {
  friend void StartConsume(Repository* repo, c10::DeviceIndex device_id);
public:
  Repository() = default;
  ~Repository() override;
  RepoStatus GetStatus() const override;
  void SetStatus(RepoStatus desired) override;
  void ChangeStatus(RepoStatus expected, RepoStatus desired) override;
  void Enqueue(void* cur_paras) override;
  void Dequeue() override;
  NPUStatus MakeSureQueueEmpty(bool check_error = true) override;
  void InitRepo(c10::DeviceIndex device_id) override;
  bool CheckInit() const override;
  std::string GetPara() override;
  void ClearQueue() override;
  void SetQueueErrMsg(const char *errmsg) override;
  std::string GetQueueErrMsg() override;
  bool IsEmptyRepo() override { return queue_core_.Snapshot().size == 0; }

private:
  void ReleaseResource();
  bool ReadQueue();
  void CheckDeviceError(int ret, std::string& err_msg);
  void ThrowDeviceError(RepoStatus current_status, void* cur_paras);
  ErrorRecord SnapshotError() const;
  void RecordRepositoryError(std::string message);
  void RecordAclError(std::string message);
  void RecordCurrentParameter(std::string message);

private:
  void* data = nullptr;
  std::thread consumer;
  std::atomic<bool> consumer_stop_requested_{false};
  c10::DeviceIndex device_idx;

private:
  using QueueCore = queue::ExperimentalSequenceMpscQueueCore;
  QueueCore queue_core_{4096};
  std::atomic<RepoStatus> repo_status{RepoStatus::INIT};
  std::atomic<Lifecycle> lifecycle_{Lifecycle::Uninitialized};
  std::mutex lifecycle_mu_;
  std::condition_variable lifecycle_cv_;
  uint64_t status_epoch_ = 0;
  mutable std::mutex error_mu_;
  ErrorRecord error_;
  ReleaseQueue releaseQueue;
};

using ACL_EXEC_FUNC     = std::function<int(void*)>;
using ACL_COPY_FUNC     = std::function<void(void*, void*)>;
using ACL_RELEASE_FUNC  = std::function<void(void*, ReleaseQueue&)>;
using ACL_NEW_FUNC      = std::function<void*(int, int&)>;
using ACL_DELETE_FUNC   = std::function<void(void*)>;
using ACL_COPY_RELEASE_PARM_FUNC = std::function<void(void*, void*)>;
using ACL_RELEASE_PARAM_FUNC = std::function<void(void*)>;

namespace register_queue_cb {
class NPUCallBackRegisterBuilder {
public:
  NPUCallBackRegisterBuilder(const ACL_EXEC_FUNC& execF, const ACL_COPY_FUNC& copyF,
    const ACL_RELEASE_FUNC& releaseF, const ACL_NEW_FUNC& newF, const ACL_DELETE_FUNC& deleteF,
    const ACL_COPY_RELEASE_PARM_FUNC& copyReleaseParamF, const ACL_RELEASE_PARAM_FUNC& releaseParamF);
  ~NPUCallBackRegisterBuilder() {}
};
} // namespace register_queue_cb

#define REGISTER_QUEUE_FUNC(execF, copyF, releaseF, newF, deleteF, copyReleaseParamF, releaseParamF)  \
    static ::c10_npu::register_queue_cb::NPUCallBackRegisterBuilder                     \
        register_queue_func_builder(execF, copyF, releaseF, newF, deleteF, copyReleaseParamF, releaseParamF);
} // namespace c10_npu
