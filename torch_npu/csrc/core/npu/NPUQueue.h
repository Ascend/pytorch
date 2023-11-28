#pragma once

#include <string>
#include <thread>
#include <mutex>
#include <atomic>

#include <c10/core/Device.h>
#include "torch_npu/csrc/core/npu/npu_log.h"
#include <third_party/acl/inc/acl/acl_op.h>

namespace c10_npu {

struct sring_idx {
  bool working = false;
  volatile unsigned int idx = 0;
};

enum RepoStatus {
  INIT = 0,
  RUN = 1,
  NEED_EXIT = 2,
  CAN_EXIT = 3,
  ERROR_EXIT = 4,
};

// c10::SmallVector max size
const int N = 32;

class ReleaseQueue {
public:
  ReleaseQueue() = default;
  ~ReleaseQueue();
  void PushToReleaseQueue(void* cur_paras);
  void PopFromReleaseQueue();
  void InitReleaseQueue();
  RepoStatus GetStatus() const;

private:
  inline bool IsEmptyQueue() {return read_idx.idx == write_idx.idx;};
  bool IsFullQueue() const;
  bool WriteToReleaseQueue(void* cur_paras);
  bool ReadFromReleaseQueue();
  void SetStatus(RepoStatus desired);
  void ChangeStatus(RepoStatus expected, RepoStatus desired);

private:
  void* datas = nullptr;
  std::thread releaser;

private:
  sring_idx read_idx;
  sring_idx write_idx;
  std::atomic<RepoStatus> repo_status;
  bool initialized = false;
};

class NPUQueueBase {
public:
  virtual ~NPUQueueBase() {}
  virtual RepoStatus GetStatus() const = 0;
  virtual void SetStatus(RepoStatus desired) = 0;
  virtual void ChangeStatus(RepoStatus expected, RepoStatus desired) = 0;
  virtual void Enqueue(void* cur_paras) = 0;
  virtual void Dequeue() = 0;
  virtual NPUStatus MakeSureQueueEmpty() = 0;
  virtual void InitRepo(c10::DeviceIndex device_id) = 0;
  virtual bool CheckInit() const = 0;
};

class NPUQueueFactoryBase {
public:
  virtual NPUQueueBase* create() = 0;
  virtual ~NPUQueueFactoryBase() {}
};

class Repository : public NPUQueueBase {
public:
  Repository() = default;
  ~Repository() override;
  RepoStatus GetStatus() const override;
  void SetStatus(RepoStatus desired) override;
  void ChangeStatus(RepoStatus expected, RepoStatus desired) override;
  void Enqueue(void* cur_paras) override;
  void Dequeue() override;
  NPUStatus MakeSureQueueEmpty() override;
  void InitRepo(c10::DeviceIndex device_id) override;
  bool CheckInit() const override;

private:
  void ReleaseResource();
  inline bool IsEmptyQueue() {return read_idx.idx == write_idx.idx;};
  bool IsFullQueue() const;
  void SetWriteWorking(bool isWorking) {write_idx.working = isWorking;};
  void SetReadWorking(bool isWorking) {read_idx.working = isWorking;};
  bool IsWriteWorking() const {return write_idx.working;};
  bool IsReadWorking() const {return read_idx.working;};
  bool WriteQueue(void* cur_paras);
  bool ReadQueue();

private:
  void* datas = nullptr;
  std::thread consumer;
  int efd_read;
  int efd_write;
  int efd_empty;
  c10::DeviceIndex device_idx;

private:
  sring_idx read_idx;
  sring_idx write_idx;
  std::atomic<RepoStatus> repo_status;
  bool need_empty = false;
  bool initialized = false;
  std::mutex mu_empty;
  // In theory, this is not necessary.
  // The logic is ensured by original pytorch, but this is added here just in
  // case.
  std::mutex mu_enqueue;
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