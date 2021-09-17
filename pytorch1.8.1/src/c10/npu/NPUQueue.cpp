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

#include "c10/npu/NPUQueue.h"
#include "c10/npu/NPUStream.h"
#include "c10/npu/npu_log.h"

#include <Python.h>

#include <sys/eventfd.h>
#include <sys/prctl.h>
#include <third_party/acl/inc/acl/acl_rt.h>

//#define OPEN_QUEUE_DEBUG
#ifdef OPEN_QUEUE_DEBUG
#define QUEUE_DEBUG(fmt, ...)                                      \
  do {                                                             \
    printf("[%s:%d]" fmt "\n", __func__, __LINE__, ##__VA_ARGS__); \
  } while (0)
#else
#define QUEUE_DEBUG(fmt, ...)
#endif

//#define OPEN_QUEUE_COUT
#ifdef OPEN_QUEUE_COUT
#define QUEUE_COUT(fmt, ...)                                       \
  do {                                                             \
    printf("[%s:%d]" fmt "\n", __func__, __LINE__, ##__VA_ARGS__); \
  } while (0)
#else
#define QUEUE_COUT(fmt, ...)
#endif

namespace c10 {
namespace npu {

namespace {

class CallBackManager {
public:
  CallBackManager() {}
  ~CallBackManager() {}
  void SetExec(const ACL_EXEC_FUNC& func) {
    this->execFunc = func;
  }

  void SetCopy(const ACL_COPY_FUNC& func) {
    this->copyFunc = func;
  }

  void SetRelease(const ACL_RELEASE_FUNC& func) {
    this->releaseFunc = func;
  }

  void SetNew(const ACL_NEW_FUNC& func) {
    this->newFunc = func;
  }

  void SetDelete(const ACL_DELETE_FUNC& func) {
    this->deleteFunc = func;
  }

  int Call(void* head, int offset, aclrtStream stream) {
    TORCH_CHECK(this->execFunc, "Failed to find execution function.");
    auto dstPtr = (uint8_t*)head + sizePerParams * offset;
    return this->execFunc(dstPtr, stream);
  }

  void Copy(void* dstHead, int offset, void* src) {
    TORCH_CHECK(this->copyFunc, "Failed to find copy function.");
    auto dstPtr = (uint8_t*)dstHead + sizePerParams * offset;
    return this->copyFunc(dstPtr, src);
  }

  void Release(void* head, int offset) {
    TORCH_CHECK(this->releaseFunc, "Failed to find release function.");
    auto ptr = (uint8_t*)head +  sizePerParams * offset;
    return this->releaseFunc(ptr);
  }

  void* Init(int capacity) {
    TORCH_CHECK(this->newFunc, "Failed to find new function.");
    void* ptr = this->newFunc(capacity, sizePerParams); // not check as CUDA
    return ptr;
  }

  void DeInit(void* ptr) {
    if (ptr != nullptr) {
      TORCH_CHECK(this->deleteFunc, "Failed to find delete function.");
      this->deleteFunc(ptr);
      ptr = nullptr;
    }
  }
private:
  int sizePerParams = 0;
  ACL_EXEC_FUNC execFunc = nullptr;
  ACL_COPY_FUNC copyFunc = nullptr;
  ACL_RELEASE_FUNC releaseFunc = nullptr;
  ACL_NEW_FUNC newFunc = nullptr;
  ACL_DELETE_FUNC deleteFunc = nullptr;
}; // class CallBackManager

CallBackManager& manager() {
  static CallBackManager instance;
  return instance;
}
} // namespace

namespace register_queue_cb {
NPUCallBackRegisterBuilder::NPUCallBackRegisterBuilder(const ACL_EXEC_FUNC& execFunc,
    const ACL_COPY_FUNC& copyFunc, const ACL_RELEASE_FUNC& releaseFunc,
    const ACL_NEW_FUNC& newFunc, const ACL_DELETE_FUNC& deleteFunc) {
  manager().SetExec(execFunc);
  manager().SetCopy(copyFunc);
  manager().SetRelease(releaseFunc);
  manager().SetNew(newFunc);
  manager().SetDelete(deleteFunc);
}
} // namespace register_queue_cb


// If the capacity is too large, when the queue is full,
// a large amount of device memory is occupied at the same time;
// if the capacity is too small, and the main thread is fast enough,
// it does not make full use of concurrent design capabilities.
static constexpr size_t kQueueCapacity = 1000;

RepoStatus Repository::GetStatus() const {
  if (initialized == false) {
    NPU_LOGE("Task queue is not initialized, shouldn't call GetStatus(). !!");
  }

  return repo_status.load();
}

void Repository::SetStatus(RepoStatus desired) {
  if (initialized == false) {
    NPU_LOGE("Task queue is not initialized, shouldn't call SetStatus(). !!");
    return;
  }

  repo_status = desired;
}

void Repository::ChangeStatus(RepoStatus expected, RepoStatus desired) {
  if (initialized == false) {
    NPU_LOGE(
        "Task queue is not initialized, shouldn't call ChangeStatus(). !!");
    return;
  }

  repo_status.compare_exchange_strong(expected, desired);
}

NPUStatus Repository::MakeSureQueueEmpty() {
  if (initialized == false) {
    NPU_LOGE(
        "Task queue is not initialized, shouldn't call MakeSureQueueEmpty(). !!");
    return FAILED;
  }

  if (consumer.joinable()) {
    ssize_t s;
    uint64_t u = 1;
    while (!IsEmptyQueue()) {
      std::lock_guard<std::mutex> lock(mu_empty);
      need_empty = true;
      __sync_synchronize();
      if (!IsEmptyQueue()) { // double-check, very important idea
        // While waiting for ACL thread to launch tasks,
        // the current thread should not hold GIL.
        // When the operator compilation is triggered in the ACL thread,
        // the TE module attempts to obtain the GIL.
        // If the current thread does not release the GIL, a deadlock will
        // occur.
        if (PyGILState_Check()) {
          Py_BEGIN_ALLOW_THREADS s = eventfd_read(efd_empty, &u);
          Py_END_ALLOW_THREADS
        } else {
          s = eventfd_read(efd_empty, &u);
        }
        if (s != 0) {
          NPU_LOGE("eventfd_read failed !!");
          return INTERNEL_ERROR;
        }
        QUEUE_DEBUG("waiting ok, queue is empty now");
      }
    }
    need_empty = false;
    QUEUE_DEBUG(
        "MakeSureQueueEmpty success, now write_idx=%d, read_idx=%d",
        write_idx.idx,
        read_idx.idx);
  }
  return SUCCESS;
}

void Repository::EnableInterrupt(RepoRole role) {
  if (role == RepoRole::READER) {
    read_idx.working = false;
  } else {
    write_idx.working = false;
  }
}

void Repository::DisableInterrupt(RepoRole role) {
  if (role == RepoRole::READER) {
    read_idx.working = true;
  } else {
    write_idx.working = true;
  }
}

bool Repository::NeedNotify(RepoRole role) const {
  bool working =
      (role == RepoRole::READER) ? read_idx.working : write_idx.working;
  return !working;
}

bool Repository::WriteQueue(void* cur_paras) {
  QUEUE_DEBUG("write_idx=%d, read_idx=%d", write_idx.idx, read_idx.idx);
  if (IsFullQueue()) {
    QUEUE_DEBUG("queue is full");
    return false;
  }

  std::lock_guard<std::mutex> lock(mu_enqueue);
  manager().Copy(datas, write_idx.idx, cur_paras);
  __sync_synchronize();

  write_idx.idx++;
  write_idx.idx %= kQueueCapacity;
  return true;
}

bool Repository::ReadQueue() {
  QUEUE_DEBUG("write_idx=%d, read_idx=%d", write_idx.idx, read_idx.idx);
  if (IsEmptyQueue()) {
    QUEUE_DEBUG("queue is empty");
    return false;
  }

  auto ret = manager().Call(datas, read_idx.idx, calcu_stream_);

  if (ret != 0) {
    while (!IsEmptyQueue()) { // ignore other tasks
      std::cout << "---Thread---" << std::this_thread::get_id()
              << ": device=" << device_idx << ", write_idx=" << write_idx.idx
              << ", read_idx=" << read_idx.idx << ", status=" << GetStatus()
              << ", ret = " << ret << std::endl;
      manager().Release(datas, read_idx.idx);
      read_idx.idx++;
      read_idx.idx %= kQueueCapacity;
    }
    ReleaseResource();
    std::stringstream msg;
    msg << __func__ << ":" << __FILE__ << ":" << __LINE__;
    TORCH_CHECK(0, msg.str());
  }

  manager().Release(datas, read_idx.idx);
  __sync_synchronize();

  read_idx.idx++;
  read_idx.idx %= kQueueCapacity;
  QUEUE_DEBUG("read success, now read of repo is %d", read_idx.idx);

  return true;
}

void Repository::Enqueue(void* cur_paras) {
  if (initialized == false) {
    NPU_LOGE("Task queue is not initialized, shouldn't call Enqueue(). !!");
    return;
  }
  if (GetStatus() != RUN && GetStatus() != INIT) {
    NPU_LOGE("Task queue thread is exit, cann't call Enqueue(). !!");
    return;
  }
  bool ret = false;
  ssize_t s;
  uint64_t u = 1;

  DisableInterrupt(RepoRole::WRITER);
  while (ret == false) {
    ret = WriteQueue(cur_paras);
    if (ret == false) {
      EnableInterrupt(RepoRole::WRITER);
      __sync_synchronize();
      if (IsFullQueue()) {
        // double check the current thread hold a Gil lock
        if (PyGILState_Check()) {
          Py_BEGIN_ALLOW_THREADS s = eventfd_read(efd_write, &u);
          Py_END_ALLOW_THREADS
        } else {
          s = eventfd_read(efd_write, &u);
        }
        if (s != 0) {
          NPU_LOGE("waiting queue not full failed !!");
          return;
        }
        DisableInterrupt(RepoRole::WRITER);
        QUEUE_DEBUG("waiting ok, queue isn't full now");
      }
      continue;
    }
    __sync_synchronize();
    if (NeedNotify(RepoRole::READER)) {
      QUEUE_DEBUG("need notify consumer");
      s = eventfd_write(efd_read, u);
      if (s != 0) {
        NPU_LOGE("notify consumer failed !!");
        return;
      }
    }
  }
  EnableInterrupt(RepoRole::WRITER);
}

void Repository::Dequeue() {
  if (initialized == false) {
    NPU_LOGE("Task queue is not initialized, shouldn't call Dequeue(). !!");
    return;
  }

  bool ret = false;
  bool notify_empty = false;
  ssize_t s;
  uint64_t u = 1;

  DisableInterrupt(RepoRole::READER);
  while (ret == false && GetStatus() != RepoStatus::CAN_EXIT) {
    ret = ReadQueue();
    if (ret == false) {
      if (GetStatus() == RepoStatus::NEED_EXIT) {
        ChangeStatus(NEED_EXIT, CAN_EXIT);
        break;
      }
      EnableInterrupt(RepoRole::READER);
      __sync_synchronize();
      if (IsEmptyQueue()) {
        s = eventfd_read(efd_read, &u);
        if (s != 0) {
          NPU_LOGE("waiting queue not empty failed !!");
          return;
        }
        DisableInterrupt(RepoRole::READER);
        QUEUE_DEBUG("waiting ok, queue isn't empty now");
      }
      continue;
    }
    __sync_synchronize();
    notify_empty = need_empty &&
        IsEmptyQueue(); // need_empty && (ret == false || IsEmptyQueue());
    if (notify_empty) {
      QUEUE_DEBUG("need notify make_sure");
      s = eventfd_write(efd_empty, u);
      if (s != 0) {
        NPU_LOGE("notify make_sure failed !!");
        return;
      }
    }
    __sync_synchronize();
    if (NeedNotify(RepoRole::WRITER)) {
      QUEUE_DEBUG("need notify producer");
      s = eventfd_write(efd_write, u);
      if (s != 0) {
        NPU_LOGE("notify producer failed !!");
        return;
      }
    }
  }
  EnableInterrupt(RepoRole::READER);
}

void Repository::ReleaseResource() {
  manager().DeInit(datas);
  if (efd_read > 0) {
    close(efd_read);
    efd_read = -1;
  }
  if (efd_write > 0) {
    close(efd_write);
    efd_write = -1;
  }
  if (efd_empty > 0) {
    close(efd_empty);
    efd_empty = -1;
  }
}

Repository::~Repository() {
  if (initialized) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    QUEUE_COUT(
        "%ds %dms %dus <-- device %d FinishRepo start.",
        (int)(tv.tv_sec),
        (int)(tv.tv_usec / 1000),
        (int)(tv.tv_usec % 1000),
        (int)device_idx);
    if (consumer.joinable()) {
      SetStatus(NEED_EXIT);
      (void)eventfd_write(efd_read, 1); // escape wait
      QUEUE_DEBUG("acl escape wait.");
      consumer.join();
      QUEUE_DEBUG("acl end, now we destruct.");
    }
    gettimeofday(&tv, NULL);
    QUEUE_COUT(
        "%ds %dms %dus <-- device %d FinishRepo start.",
        (int)(tv.tv_sec),
        (int)(tv.tv_usec / 1000),
        (int)(tv.tv_usec % 1000),
        (int)device_idx);
    eventfd_write(efd_empty, 1);
    ReleaseResource();
  }
}

bool Repository::IsEmptyQueue() const {
  return read_idx.idx == write_idx.idx;
}

bool Repository::IsFullQueue() const {
  return ((write_idx.idx + 1) % kQueueCapacity) == read_idx.idx;
}

bool Repository::CheckInit() const {
  return initialized;
}

void StartConsume(Repository* repo, DeviceIndex device_id) {
  if (prctl(PR_SET_NAME, ("ACL_thread")) != 0) {
    std::cout << "set thread name failed!" << std::endl;
  }

  aclError ret = aclrtSetDevice(device_id);
  if (ret != 0) {
    std::cout << "***Thread*" << std::this_thread::get_id() << ": set device ("
              << device_id << "): ret = " << ret << std::endl;
  }

  while (repo->GetStatus() != RepoStatus::CAN_EXIT) {
    repo->Dequeue();
  }
  return;
}

void Repository::InitRepo(DeviceIndex device_id, aclrtStream calcu_stream) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  QUEUE_COUT(
      "%ds %dms %dus <--InitRepo start.",
      (int)(tv.tv_sec),
      (int)(tv.tv_usec / 1000),
      (int)(tv.tv_usec % 1000));

  if (datas == nullptr) {
    datas = manager().Init(kQueueCapacity);
  }
  if (calcu_stream == nullptr) {
    NPU_LOGE("stream should not be null when init task queue.");
    return;
  }
  calcu_stream_ = calcu_stream;
  efd_read = eventfd(0, 0);
  efd_write = eventfd(0, 0);
  efd_empty = eventfd(0, 0);

  initialized = true;
  SetStatus(INIT);
  device_idx = device_id;
  std::thread cur_consumer(StartConsume, this, device_id);
  consumer = std::move(cur_consumer);
}


} // namespace npu
} // namespace c10