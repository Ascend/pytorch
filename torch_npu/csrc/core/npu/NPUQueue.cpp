#include "torch_npu/csrc/core/npu/NPUQueue.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/npu_log.h"

#include <Python.h>

#include <sys/eventfd.h>
#include <sys/prctl.h>
#include <third_party/acl/inc/acl/acl_rt.h>

#ifdef OPEN_QUEUE_DEBUG
#define QUEUE_DEBUG(fmt, ...)                                      \
  do {                                                             \
    printf("[%s:%d]" fmt "\n", __func__, __LINE__, ##__VA_ARGS__); \
  } while (0)
#else
#define QUEUE_DEBUG(fmt, ...)
#endif

#ifdef OPEN_QUEUE_COUT
#define QUEUE_COUT(fmt, ...)                                       \
  do {                                                             \
    printf("[%s:%d]" fmt "\n", __func__, __LINE__, ##__VA_ARGS__); \
  } while (0)
#else
#define QUEUE_COUT(fmt, ...)
#endif

namespace c10_npu {

constexpr int32_t MAX_JUDGE_NUM = 1000000;
struct timeval delay = {0, 1};

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

  void SetCopyReleaseParam(const ACL_COPY_RELEASE_PARM_FUNC& func) {
    this->copyReleaseParamFunc = func;
  }

  void SetReleaseParam(const ACL_RELEASE_PARAM_FUNC& func) {
    this->releaseParamFunc = func;
  }

  void SetNew(const ACL_NEW_FUNC& func) {
    this->newFunc = func;
  }

  void SetDelete(const ACL_DELETE_FUNC& func) {
    this->deleteFunc = func;
  }

  int Call(void* head, int offset, uint32_t queueLen) {
    TORCH_CHECK(this->execFunc, "Failed to find execution function.");
    auto dstPtr = (uint8_t*)head + sizePerParams * offset;
    return this->execFunc(dstPtr, queueLen);
  }

  void Copy(void* dstHead, int offset, void* src, uint32_t queueLen) {
    TORCH_CHECK(this->copyFunc, "Failed to find copy function.");
    auto dstPtr = (uint8_t*)dstHead + sizePerParams * offset;
    return this->copyFunc(dstPtr, src, queueLen);
  }

  void Release(void* head, int offset, ReleaseQueue& releaseQueue) {
    TORCH_CHECK(this->releaseFunc, "Failed to find release function.");
    auto ptr = (uint8_t*)head +  sizePerParams * offset;
    return this->releaseFunc(ptr, releaseQueue);
  }

  void CopyRealseParam(void* dstHead, int offset, void* src) {
    TORCH_CHECK(this->copyReleaseParamFunc, "Failed to find copy release params function.");
    auto dstPtr = (uint8_t*)dstHead + sizePerParams * offset;
    return this->copyReleaseParamFunc(dstPtr, src);
  }

  void ReleaseParam(void* head, int offset) {
    TORCH_CHECK(this->releaseParamFunc, "Failed to find release params function.");
    auto ptr = (uint8_t*)head +  sizePerParams * offset;
    return this->releaseParamFunc(ptr);
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
  ACL_COPY_RELEASE_PARM_FUNC copyReleaseParamFunc = nullptr;
  ACL_RELEASE_PARAM_FUNC releaseParamFunc = nullptr;
}; // class CallBackManager

CallBackManager& manager() {
  static CallBackManager instance;
  return instance;
}

CallBackManager& releaseManager() {
  static CallBackManager releaseinstance;
  return releaseinstance;
}
} // namespace

namespace register_queue_cb {
NPUCallBackRegisterBuilder::NPUCallBackRegisterBuilder(const ACL_EXEC_FUNC& execFunc,
    const ACL_COPY_FUNC& copyFunc, const ACL_RELEASE_FUNC& releaseFunc,
    const ACL_NEW_FUNC& newFunc, const ACL_DELETE_FUNC& deleteFunc,
    const ACL_COPY_RELEASE_PARM_FUNC& copyReleaseParamF, const ACL_RELEASE_PARAM_FUNC& releaseParamF) {
  manager().SetExec(execFunc);
  manager().SetCopy(copyFunc);
  manager().SetRelease(releaseFunc);
  manager().SetNew(newFunc);
  manager().SetDelete(deleteFunc);
  releaseManager().SetCopyReleaseParam(copyReleaseParamF);
  releaseManager().SetReleaseParam(releaseParamF);
  releaseManager().SetNew(newFunc);
  releaseManager().SetDelete(deleteFunc);
}
} // namespace register_queue_cb


// If the capacity is too large, when the queue is full,
// a large amount of device memory is occupied at the same time;
// if the capacity is too small, and the main thread is fast enough,
// it does not make full use of concurrent design capabilities.
static constexpr size_t kQueueCapacity = 4096;

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

  // While waiting for ACL thread to launch tasks,
  // the current thread should not hold GIL.
  // When the operator compilation is triggered in the ACL thread,
  // the TE module attempts to obtain the GIL.
  // If the current thread does not release the GIL, a deadlock will
  // occur.
  PyThreadState *gilState = nullptr;
  if(PyGILState_Check()) {
    gilState = PyEval_SaveThread();
  }

  if (consumer.joinable()) {
    ssize_t s;
    uint64_t u = 1;
    while (!IsEmptyQueue()) {
      std::lock_guard<std::mutex> lock(mu_empty);
      need_empty = true;
      __sync_synchronize();
      if (!IsEmptyQueue()) { // double-check, very important idea
        s = eventfd_read(efd_empty, &u);
        if (s != 0) {
          if (errno == EINTR) {
            QUEUE_DEBUG("EINTR occurs on the eventfd_read");
            continue;
          }
          NPU_LOGE("eventfd_read failed. s=%zd, errno=%s.", s, strerror(errno));
          // Get the GIL
          if(gilState) {
            PyEval_RestoreThread(gilState);
          }
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

  // Get the GIL
  if(gilState) {
    PyEval_RestoreThread(gilState);
  }

  return SUCCESS;
}

bool Repository::WriteQueue(void* cur_paras) {
  std::lock_guard<std::mutex> lock(mu_enqueue);
  QUEUE_DEBUG("write_idx=%d, read_idx=%d", write_idx.idx, read_idx.idx);
  if (IsFullQueue()) {
    QUEUE_DEBUG("queue is full");
    return false;
  }

  uint32_t queueLen = (write_idx.idx - read_idx.idx + kQueueCapacity) % kQueueCapacity;
  __sync_synchronize();
  manager().Copy(datas, write_idx.idx, cur_paras, queueLen);
  __sync_synchronize();

  write_idx.idx = (write_idx.idx + 1) % kQueueCapacity;
  return true;
}

bool Repository::ReadQueue() {
  QUEUE_DEBUG("write_idx=%d, read_idx=%d", write_idx.idx, read_idx.idx);
  if (IsEmptyQueue()) {
    QUEUE_DEBUG("queue is empty");
    return false;
  }

  __sync_synchronize();
  uint32_t queueLen = (write_idx.idx - read_idx.idx + kQueueCapacity) % kQueueCapacity;
  if (queueLen == 1) {
    usleep(2);
  }
  auto ret = manager().Call(datas, read_idx.idx, queueLen);

  if (ret != 0) {
    ASCEND_LOGE("---Thread---%llu: device = %d, write_idx = %d, read_idx = %d, status = %d, ret = %d",
                std::this_thread::get_id(), device_idx, write_idx.idx, read_idx.idx, GetStatus(), ret);
    while (!IsEmptyQueue()) { // ignore other tasks
      manager().Release(datas, read_idx.idx, releaseQueue);
      read_idx.idx = (read_idx.idx + 1) % kQueueCapacity;
    }
    ReleaseResource();
    std::stringstream msg;
    msg << __func__ << ":" << __FILE__ << ":" << __LINE__;
    TORCH_CHECK(0, msg.str());
  }

  manager().Release(datas, read_idx.idx, releaseQueue);
  __sync_synchronize();

  read_idx.idx = (read_idx.idx + 1) % kQueueCapacity;
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

  SetWriteWorking(true);
  while (ret == false) {
    ret = WriteQueue(cur_paras);
    if (ret == false) {
      SetWriteWorking(false);
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
          if (errno == EINTR) {
            QUEUE_DEBUG("EINTR occurs on the eventfd_read");
            continue;
          }
          NPU_LOGE("waiting dequeue failed. s=%zd, errno=%s.", s, strerror(errno));
          return;
        }
        SetWriteWorking(true);
        QUEUE_DEBUG("waiting ok, queue isn't full now");
      }
      continue;
    }
    __sync_synchronize();
    while (!IsReadWorking()) {
      QUEUE_DEBUG("need notify consumer");
      s = eventfd_write(efd_read, u);
      if (s != 0) {
        if (errno == EINTR) {
          QUEUE_DEBUG("EINTR occurs on the eventfd_write");
          continue;
        }
        NPU_LOGE("notify consumer failed!! s=%zd, errno=%s", s, strerror(errno));
        return;
      }
      break;
    }
  }
  SetWriteWorking(false);
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

  SetReadWorking(true);
  while (ret == false && GetStatus() != RepoStatus::CAN_EXIT) {
    ret = ReadQueue();
    if (ret == false) {
      if (GetStatus() == RepoStatus::NEED_EXIT) {
        ChangeStatus(NEED_EXIT, CAN_EXIT);
        break;
      }

      SetReadWorking(false);
      __sync_synchronize();
      if (IsEmptyQueue()) {
        s = eventfd_read(efd_read, &u);
        if (s != 0) {
          if (errno == EINTR) {
            QUEUE_DEBUG("EINTR occurs on the eventfd_read");
            continue;
          }
          NPU_LOGE("waiting enqueue failed. s=%zd, errno=%s.", s, strerror(errno));
          return;
        }
        SetReadWorking(true);
        QUEUE_DEBUG("waiting ok, queue isn't empty now");
      }
      continue;
    }
    __sync_synchronize();
    notify_empty = need_empty &&
        IsEmptyQueue(); // need_empty && (ret == false || IsEmptyQueue());
    while (notify_empty) {
      QUEUE_DEBUG("need notify make_sure");
      s = eventfd_write(efd_empty, u);
      if (s != 0) {
        if (errno == EINTR) {
          QUEUE_DEBUG("EINTR occurs on the eventfd_write");
          continue;
        }
        NPU_LOGE("notify make_sure failed. s=%zd, errno=%s.", s, strerror(errno));
        return;
      }
      break;
    }
    __sync_synchronize();
    while (!IsWriteWorking()) {
      QUEUE_DEBUG("need notify producer");
      s = eventfd_write(efd_write, u);
      if (s != 0) {
        if (errno == EINTR) {
          QUEUE_DEBUG("EINTR occurs on the eventfd_write");
          continue;
        }
        NPU_LOGE("notify producer failed. s=%zd, errno=%s.", s, strerror(errno));
        return;
      }
      break;
    }
  }
  SetReadWorking(false);
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

bool Repository::IsFullQueue() const {
  return ((write_idx.idx + 1) % kQueueCapacity) == read_idx.idx;
}

bool Repository::CheckInit() const {
  return initialized;
}

void StartConsume(Repository* repo, c10::DeviceIndex device_id) {
  if (prctl(PR_SET_NAME, ("ACL_thread")) != 0) {
    std::cout << "set thread name failed!" << std::endl;
  }

  aclError ret = aclrtSetDevice(device_id);
  if (ret != 0) {
    C10_NPU_SHOW_ERR_MSG();
    std::cout << "***Thread*" << std::this_thread::get_id() << ": set device ("
              << device_id << "): ret = " << ret << std::endl;
  }

  while (repo->GetStatus() != RepoStatus::CAN_EXIT) {
    repo->Dequeue();
  }
  return;
}

void Repository::InitRepo(c10::DeviceIndex device_id) {
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

  efd_read = eventfd(0, 0);
  efd_write = eventfd(0, 0);
  efd_empty = eventfd(0, 0);

  initialized = true;
  SetStatus(INIT);
  device_idx = device_id;
  std::thread cur_consumer(StartConsume, this, device_id);
  consumer = std::move(cur_consumer);

  releaseQueue.InitReleaseQueue();
}

static constexpr size_t kReleaseQueueCapacity = 8192;
bool ReleaseQueue::WriteToReleaseQueue(void* cur_paras)
{
  if (IsFullQueue()) {
    QUEUE_DEBUG("Release queue is full");
    return false;
  }

  __sync_synchronize();
  releaseManager().CopyRealseParam(datas, write_idx.idx, cur_paras);

  __sync_synchronize();
  write_idx.idx = (write_idx.idx + 1) % kReleaseQueueCapacity;
  return true;
}

void ReleaseQueue::PushToReleaseQueue(void* cur_paras) {
  if (initialized == false) {
    NPU_LOGE("Release queue is not initialized, shouldn't call PushToReleaseQueue(). !!");
    return;
  }

  bool ret = false;
  while (ret == false) {
    ret = WriteToReleaseQueue(cur_paras);
    if (ret == true) {
      break;
    }
  }
}

bool ReleaseQueue::ReadFromReleaseQueue() {
  if (IsEmptyQueue()) {
    QUEUE_DEBUG("Release queue is empty");
    return false;
  }

  __sync_synchronize();
  releaseManager().ReleaseParam(datas, read_idx.idx);

  __sync_synchronize();
  read_idx.idx = (read_idx.idx + 1) % kReleaseQueueCapacity;

  return true;
}

void ReleaseQueue::PopFromReleaseQueue() {
  if (initialized == false) {
    NPU_LOGE("Release queue is not initialized, shouldn't call PopFromReleaseQueue(). !!");
    return;
  }

  bool ret = false;
  while ((ret == false) && (GetStatus() != RepoStatus::CAN_EXIT)) {
    ret = ReadFromReleaseQueue();
    if (ret == false) {
      if (GetStatus() == RepoStatus::NEED_EXIT) {
        ChangeStatus(NEED_EXIT, CAN_EXIT);
        break;
      }
      delay.tv_usec = 1;
      select(0, nullptr, nullptr, nullptr, &delay);
    }
  }
}

void StartRelease(ReleaseQueue* releaseQue) {
  if (prctl(PR_SET_NAME, ("Release_thread")) != 0) {
    std::cout << "set thread name failed!" << std::endl;
  }

  while (releaseQue->GetStatus() != RepoStatus::CAN_EXIT) {
    releaseQue->PopFromReleaseQueue();
  }
  return;
}

void ReleaseQueue::InitReleaseQueue() {
  if (datas == nullptr) {
    datas = releaseManager().Init(kReleaseQueueCapacity);
  }

  initialized = true;
  SetStatus(INIT);
  std::thread cur_releaser(StartRelease, this);
  releaser = std::move(cur_releaser);
}

ReleaseQueue::~ReleaseQueue() {
  if (initialized) {
    if (releaser.joinable()) {
      SetStatus(NEED_EXIT);
      releaser.join();
    }
  }
  releaseManager().DeInit(datas);
}

bool ReleaseQueue::IsFullQueue() const {
  return ((write_idx.idx + 1) % kReleaseQueueCapacity) == read_idx.idx;
}

RepoStatus ReleaseQueue::GetStatus() const {
  if (initialized == false) {
    NPU_LOGE("Release queue is not initialized, shouldn't call GetStatus(). !!");
  }

  return repo_status.load();
}

void ReleaseQueue::SetStatus(RepoStatus desired) {
  if (initialized == false) {
    NPU_LOGE("Release queue is not initialized, shouldn't call SetStatus(). !!");
    return;
  }

  repo_status = desired;
}

void ReleaseQueue::ChangeStatus(RepoStatus expected, RepoStatus desired) {
  if (initialized == false) {
    NPU_LOGE("Release queue is not initialized, shouldn't call ChangeStatus(). !!");
    return;
  }

  repo_status.compare_exchange_strong(expected, desired);
}
} // namespace c10_npu
