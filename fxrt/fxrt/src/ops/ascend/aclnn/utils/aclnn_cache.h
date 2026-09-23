#ifndef __OPS_ASCEND_ACLNN_UTILS_ACLNN_CACHE_H__
#define __OPS_ASCEND_ACLNN_UTILS_ACLNN_CACHE_H__

#include <cstddef>
#include <vector>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <functional>
#include <utility>
#include <optional>
#include <list>

#include "common/common.h"
#include "ops/utils/utils.h"
#include "ir/value/value.h"
#include "ir/common/intrusive_ptr.h"
#include "ops/ascend/aclnn/utils/aclnn_common_meta.h"
#include "ops/ascend/aclnn/utils/aclnn_deleter.h"
#include "ops/utils/op_constants.h"

namespace fxrt {
namespace ops {
// cache process type
enum class CacheReleaseType {
  RELEASE_PARAMS, // release converted params
  RELEASE_EXECUTOR, // release executor
  RELEASE_PARAMS_AND_EXECUTOR, // release converted params and executor
};

// Base class for aclnn cache with reference counting
class CacheEntry : public ir::RefCounted {
 public:
  CacheEntry() = default;
  virtual ~CacheEntry() = default;

  virtual void Release(const CacheReleaseType& type) = 0;
  virtual void UpdateTensorAddr(size_t* irIndex, size_t* tensorIndex, size_t* relativeIndex, void* tensorAddr) = 0;
  virtual aclOpExecutor* GetExecutor() = 0;
  virtual uint64_t GetWorkspaceSize() = 0;
  virtual uint64_t GetHashId() = 0;

 private:
  DISABLE_COPY_AND_ASSIGN(CacheEntry)
};
using CacheEntryPtr = ir::IntrusivePtr<CacheEntry>;

// Manager for cache entry
class CacheEntryManager : public ir::RefCounted {
 public:
  CacheEntryManager() = default;
  ~CacheEntryManager() = default;

  void AddCacheEntry(uint64_t hashId, const CacheEntryPtr& cacheEntry) {
    CHECK_IF_NULL(cacheEntry);
    if (cacheCapacity_ == 0 || hashId == 0) {
      return;
    }
    cacheList_.emplace_front(cacheEntry);
    cacheMap_[hashId] = cacheList_.begin();
    if (cacheList_.size() > cacheCapacity_) {
      cacheMap_.erase(cacheList_.back()->GetHashId());
      cacheList_.pop_back();
    }
  }

  CacheEntryPtr GetCacheEntry(uint64_t hashId) {
    if (cacheCapacity_ == 0 || hashId == 0) {
      RT_VLOG(VL_OPS) << "Get cache entry skipped, cacheCapacity: " << cacheCapacity_ << ", hashId: " << hashId;
      return nullptr;
    }
    auto it = cacheMap_.find(hashId);
    if (it != cacheMap_.end()) {
      return *it->second;
    }
    return nullptr;
  }

 private:
  DISABLE_COPY_AND_ASSIGN(CacheEntryManager)
  inline static size_t cacheCapacity_{GetAclnnCacheCapacity()};
  std::list<CacheEntryPtr> cacheList_;
  std::unordered_map<uint64_t, std::list<CacheEntryPtr>::iterator> cacheMap_;
};
using CacheEntryManagerPtr = ir::IntrusivePtr<CacheEntryManager>;

// Update tensor address
template <typename T>
inline void UpdateAddr(const CacheEntryPtr& cacheEntry, const T& value, size_t* irIndex, size_t* tensorIndex) {
  ++(*irIndex);
}

/// Some ops treat the `IntList` parameter as a tensor internally, such as `fias`.
/// In order to make the `tensorIndex` be increased correctly when updating tensor address,
/// the `IntList` parameter is wrapped in a pair whose second element is a boolean value,
/// which indicates whether the `IntList` parameter should be treated as a tensor.
inline void UpdateAddr(
    const CacheEntryPtr& cacheEntry,
    const std::pair<std::vector<int64_t>, bool>& value,
    size_t* irIndex,
    size_t* tensorIndex) {
  ++(*irIndex);
  if (value.second) {
    ++(*tensorIndex);
  }
}

inline void UpdateAddr(
    const CacheEntryPtr& cacheEntry,
    const ir::TensorPtr& tensor,
    size_t* irIndex,
    size_t* tensorIndex) {
  if (tensor != nullptr) {
    cacheEntry->UpdateTensorAddr(irIndex, tensorIndex, nullptr, tensor->GetStorage()->Data());
  }
  ++(*irIndex);
  ++(*tensorIndex);
}

inline void UpdateAddr(
    const CacheEntryPtr& cacheEntry,
    const std::optional<ir::TensorPtr>& tensor,
    size_t* irIndex,
    size_t* tensorIndex) {
  if (tensor.has_value()) {
    UpdateAddr(cacheEntry, tensor.value(), irIndex, tensorIndex);
    return;
  }
  ++(*irIndex);
  ++(*tensorIndex);
}

inline void UpdateAddr(
    const CacheEntryPtr& cacheEntry,
    const std::vector<ir::TensorPtr>& tensorList,
    size_t* irIndex,
    size_t* tensorIndex) {
  for (size_t i = 0; i < tensorList.size(); ++i) {
    cacheEntry->UpdateTensorAddr(irIndex, nullptr, &i, tensorList[i]->GetStorage()->Data());
  }
  ++(*irIndex);
  *tensorIndex += tensorList.empty() ? 1 : tensorList.size();
}

inline void UpdateAddr(
    const CacheEntryPtr& cacheEntry,
    const ir::TuplePtr& tuple,
    size_t* irIndex,
    size_t* tensorIndex) {
  if (tuple == nullptr || tuple->Size() == 0) {
    RT_VLOG(VL_OPS) << "tuple is empty";
    ++(*irIndex);
    return;
  }
  if ((*tuple)[kIndex0]->IsTensor()) {
    UpdateAddr(cacheEntry, tuple->ToTensorList(), irIndex, tensorIndex);
    return;
  }
  ++(*irIndex);
}

// Main entry for update tensor address
template <typename... Args>
void CallUpdateAddr(const CacheEntryPtr& cacheEntry, const Args&... args) {
  size_t irIndex = 0;
  size_t tensorIndex = 0;
  (UpdateAddr(cacheEntry, args, &irIndex, &tensorIndex), ...);
}

// Update a single tensor address
inline void UpdateAclTensorAddr(
    aclTensor* tensor,
    size_t* irIndex,
    size_t* tensorIndex,
    aclOpExecutor* executor,
    void* tensorAddr) {
  static const auto aclSetTensorAddr = GET_ACLNN_COMMON_META_FUNC(aclSetTensorAddr);
  if (aclSetTensorAddr == nullptr) {
    RT_GLOG(EXCEPTION) << "aclSetTensorAddr is nullptr";
    return;
  }
  auto ret = aclSetTensorAddr(executor, *tensorIndex, tensor, tensorAddr);
  if (ret != 0) {
    RT_GLOG(EXCEPTION) << "Call aclSetTensorAddr failed, index: " << *irIndex << ", tensorIndex: " << *tensorIndex
                       << ", tensorAddr: " << tensorAddr << ", ret: " << ret;
  }
}

// Update a tensor list address
inline void UpdateAclTensorListAddr(
    aclTensorList* tensorList,
    size_t* irIndex,
    size_t* tensorIndex,
    size_t* relativeIndex,
    aclOpExecutor* executor,
    void* tensorAddr) {
  static const auto aclSetDynamicTensorAddr = GET_ACLNN_COMMON_META_FUNC(aclSetDynamicTensorAddr);
  if (aclSetDynamicTensorAddr == nullptr) {
    RT_GLOG(EXCEPTION) << "aclSetDynamicTensorAddr is nullptr";
    return;
  }
  auto ret = aclSetDynamicTensorAddr(executor, *irIndex, *relativeIndex, tensorList, tensorAddr);
  if (ret != 0) {
    RT_GLOG(EXCEPTION) << "Call aclSetDynamicTensorAddr failed, index: " << *irIndex
                       << ", relativeIndex: " << *relativeIndex
                       << ", tensorIndex: " << (tensorIndex == nullptr ? "null" : std::to_string(*tensorIndex))
                       << ", tensorAddr: " << tensorAddr << ", ret: " << ret;
  }
}

// Cache processor for cache operations
template <typename Tuple>
class CacheProcessor {
 public:
  explicit CacheProcessor(uint64_t hashId, Tuple&& tuple, aclOpExecutor* executor, uint64_t workspaceSize)
      : hashId_(hashId), convertedParams_(std::move(tuple)), executor_(executor), workspaceSize_(workspaceSize) {
    InitTensorAddrUpdaters();
  }

  CacheProcessor(CacheProcessor&& other) noexcept
      : hashId_(other.hashId_),
        convertedParams_(std::move(other.convertedParams_)),
        executor_(other.executor_),
        workspaceSize_(other.workspaceSize_),
        isParamsReleased_(other.isParamsReleased_),
        isExecutorReleased_(other.isExecutorReleased_) {
    other.executor_ = nullptr;
    other.isParamsReleased_ = true;
    other.isExecutorReleased_ = true;
  }

  CacheProcessor& operator=(CacheProcessor&& other) noexcept {
    if (this != &other) {
      if (!isParamsReleased_) {
        ReleaseConvertedParams(convertedParams_);
      }

      hashId_ = other.hashId_;
      convertedParams_ = std::move(other.convertedParams_);
      executor_ = other.executor_;
      workspaceSize_ = other.workspaceSize_;
      isParamsReleased_ = other.isParamsReleased_;
      isExecutorReleased_ = other.isExecutorReleased_;

      other.executor_ = nullptr;
      other.isParamsReleased_ = true;
      other.isExecutorReleased_ = true;
    }
    return *this;
  }

  template <size_t I>
  static void BuildTensorAddrUpdater() {
    using elementType = std::decay_t<std::tuple_element_t<I, Tuple>>;
    if constexpr (std::is_same_v<elementType, aclTensor*>) {
      tensorAddrUpdatersMap_[I] = [](const Tuple& convertedParams,
                                     aclOpExecutor* executor,
                                     size_t* irIndex,
                                     size_t* tensorIndex,
                                     size_t* relativeIndex,
                                     void* tensorAddr) {
        UpdateAclTensorAddr(std::get<I>(convertedParams), irIndex, tensorIndex, executor, tensorAddr);
      };
    }
    if constexpr (std::is_same_v<elementType, aclTensorList*>) {
      tensorAddrUpdatersMap_[I] = [](const Tuple& convertedParams,
                                     aclOpExecutor* executor,
                                     size_t* irIndex,
                                     size_t* tensorIndex,
                                     size_t* relativeIndex,
                                     void* tensorAddr) {
        UpdateAclTensorListAddr(
            std::get<I>(convertedParams), irIndex, tensorIndex, relativeIndex, executor, tensorAddr);
      };
    }
  }

  template <size_t... I>
  static void BuildTensorAddrUpdaters(std::index_sequence<I...>) {
    (BuildTensorAddrUpdater<I>(), ...);
  }

  static void InitTensorAddrUpdaters() {
    constexpr size_t tuple_size = std::tuple_size_v<Tuple>;
    static_assert(tuple_size > 0, "Tuple size must be greater than 0");
    static bool isInitialized = false;
    if (isInitialized) {
      return;
    }
    isInitialized = true;
    RT_VLOG(VL_OPS) << "Initializing tensor address updaters for tuple of size: " << tuple_size;

    BuildTensorAddrUpdaters(std::make_index_sequence<tuple_size>{});
  }

  ~CacheProcessor() {
    // release params and executor
    if (!isParamsReleased_) {
      ReleaseConvertedParams(convertedParams_);
    }
    if (!isExecutorReleased_) {
      ReleaseExecutor(executor_);
    }
  }

  void Release(const CacheReleaseType& type) {
    switch (type) {
      case CacheReleaseType::RELEASE_PARAMS:
        if (!isParamsReleased_) {
          ReleaseConvertedParams(convertedParams_);
          isParamsReleased_ = true;
        }
        break;
      case CacheReleaseType::RELEASE_EXECUTOR:
        if (!isExecutorReleased_) {
          ReleaseExecutor(executor_);
          isExecutorReleased_ = true;
        }
        break;
      case CacheReleaseType::RELEASE_PARAMS_AND_EXECUTOR:
        if (!isParamsReleased_) {
          ReleaseConvertedParams(convertedParams_);
          isParamsReleased_ = true;
        }
        if (!isExecutorReleased_) {
          ReleaseExecutor(executor_);
          isExecutorReleased_ = true;
        }
        break;
      default:
        RT_GLOG(EXCEPTION) << "Invalid cache release type: " << static_cast<int>(type);
        break;
    }
  }

  void UpdateTensorAddr(size_t* irIndex, size_t* tensorIndex, size_t* relativeIndex, void* tensorAddr) {
    // Use the static map for efficient lookup, no need lookup in the future
    auto it = tensorAddrUpdatersMap_.find(*irIndex);
    if (it != tensorAddrUpdatersMap_.end()) {
      it->second(convertedParams_, executor_, irIndex, tensorIndex, relativeIndex, tensorAddr);
    } else {
      RT_GLOG(EXCEPTION) << "No updater found for index: " << *irIndex << ", available indices: ";
    }
  }

  aclOpExecutor* GetExecutor() {
    return executor_;
  }

  uint64_t GetWorkspaceSize() {
    return workspaceSize_;
  }

  uint64_t GetHashId() {
    return hashId_;
  }

  using TensorAddrUpdater = std::function<void(const Tuple&, aclOpExecutor*, size_t*, size_t*, size_t*, void*)>;

 private:
  DISABLE_COPY_AND_ASSIGN(CacheProcessor)
  uint64_t hashId_;
  Tuple convertedParams_;
  aclOpExecutor* executor_;
  uint64_t workspaceSize_;

  // Static map for updater functions (no instance data)
  inline static std::unordered_map<size_t, TensorAddrUpdater> tensorAddrUpdatersMap_;

  bool isParamsReleased_{false};
  bool isExecutorReleased_{false};
};

// Wrapper class for CacheEntry
template <typename CacheProcessor>
class CacheEntryImpl : public CacheEntry {
 public:
  explicit CacheEntryImpl(CacheProcessor&& cacheProcessor) : cacheProcessor_(std::move(cacheProcessor)) {}
  ~CacheEntryImpl() override = default;

  void Release(const CacheReleaseType& type) override {
    cacheProcessor_.Release(type);
  }

  void UpdateTensorAddr(size_t* irIndex, size_t* tensorIndex, size_t* relativeIndex, void* tensorAddr) override {
    cacheProcessor_.UpdateTensorAddr(irIndex, tensorIndex, relativeIndex, tensorAddr);
  }

  aclOpExecutor* GetExecutor() override {
    return cacheProcessor_.GetExecutor();
  }
  uint64_t GetWorkspaceSize() override {
    return cacheProcessor_.GetWorkspaceSize();
  }
  uint64_t GetHashId() override {
    return cacheProcessor_.GetHashId();
  }

 private:
  DISABLE_COPY_AND_ASSIGN(CacheEntryImpl)
  CacheProcessor cacheProcessor_;
};

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_UTILS_ACLNN_CACHE_H__
