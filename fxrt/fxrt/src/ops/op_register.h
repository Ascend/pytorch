#ifndef __OPS_OP_REGISTER_H__
#define __OPS_OP_REGISTER_H__

#include <utility>
#include <string>
#include <string_view>
#include <memory>
#include <functional>
#include <type_traits>
#include <map>
#include <unordered_map>

#include "common/logger.h"
#include "common/common.h"
#include "ops/operator.h"
#include "hardware/device.h"

namespace fxrt {
namespace ops {
inline constexpr std::string_view kUnknownOpFactory = "Unknown";
inline constexpr std::string_view kAscendOpFactory = "Ascend";
inline constexpr std::string_view kCPUOpFactory = "CPU";

struct UnknownOpFactory {};
struct AscendOpFactory {};
struct CPUOpFactory {};

template <typename T>
struct OpFactoryTraits;

template <>
struct OpFactoryTraits<UnknownOpFactory> {
  static constexpr std::string_view value = kUnknownOpFactory;
};

template <>
struct OpFactoryTraits<AscendOpFactory> {
  static constexpr std::string_view value = kAscendOpFactory;
};

template <>
struct OpFactoryTraits<CPUOpFactory> {
  static constexpr std::string_view value = kCPUOpFactory;
};

// Function for loading op libs.
DA_API bool LoadOpLib(const std::string& opLibPrefix, std::stringstream* errMsg);

class DA_API OpFactoryBase {
  using OpFactoryMapType = std::unordered_map<std::string_view, std::unique_ptr<OpFactoryBase>>;

 public:
  OpFactoryBase() = default;
  virtual ~OpFactoryBase() = default;

 protected:
  static OpFactoryBase* GetOpFactory(const std::string_view& name);

  static OpFactoryBase* CreateOpFactory(const std::string_view& name, std::unique_ptr<OpFactoryBase>&& factory);

 private:
  static OpFactoryMapType& OpFactoryMap();
};

template <typename T, typename OpFactoryType = AscendOpFactory>
class OpFactory : public OpFactoryBase {
  using CreatorFunc = std::function<std::unique_ptr<T>()>;

 public:
  OpFactory() = default;
  ~OpFactory() = default;
  OpFactory(const OpFactory&) = delete;
  void operator=(const OpFactory&) = delete;

  static OpFactory<T, OpFactoryType>& GetInstance() {
    auto factoryBase = OpFactoryBase::GetOpFactory(OpFactoryTraits<OpFactoryType>::value);
    if (factoryBase == nullptr) {
      factoryBase = OpFactoryBase::CreateOpFactory(
          OpFactoryTraits<OpFactoryType>::value, std::make_unique<OpFactory<T, OpFactoryType>>());
    }
    auto* opFactory = static_cast<OpFactory<T, OpFactoryType>*>(factoryBase);
    CHECK_IF_NULL(opFactory);
    opFactory->LoadOpPlugin();
    return *opFactory;
  }

  void LoadOpPlugin() {
    if (isPluginLoaded_) {
      return;
    }
    isPluginLoaded_ = true;
    std::stringstream errMsg;
    if constexpr (std::is_same_v<OpFactoryType, AscendOpFactory>) {
      if (!LoadOpLib("libops_ascend", &errMsg)) {
        RT_GLOG(EXCEPTION) << "Load Ascend Op Lib failed, error message: " << errMsg.str();
      }
    } else {
      if constexpr (std::is_same_v<OpFactoryType, CPUOpFactory>) {
        if (!LoadOpLib("libops_cpu", &errMsg)) {
          RT_GLOG(EXCEPTION) << "Load CPU Op Lib failed, error message: " << errMsg.str();
        }
      } else if constexpr (std::is_same_v<OpFactoryType, UnknownOpFactory>) {
        RT_VLOG(VL_OPS) << "Unknown Op Factory, skip load Op Lib, error message: " << errMsg.str();
      } else {
        RT_GLOG(EXCEPTION)
            << "Got invalid OpFactoryType, only supports AscendOpFactory, CPUOpFactory and UnknownOpFactory.";
      }
    }
#ifdef ENABLE_TORCH_FRONT
    errMsg.str("");
    if (!LoadOpLib("libops_torch", &errMsg)) {
      RT_GLOG(EXCEPTION) << "Load torch Op Lib failed, error message: " << errMsg.str();
    }
#endif
  }

  void Register(const std::string& opName, CreatorFunc&& creator) {
    if (IsRegistered(opName)) {
      RT_GLOG(EXCEPTION) << "Repeat register for op " << opName;
    }
    (void)opCreatorsMap_.emplace(opName, std::move(creator));
  }

  void UnRegister(const std::string& opName) {
    auto iter = opCreatorsMap_.find(opName);
    if (iter != opCreatorsMap_.end()) {
      opCreatorsMap_.erase(iter);
    }
  }

  bool IsRegistered(const std::string& opName) const {
    return opCreatorsMap_.find(opName) != opCreatorsMap_.end();
  }

  std::unique_ptr<T> Create(const std::string& opName) const {
    typename std::unordered_map<std::string, CreatorFunc>::const_iterator iter = opCreatorsMap_.find(opName);
    if (iter != opCreatorsMap_.cend()) {
      return (iter->second)();
    }
    return nullptr;
  }

 private:
  std::unordered_map<std::string, CreatorFunc> opCreatorsMap_;
  bool isPluginLoaded_ = false;
};

template <typename T, typename OpFactoryType = AscendOpFactory>
class OpRegistrar {
 public:
  explicit OpRegistrar(const std::string&& opName, std::function<std::unique_ptr<T>()>&& creator)
      : opName_(std::move(opName)) {
    OpFactory<T, OpFactoryType>::GetInstance().Register(opName_, std::move(creator));
  }
  ~OpRegistrar() {
    OpFactory<T, OpFactoryType>::GetInstance().UnRegister(opName_);
  }

 private:
  std::string opName_;
};

#define FXRT_REG_OP(OP_NAME, OP_CLASS, DEVICE_NAME)                                                                 \
  static_assert(std::is_base_of<ops::Operator, OP_CLASS>::value, #OP_CLASS " must be derived from class Operator"); \
  static const ops::OpRegistrar<ops::Operator, ops::DEVICE_NAME##OpFactory>                                         \
      g_##OP_NAME##_##OP_CLASS##_##DEVICE_NAME##_reg(#OP_NAME, []() { return std::make_unique<OP_CLASS>(); })

#define FXRT_REG_OP_WITH_CREATOR(OP_NAME, OP_CLASS, DEVICE_NAME, CREATOR)                                           \
  static_assert(std::is_base_of<ops::Operator, OP_CLASS>::value, #OP_CLASS " must be derived from class Operator"); \
  static const ops::OpRegistrar<ops::Operator, ops::DEVICE_NAME##OpFactory>                                         \
      g_##OP_NAME##_##OP_CLASS##_##DEVICE_NAME##_reg(#OP_NAME, CREATOR)

// Op prototype registry: the number of fixed tensor inputs declared by the op schema.
// Consumed by runtime::CheckOpSupport (ops/utils/op_support.cpp) to validate actual
// input count against the op prototype and drive the custom_call fallback.
// The registration site (next to FXRT_REG_OP in each op file) is the single source
// of truth for the op prototype; there is no external schema file any more.
class DA_API OpPrototypeRegistry {
 public:
  static void Register(const std::string& opName, size_t tensorInputCount);
  static const size_t* GetTensorInputCount(const std::string& opName);

 private:
  static std::unordered_map<std::string, size_t>& PrototypeMap();
};

#define FXRT_REG_OP_PROTOTYPE(OP_NAME, TENSOR_INPUTS)                    \
  static const bool g_##OP_NAME##_prototype_reg = []() {                 \
    ::fxrt::ops::OpPrototypeRegistry::Register(#OP_NAME, TENSOR_INPUTS); \
    return true;                                                         \
  }()

inline std::unique_ptr<Operator> CreateOperator(const std::string& name, const hardware::DeviceType type) {
  if (type == hardware::DeviceType::NPU) {
    return OpFactory<Operator>::GetInstance().Create(name);
  } else if (type == hardware::DeviceType::CPU) {
    return OpFactory<Operator, CPUOpFactory>::GetInstance().Create(name);
  } else {
    RT_GLOG(EXCEPTION) << "Got invalid device type, only supports CPU and NPU.";
    return nullptr;
  }
}
} // namespace ops
} // namespace fxrt
#endif // __OPS_OP_REGISTER_H__
