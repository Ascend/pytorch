#ifndef __KERNEL_KERNEL_LIB_H__
#define __KERNEL_KERNEL_LIB_H__

#include <string>
#include <functional>

#include "ops/operator.h"
#include "ir/graph.h"

namespace fxrt {
namespace ops {

// The register entry of new kernel lib.
#define DART_REGISTER_KERNEL_LIB(KERNEL_LIB_NAME, KERNEL_LIB_CLASS)                 \
  static const fxrt::ops::KernelLibRegistrar g_kernel_lib_##KERNEL_LIB_CLASS##_reg( \
      KERNEL_LIB_NAME, []() { return new (std::nothrow) KERNEL_LIB_CLASS(); });

class KernelLib;
using KernelLibCreator = std::function<KernelLib*()>;

class DA_API KernelLib {
 public:
  explicit KernelLib(const std::string&& name) : name_(std::move(name)) {}
  virtual ~KernelLib() = default;

  virtual DAKernel* CreateKernel(ir::NodePtr node) const = 0;
  std::string Name() const {
    return name_;
  }

 protected:
  std::string name_;
};

class DA_API KernelLibRegistry {
 public:
  static KernelLibRegistry& Instance();

  void Register(const std::string& name, const KernelLibCreator&& creator);
  void Load(const std::string& path);
  const KernelLib* Get(const std::string& name);

 private:
  KernelLibRegistry() = default;
  ~KernelLibRegistry();

 private:
  std::unordered_map<std::string, const KernelLib*> kernelLibs_;
  std::unordered_map<std::string, const KernelLibCreator> kernelLibCreators_;
  std::unordered_map<std::string, void*> kernelLibHandles_;
};

class DA_API KernelLibRegistrar {
 public:
  KernelLibRegistrar(const std::string& name, const KernelLibCreator&& creator) {
    KernelLibRegistry::Instance().Register(name, std::move(creator));
  }
  ~KernelLibRegistrar() = default;
};

} // namespace ops
} // namespace fxrt
#endif // __KERNEL_KERNEL_LIB_H__
