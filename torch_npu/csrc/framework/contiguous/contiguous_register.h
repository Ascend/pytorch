#ifndef __PLUGIN_NATIVE_NPU_CONTIGUOUS_CONTIGUOUS_REGISTER__
#define __PLUGIN_NATIVE_NPU_CONTIGUOUS_CONTIGUOUS_REGISTER__

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

#include <map>
#include <mutex>
#include <string>

#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/framework/contiguous/ContiguousOpt.h"
#include "torch_npu/csrc/framework/contiguous/ContiguousUtils.h"

namespace at_npu {
namespace native {

class ContiguousOpt {
public:
  ContiguousOpt() {}
  virtual ~ContiguousOpt() = default;
  virtual bool Optimizer(at::Tensor &self, const at::Tensor &src,
                         const ContiguousTensorDesc &src_desc) = 0;
  virtual bool CanOptimizer(const ContiguousTensorDesc &src_desc) {
    return false;
  }
    virtual bool CachedOptimizer(at::Tensor &self, const at::Tensor &src,
                                 const ContiguousTensorDesc &src_desc)
    {
        return Optimizer(self, src, src_desc);
    }
};

namespace register_opt {
class CopyOptRegister {
public:
  ~CopyOptRegister() = default;
  static CopyOptRegister *GetInstance() {
    static CopyOptRegister instance;
    return &instance;
  }
  void Register(std::string &name, ::std::unique_ptr<ContiguousOpt> &ptr) {
    std::lock_guard<std::mutex> lock(mu_);
    registry.emplace(name, std::move(ptr));
  }

  bool CanOptimize(std::string &name, const ContiguousTensorDesc &src_desc) {
    auto itr = registry.find(name);
    if (itr != registry.end()) {
      return itr->second->CanOptimizer(src_desc);
    }
    return false;
  }

  bool Run(const std::string &name, at::Tensor &self, const at::Tensor &src,
           const ContiguousTensorDesc &src_desc) {
    auto itr = registry.find(name);
    if (itr != registry.end()) {
      return itr->second->Optimizer(self, src, src_desc);
    }
    return false;
  }

    bool CachedRun(const std::string &name, at::Tensor &self, const at::Tensor &src,
                   const ContiguousTensorDesc &src_desc)
    {
        auto itr = registry.find(name);
        if (itr != registry.end()) {
            return itr->second->CachedOptimizer(self, src, src_desc);
        }
        return false;
    }

private:
  CopyOptRegister() {}
  mutable std::mutex mu_;
  mutable std::map<std::string, ::std::unique_ptr<ContiguousOpt>> registry;
}; // class CopyOptRegister

class CopyOptBuilder {
public:
  CopyOptBuilder(std::string name, ::std::unique_ptr<ContiguousOpt> &ptr) {
    CopyOptRegister::GetInstance()->Register(name, ptr);
  }
  ~CopyOptBuilder() = default;
}; // class CopyOptBuilder
} // namespace register_opt

#define REGISTER_COPY_OPT(name, optimization)                                  \
  REGISTER_COPY_OPT_UNIQ(name, name, optimization)
#define REGISTER_COPY_OPT_UNIQ(id, name, optimization)                         \
  auto copy_opt_##id = ::std::unique_ptr<ContiguousOpt>(new optimization());   \
  static register_opt::CopyOptBuilder register_copy_opt##id(#name,             \
                                                            copy_opt_##id);

} // namespace native
} // namespace at_npu

#endif