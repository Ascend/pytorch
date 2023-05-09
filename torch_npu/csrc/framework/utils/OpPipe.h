#ifndef __PULGIN_NATIVE_NPU_UTILS_OP_PIPE__
#define __PULGIN_NATIVE_NPU_UTILS_OP_PIPE__

#include <ATen/ATen.h>

namespace at_npu {
namespace native {

//
template<class Derived>
class OpPipe {
public:
  using PROCESS_FUNC = std::function<void(at::Tensor&)>;
  Derived& Func(const PROCESS_FUNC& func) {
    this->func = func;
    return static_cast<Derived&>(*this);
  }
protected:
  PROCESS_FUNC func = nullptr;
};

//
class OpPipeWithDefinedOut : public OpPipe<OpPipeWithDefinedOut> {
public:
  OpPipeWithDefinedOut& CheckMemory(const std::initializer_list<at::Tensor>& inputs, const std::initializer_list<at::Tensor>& outputs);
  at::Tensor& Call(at::Tensor& dst);
};

//
class OpPipeWithApplyOut : public OpPipe<OpPipeWithApplyOut> {
public:
  using PROCESS_FUNC = std::function<void(at::Tensor&)>;
  OpPipeWithApplyOut& ApplyOutputSameAs(const at::Tensor& src);
  at::Tensor& Call();
private:
  at::Tensor dst;
};

} // namespace native
} // namespace at_npu

#endif
