
#include <ATen/detail/NPUHooksInterface.h>

#include <ATen/Generator.h>
#include <THNPU/THNPUCachingHostAllocator.h>
#include <c10/util/Optional.h>

// TODO: No need to have this whole header, we can just put it all in
// the cpp file

namespace at { namespace npu { namespace detail {

// The real implementation of NPUHooksInterface
struct NPUHooks : public at::NPUHooksInterface {
  NPUHooks(at::NPUHooksArgs) {}
  void initNPU() const override;
  const Generator& getDefaultNPUGenerator(DeviceIndex device_index = -1) const override;
  bool hasNPU() const override;
  int64_t current_device() const override;
  Allocator* getPinnedMemoryAllocator() const override;
  int getNumNPUs() const override;
};

}}} // at::npu::detail
