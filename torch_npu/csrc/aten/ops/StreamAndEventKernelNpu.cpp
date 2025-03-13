#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

void NPUNativeFunctions::record_stream(at::Tensor& self, c10::Stream stream)
{
    struct c10::StreamData3 data = stream.pack3();
    c10_npu::NPUCachingAllocator::recordStream(
        self.storage().data_ptr(),
        c10_npu::NPUStream::unpack3(
            data.stream_id, data.device_index, data.device_type));
}

} // namespace native
} // namespace at_npu
