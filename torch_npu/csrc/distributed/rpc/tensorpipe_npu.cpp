#ifdef USE_RPC_FRAMEWORK

#include <iostream>

#include <tensorpipe/tensorpipe.h>
#include <tensorpipe/tensorpipe_npu.h>
#include <torch_npu/csrc/distributed/rpc/tensorpipe_agent.h>
#include <torch_npu/csrc/distributed/rpc/tensorpipe_utils.h>

#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/SecondaryStreamGuard.h"

namespace torch_npu {
namespace distributed {
namespace rpc {
namespace {

// Register Channel
std::unique_ptr<ChannelRegistration> makeNpuBasicChannel()
{
    auto context = tensorpipe_npu::channel::npu_basic::create(tensorpipe_npu::channel::basic::create());
    return std::make_unique<ChannelRegistration>(ChannelRegistration{std::move(context), kNpuBasicChannelPriority});
}

C10_REGISTER_CREATOR(TensorPipeChannelRegistry, npu_basic, makeNpuBasicChannel);

// Tensor Send/Recv Preparation
class TensorpipeNpuConverter : public TensorpipeDeviceTypeConverter {
public:
    c10::optional<std::vector<char>> prepareTensorForSending(const c10::Storage &storage,
                                                             const std::vector<c10::Stream> &streams,
                                                             tensorpipe_npu::Message &message) const override
    {
        auto stream = c10_npu::NPUStream(getStreamForDevice(streams, storage.device()));
        // record tensor data ptrs on TensorPipe streams, so that the tensors
        // won't be destructed before TensorPipe finishing sending them.
        c10_npu::NPUCachingAllocator::recordStream(storage.data_ptr(), stream);

        tensorpipe_npu::NPUBuffer buffer;
        buffer.ptr = static_cast<char *>(storage.mutable_data());
        buffer.stream = stream.stream();

        tensorpipe_npu::Message::Tensor tensor;
        tensor.buffer = buffer;
        tensor.length = storage.nbytes();

        message.tensors.push_back(std::move(tensor));

        return c10::nullopt;
    }

    at::DataPtr allocateTensorForReceiving(int deviceIndex, size_t length, const std::vector<c10::Stream> &streams,
                                           tensorpipe_npu::Allocation &allocation) const override
    {
        c10::Device device(c10::DeviceType::PrivateUse1, deviceIndex);
        c10_npu::NPUStream stream(getStreamForDevice(streams, device));
        // NPUCachingAllocator will call recordStream accordingly on the current
        // stream.
        c10_npu::NPUStreamGuard guard(stream);
        at::DataPtr dataPtr = c10_npu::NPUCachingAllocator::get()->allocate(length);

        tensorpipe_npu::NPUBuffer buffer;
        buffer.ptr = dataPtr.get();
        buffer.stream = stream.stream();

        tensorpipe_npu::Allocation::Tensor tensor;
        tensor.buffer = buffer;

        allocation.tensors.push_back(tensor);

        return dataPtr;
    }
};

C10_REGISTER_TENSORPIPE_DEVICE_TYPE_CONVERTER(PrivateUse1, TensorpipeNpuConverter);

} // namespace
} // namespace rpc
} // namespace distributed
} // namespace torch_npu

#endif
