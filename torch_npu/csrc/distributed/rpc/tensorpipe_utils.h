#ifdef USE_RPC_FRAMEWORK

#pragma once

#include <torch/csrc/distributed/rpc/utils.h>

namespace tensorpipe_npu {
class Message;
class Allocation;
class Descriptor;
} // namespace tensorpipe_npu

namespace torch_npu {
namespace distributed {
namespace rpc {

using torch::distributed::rpc::cloneSparseTensors;
using torch::distributed::rpc::Message;
using torch::distributed::rpc::MessageType;

const c10::Stream &getStreamForDevice(const std::vector<c10::Stream> &streams, const c10::Device &device);

// Inspired by c10/core/impl/DeviceGuardImplInterface.h.

class TensorpipeDeviceTypeConverter {
public:
    // Ideally we'd want this to also return a tensorpipe_npu::Message::Tensor object
    // but we cannot forward-declare that class (because it's nested), and we
    // cannot include the TensorPipe headers because it's a private dependency.
    // Thus we bend over backwards and entrust this method with appending that
    // object to the `tensors` field of the tensorpipe_npu::Message object we pass.
    virtual c10::optional<std::vector<char>> prepareTensorForSending(const c10::Storage &storage,
                                                                     const std::vector<c10::Stream> &streams,
                                                                     tensorpipe_npu::Message &message) const = 0;

    // Same as above: this method cannot return a tensorpipe_npu::Allocation::Tensor,
    // thus it appends it to the `tensors` field of the tensorpipe_npu::Allocation.
    virtual at::DataPtr allocateTensorForReceiving(int deviceIndex, size_t length,
                                                   const std::vector<c10::Stream> &streams,
                                                   tensorpipe_npu::Allocation &allocation) const = 0;

    virtual ~TensorpipeDeviceTypeConverter() = default;
};

extern std::array<std::atomic<const TensorpipeDeviceTypeConverter *>,
                  static_cast<size_t>(c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)>
    device_type_converter_registry;

class TensorpipeDeviceTypeConverterRegistrar {
public:
    TensorpipeDeviceTypeConverterRegistrar(c10::DeviceType, const TensorpipeDeviceTypeConverter *);
};

#define C10_REGISTER_TENSORPIPE_DEVICE_TYPE_CONVERTER(DevType, TensorpipeDeviceTypeConverter)            \
    static ::torch_npu::distributed::rpc::TensorpipeDeviceTypeConverterRegistrar C10_ANONYMOUS_VARIABLE( \
        g_##DeviceType)(::c10::DeviceType::DevType, new TensorpipeDeviceTypeConverter());

inline const TensorpipeDeviceTypeConverter *getDeviceTypeConverter(c10::DeviceType type)
{
    return device_type_converter_registry[static_cast<size_t>(type)].load();
}

// A struct that holds pointers that keep alive all the memory that will be
// accessed by TensorPipe during a write operation.
struct TensorpipeWriteBuffers {
    // Allocate on heap so pointers stay valid as we move the holder.
    std::unique_ptr<MessageType> type;
    std::unique_ptr<int64_t> id;
    std::vector<char> payload;
    std::vector<char> pickle;
    // This contains the original tensors and the clones of the sparse tensors.
    std::vector<torch::Tensor> tensors;
    // This contains the copies of the data of the tensors that didn't own their
    // memory, e.g., the ones created from torch::from_blob() with no deleter.
    std::vector<std::vector<char>> copiedTensors;
};

// A struct that holds pointers that keep alive all the memory that will be
// accessed by TensorPipe during a read operation.
struct TensorpipeReadBuffers {
    // Allocate on heap so pointers stay valid as we move the holder.
    std::unique_ptr<MessageType> type;
    std::unique_ptr<int64_t> id;
    std::vector<char> payload;
    std::vector<char> pickle;
    std::vector<c10::DataPtr> tensors;
};

// Convert an RPC message into a TensorPipe message, plus a holder to all the
// data that must be kept alive while the write is performed asynchronously.
std::tuple<tensorpipe_npu::Message, TensorpipeWriteBuffers> tensorpipeSerialize(
    c10::intrusive_ptr<Message> rpcMessage, std::vector<c10::Device> devices, const std::vector<c10::Stream> &streams);

// Allocate the buffers that will hold the incoming data. They will be managed
// by the returned holder, which must be kept alive until the asynchronous read
// has finished. Pointers to these buffers will be stored in the returned
// tensorpipe_npu::Allocation struct.
std::pair<tensorpipe_npu::Allocation, TensorpipeReadBuffers> tensorpipeAllocate(
    const tensorpipe_npu::Descriptor &tpDescriptor, const std::vector<c10::Stream> &streams);

// Convert a TensorPipe message back into an RPC message. This requires the data
// to be available and can thus only be performed once the asynchronous read has
// completed. The holder can be destroyed once this function returns.
c10::intrusive_ptr<Message> tensorpipeDeserialize(tensorpipe_npu::Descriptor &&tpDescriptor,
                                                  TensorpipeReadBuffers &&holder);

} // namespace rpc
} // namespace distributed
} // namespace torch_npu

#endif
