#include "torch_npu/csrc/npu/DataParallelComm.h"

#include <unordered_map>
#include <memory>
#include <bitset>
#include <thread>
#include <ATen/ATen.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/functional.h>
#include <c10/util/irange.h>
#include <c10/util/Optional.h>
#include <c10/util/ArrayRef.h>
#include <torch/csrc/utils/tensor_flatten.h>
#include <torch/csrc/autograd/variable.h>

#include "torch_npu/csrc/distributed/HCCLUtils.hpp"
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace torch_npu {
using namespace at;

namespace data_parallel {

template <typename T>
struct GetSecondArgType;

template <typename R, typename Arg0, typename Arg1, typename... Args>
struct GetSecondArgType<R(Arg0, Arg1, Args...)> {
    typedef typename std::decay<Arg1>::type type;
};

constexpr auto count_max = std::numeric_limits<GetSecondArgType<decltype(HcclBroadcast)>::type>::max();

inline size_t get_max_count()
{
    return count_max;
}

struct HcclCommList {
    std::unique_ptr<HcclComm[]> comms;
    int ndevices;
    explicit HcclCommList(const std::vector<int>& devices): comms(new HcclComm[devices.size()]), ndevices(devices.size())
    {
        HCCL_CHECK_ERROR(HcclCommInitAll(devices.size(), const_cast<int32_t*>(devices.data()), comms.get()));
    }
    HcclCommList(HcclCommList&& foo) = default;
    HcclCommList& operator=(HcclCommList&& foo) = default;
    ~HcclCommList() = default;
    void Release()
    {
        if (comms) {
            for (const auto i : c10::irange(ndevices)) {
                int dummy_var;
                auto ret = c10_npu::GetDevice(&dummy_var);
                if (ret != ACL_ERROR_NONE) {
                    /* there are cases when this destructor is called after the
                    NPU driver is already unloaded from the process.
                    In these cases, skip hcclCommDestroy */
                    return;
                }
                HcclCommDestroy(comms[i]);
            }
        }
    }
    ArrayRef<HcclComm> ref() const
    {
        return ArrayRef<HcclComm>(comms.get(), ndevices);
    }
};

using device_set = std::bitset<C10_COMPILE_TIME_MAX_NPUS>;

using device_list = std::vector<int>;
static std::unordered_map<device_list, HcclCommList, c10::hash<device_list>> _communicators;

using comm_list = std::vector<HcclComm>;
using stream_list = std::vector<c10::optional<c10_npu::NPUStream>>;

void ReleaseHcclCommList()
{
    for (auto& [_, comm] : _communicators) {
        comm.Release();
    }
}

ArrayRef<HcclComm> get_communicators(TensorList inputs)
{
    static auto get_device = [](const at::Tensor& t) -> int {return t.get_device();};
    device_list devices = c10::fmap(inputs, get_device);
    auto it = _communicators.find(devices);
    if (it == _communicators.end()) {
        std::tie(it, std::ignore) = _communicators.emplace(devices, devices);
    }
    return it->second.ref();
}

static inline void check_tensor(const at::Tensor& input, const at::optional<at::Tensor>& output,
    int input_multiplier, int output_multiplier, int64_t ref_numel, ScalarType ref_dtype)
{
    auto check_one = [&](const at::Tensor& tensor) {
        if (!torch_npu::utils::is_npu(tensor) || tensor.is_sparse()) {
            throw std::runtime_error("input and output elements have to be npu dense Tensors" + PTA_ERROR(ErrCode::TYPE));
        }

        if (ref_dtype != tensor.scalar_type()) {
            throw std::runtime_error("all inputs and outputs must be of the same Tensor dtype" + PTA_ERROR(ErrCode::TYPE));
        }

        if (!tensor.is_contiguous()) {
            throw std::runtime_error("all inputs and outputs have to be contiguous" + PTA_ERROR(ErrCode::TYPE));
        }
    };

    check_one(input);
    // all inputs must be same size
    if (input.numel() != ref_numel) {
        throw std::runtime_error("all inputs must have the same number of elements" + PTA_ERROR(ErrCode::PARAM));
    }

    if (output) {
        check_one(*output);
        // inputs and outputs must be on same device respectively
        if (input.get_device() != output->get_device()) {
            throw std::runtime_error("input and output must be on the same device" + PTA_ERROR(ErrCode::PARAM));
        }

        if (output->numel() * output_multiplier != ref_numel * input_multiplier) {
            throw std::runtime_error("output must be of size input_size * size_multiplier" + PTA_ERROR(ErrCode::PARAM));
        }
    }
}

void check_inputs(TensorList inputs, TensorList outputs, int input_multiplier, int output_multiplier)
{
    // need to check len(inputs) == len(outputs)
    size_t len = inputs.size();
    if (len <= 0) {
        throw std::runtime_error("input sequence can't be empty" + PTA_ERROR(ErrCode::PARAM));
    }

    if (len != outputs.size()) {
        std::stringstream err;
        err << "inputs and outputs sequences have to be of the same length, but got input of length "
            << len << " and output of length " << outputs.size() << PTA_ERROR(ErrCode::PARAM);
        throw std::runtime_error(err.str());
    }

    device_set devices;
    int64_t numel = inputs[0].numel();
    auto dtype = inputs[0].scalar_type();

    for (const auto i : c10::irange(len)) {
        auto input = inputs[i];
        auto output = outputs[i];

        check_tensor(input, output, input_multiplier, output_multiplier, numel, dtype);

        auto input_device = input.get_device();
        // inputs must be on unique devices
        if (devices.test(input_device)) {
            throw std::runtime_error("inputs must be on unique devices" + PTA_ERROR(ErrCode::TYPE));
        }
        devices.set(input_device);
    }
}

struct unique_type_checker {
    void show(size_t type_id)
    {
        if (!unique) {
            return;
        }
        if (!type_id_) {
            type_id_ = type_id;
        }

        unique = type_id_.value() == type_id;
    }

    c10::optional<size_t> type_id_;
    bool unique = true;
};

bool is_available(TensorList tensors)
{
    device_set devices;
    for (auto& tensor : tensors) {
        if (!torch_npu::utils::is_npu(tensor) || tensor.is_sparse()) {
            return false;
        }
        if (!tensor.is_contiguous()) {
            return false;
        }
        auto device = tensor.get_device();
        if (devices[device]) {
            return false;
        }
        devices[device] = true;
    }
    return true;
}

// Broadcast
void broadcast(TensorList tensors, const stream_list& streams = {}, const comm_list& user_comms = {})
{
    check_inputs(tensors, tensors, 1, 1);
    auto data_type = c10d_npu::getHcclDataType(tensors[0].scalar_type());
    int64_t numel = tensors[0].numel();

    const auto comms = user_comms.empty() ? get_communicators(tensors) : ArrayRef<HcclComm>(user_comms);
    c10_npu::OptionalNPUGuard device_guard;
    std::vector<std::thread> threads;
    for (size_t i = 0, num_tensors = tensors.size(); i < num_tensors; i++) {
        int device = tensors[i].get_device();
        threads.emplace_back([&, i, device]() {
            NPU_CHECK_ERROR(c10_npu::SetDevice(device));
            // Default to the current stream
            const auto stream = (streams.empty() || !streams[i]) ?
                c10_npu::getCurrentNPUStream(device).stream() : streams[i]->stream();
            TORCH_CHECK(
                static_cast<uint64_t>(numel) <= static_cast<uint64_t>(count_max),
                "Broadcast tensor has ",
                numel,
                " elements, which exceeds the maximum HCCL supports (",
                count_max,
                ")" + PTA_ERROR(ErrCode::VALUE));
            HcclComm comm = comms[i];
            HCCL_CHECK_ERROR(HcclBroadcast(tensors[i].data_ptr(), numel, data_type, 0, comm, stream));
        });
    }
    for (auto &t : threads) {
        t.join();
    }
}

static inline std::vector<Tensor>& _broadcast_out_impl(const Tensor& tensor, std::vector<Tensor>& out_tensors)
{
    std::vector<Tensor> hccl_list;
    hccl_list.reserve(out_tensors.size() + 1);
    hccl_list.emplace_back(tensor);
    for (auto& out_tensor : out_tensors) {
        hccl_list.emplace_back(out_tensor);
    }
    if (is_available(hccl_list)) {
        broadcast(hccl_list);
    } else {
        for (auto& out_tensor : out_tensors) {
            out_tensor.copy_(tensor, true);
        }
    }
    return out_tensors;
}

std::vector<Tensor>& broadcast_out(const Tensor& tensor, std::vector<Tensor>& out_tensors)
{
    for (const auto i : c10::irange(out_tensors.size())) {
        TORCH_CHECK(torch_npu::utils::is_npu(out_tensors[i]),
            "Expected all output tensors to be NPU tensors, but output tensor at index ",
            i,
            " has device '",
            out_tensors[i].device(),
            "'" + PTA_ERROR(ErrCode::TYPE));
        TORCH_CHECK(out_tensors[i].sizes() == tensor.sizes(),
            "Expected all output tensors to have same shape as the source tensor ",
            tensor.sizes(),
            ", but output tensor at index ",
            i,
            " has shape ",
            out_tensors[i].sizes(), PTA_ERROR(ErrCode::VALUE));
    }
    return _broadcast_out_impl(tensor, out_tensors);
}

std::vector<Tensor> broadcast(const Tensor& tensor, IntArrayRef devices)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<Tensor> diff_device_dst_tensors;
    diff_device_dst_tensors.reserve(devices.size());
    for (auto device : devices) {
        TORCH_CHECK(device >= 0, "Expected non-negative device index, but got ", device, PTA_ERROR(ErrCode::VALUE));
        if (device != tensor.get_device()) {
            diff_device_dst_tensors.emplace_back(
                at::empty(tensor.sizes(), tensor.options().device(at::Device(DeviceType::PrivateUse1, device))));
        }
    }
    _broadcast_out_impl(tensor, diff_device_dst_tensors);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<Tensor> dst_tensors;
    dst_tensors.reserve(devices.size());
    auto it = diff_device_dst_tensors.begin();
    for (auto device : devices) {
        // NOLINTNEXTLINE(bugprone-branch-clone)
        if (device != tensor.get_device()) {
            dst_tensors.emplace_back(*it++);
        } else {
            dst_tensors.emplace_back(tensor);
        }
    }
    TORCH_INTERNAL_ASSERT(it == diff_device_dst_tensors.end());
    return dst_tensors;
}

tensor_list2d broadcast_coalesced(TensorList tensors, IntArrayRef devices, size_t buffer_size)
{
    TORCH_CHECK(
        std::all_of(tensors.begin(), tensors.end(), [&](const at::Tensor& t) { return t.get_device() == devices[0]; }),
        "All tensors must be on devices[0]: ",
        devices[0], PTA_ERROR(ErrCode::TYPE));

    buffer_size = std::min(get_max_count(), buffer_size);

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    tensor_list2d outputs(devices.size());
    outputs[0] = tensors.vec();
    for (auto& o : outputs) {
        o.reserve(tensors.size());
    }

    unique_type_checker type_checker;
    c10_npu::NPUGuard device_guard{devices[0]};
    for (auto& chunk : torch::utils::take_tensors(tensors, buffer_size)) {
        auto type_id = chunk.type_id();
        type_checker.show(type_id);
        std::vector<at::Tensor> results;
        if (chunk.options().is_sparse()) {
            auto flat_tuple = torch::utils::flatten_sparse_tensors(chunk.tensors);
            auto broadcast_indices = broadcast(flat_tuple.first, devices);
            auto broadcast_values = broadcast(flat_tuple.second, devices);
            results.reserve(devices.size());
            for (size_t i = 1, num_devices = devices.size(); i < num_devices; ++i) {
                device_guard.set_index(devices[i]);
                auto& device_outputs = outputs[i];
                auto& inds = broadcast_indices[i];
                auto& vals = broadcast_values[i];
                for (const auto& var : torch::utils::unflatten_sparse_tensors(inds, vals, chunk.tensors)) {
                    // See NOTE [ Version Counter in comm.*_coalesced ]
                    device_outputs.emplace_back(torch::autograd::make_variable(var.tensor_data(), false));
                }
            }
        } else {
            auto results = broadcast(torch::utils::flatten_dense_tensors(chunk.tensors), devices);
            for (size_t i = 1, num_devices = devices.size(); i < num_devices; ++i) {
                device_guard.set_index(devices[i]);
                auto& device_outputs = outputs[i];
                for (auto& var : torch::utils::unflatten_dense_tensors(results[i], chunk.tensors)) {
                    // See NOTE [ Version Counter in comm.*_coalesced ]
                    device_outputs.emplace_back(torch::autograd::make_variable(var.tensor_data(), false));
                }
            }
        }
    }

    // If we only saw a single tensor type, then we can skip expensive reordering
    if (!type_checker.unique) {
        for (auto& o : outputs) {
            torch::utils::reorder_tensors_like(o, tensors);
        }
    }
    return outputs;
}

// Scatter
std::vector<at::Tensor>& scatter_out(const at::Tensor& tensor, std::vector<at::Tensor>& out_tensors, int64_t dim,
    const c10::optional<std::vector<c10::optional<c10_npu::NPUStream>>>& streams)
{
    TORCH_CHECK(!out_tensors.empty(), "Expected at least one output tensor to scatter to" + PTA_ERROR(ErrCode::VALUE));
    dim = at::maybe_wrap_dim(dim, tensor);
    int64_t total_size = 0;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<int64_t> chunk_sizes;
    chunk_sizes.reserve(out_tensors.size());
    for (const auto i : c10::irange(out_tensors.size())) {
        TORCH_CHECK(torch_npu::utils::is_npu(out_tensors[i]),
            "Expected all output tensors to be NPU tensors, but output tensor at index ",
            i,
            " has device '",
            out_tensors[i].device(),
            "'" + PTA_ERROR(ErrCode::TYPE));
        auto out_sizes = out_tensors[i].sizes().vec();
        bool same_ndim = out_sizes.size() == static_cast<size_t>(tensor.dim());
        if (same_ndim) {
            total_size += out_sizes[dim];
            chunk_sizes.emplace_back(out_sizes[dim]);
            out_sizes[dim] = tensor.size(dim);
        }
        TORCH_CHECK(same_ndim && out_sizes == tensor.sizes(),
            "Output tensor at index ",
            i,
            " has incorrect shape: ",
            out_tensors[i].sizes(),
            ". Expected same shape except for scatter dim ",
            dim,
            " as the source tensor: ",
            at::IntArrayRef(tensor.sizes()), PTA_ERROR(ErrCode::VALUE));
    }
    TORCH_CHECK(total_size == tensor.size(dim),
        "Total size for output tensors along scatter dim ",
        dim,
        " does not match the source tensor size at dim ",
        dim,
        ". Expected ",
        tensor.size(dim),
        ", but got total size ",
        total_size, PTA_ERROR(ErrCode::VALUE));

    auto chunks = tensor.split_with_sizes(chunk_sizes, dim);
    c10_npu::OptionalNPUStreamGuard npu_guard;
    for (const auto i : c10::irange(chunks.size())) {
        if (i < (streams ? streams->size() : 0U) && (*streams)[i]) {
            const auto device_index = static_cast<int16_t>(out_tensors[i].get_device());
            TORCH_CHECK((*streams)[i]->device_index() == device_index,
                "Expected the device associated with the stream at index ",
                i,
                " (was ",
                (*streams)[i]->device_index(),
                ") ",
                "to match the device supplied at that index (expected ",
                device_index,
                ")" + PTA_ERROR(ErrCode::VALUE));
            npu_guard.reset_stream(*(*streams)[i]);
        }
        // NB: We don't detect the case where `out_tensor` is already the correct
        //     view of `tensor` since that would be nontrivial and involve checking
        //     ptr, offset, and strides. So `scatter_out(src, src.chunk(...))` does
        //     more copying than `scatter(src)`.
        out_tensors[i].copy_(chunks[i], true);
    }
    return out_tensors;
}

std::vector<at::Tensor> scatter(const at::Tensor& tensor, at::IntArrayRef devices,
    const c10::optional<std::vector<int64_t>>& chunk_sizes, int64_t dim,
    const c10::optional<std::vector<c10::optional<c10_npu::NPUStream>>>& streams)
{
    TORCH_CHECK(!devices.empty(), "Expected at least one device to scatter to" + PTA_ERROR(ErrCode::VALUE));
    if (chunk_sizes.has_value()) {
        TORCH_CHECK(chunk_sizes->size() == devices.size(),
            "Expected devices and chunk_sizes to be of same length, but got len(devices) = ",
            devices.size(),
            " and len(chunk_sizes) = ",
            chunk_sizes->size(), PTA_ERROR(ErrCode::VALUE));
    }
    dim = at::maybe_wrap_dim(dim, tensor);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<at::Tensor> chunks = chunk_sizes
        ? tensor.split_with_sizes(*chunk_sizes, dim)
        : tensor.chunk(devices.size(), dim);
    c10_npu::OptionalNPUStreamGuard npu_guard;

    for (const auto i : c10::irange(chunks.size())) {
        const auto device_index = static_cast<int16_t>(devices[i]);
        if (device_index != tensor.get_device()) {
            if (i < (streams ? streams->size() : 0U) && (*streams)[i]) {
                TORCH_CHECK((*streams)[i]->device_index() == device_index,
                    "Expected the device associated with the stream at index ",
                    i,
                    " (was ",
                    (*streams)[i]->device_index(),
                    ") to match the device supplied at that index (expected ",
                    device_index,
                    ")" + PTA_ERROR(ErrCode::VALUE));
                npu_guard.reset_stream(*(*streams)[i]);
            }
            TORCH_CHECK(device_index >= 0, "Expected non-negative device index, but got ", device_index, PTA_ERROR(ErrCode::VALUE));
            chunks[i] = chunks[i].to({DeviceType::PrivateUse1, device_index}, true, false);
        }
    }
    return chunks;
}

// Gather
static inline at::Tensor& _gather_out_impl(at::TensorList tensors, at::Tensor& out_tensor, int64_t dim)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<int64_t> chunk_sizes;
    chunk_sizes.reserve(tensors.size());
    for (auto& tensor : tensors) {
        chunk_sizes.emplace_back(tensor.size(dim));
    }
    auto chunks = out_tensor.split_with_sizes(chunk_sizes, dim);
    for (const auto i : c10::irange(tensors.size())) {
        chunks[i].copy_(tensors[i], torch_npu::utils::is_npu(out_tensor));
    }
    return out_tensor;
}

at::Tensor& gather_out(at::TensorList tensors, at::Tensor& out_tensor, int64_t dim)
{
    TORCH_CHECK(!tensors.empty(), "Expected at least one tensor to gather from" + PTA_ERROR(ErrCode::VALUE));
    int64_t total_size = 0;
    auto& first = tensors.front();
    const auto first_size = first.sizes();
    dim = at::maybe_wrap_dim(dim, first);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<int64_t> expected_size(first_size.begin(), first_size.end());
    for (const auto i : c10::irange(tensors.size())) {
        const auto& tensor = tensors[i];
        TORCH_CHECK(torch_npu::utils::is_npu(tensor),
            "Expected all input tensors to be NPU tensors, but "
            "tensor at index ",
            i,
            " has device '",
            tensor.device(),
            "'", PTA_ERROR(ErrCode::TYPE));
        TORCH_CHECK(tensor.ndimension() == static_cast<int64_t>(expected_size.size()),
            "Expected all input tensors to have the same number of dimensions, but ",
            "tensor at index ",
            i,
            "has ",
            tensor.ndimension(),
            " dimensions, (expected ",
            expected_size.size(),
            ")", PTA_ERROR(ErrCode::VALUE));
        expected_size[dim] = tensor.size(dim);
        for (const auto dimension : c10::irange(expected_size.size())) {
            TORCH_CHECK(expected_size[dimension] == tensor.size(dimension),
                "Input tensor at index ",
                i,
                " has invalid shape ",
                tensor.sizes(),
                ", but expected ",
                at::IntArrayRef(expected_size), PTA_ERROR(ErrCode::VALUE));
        }
        total_size += tensor.size(dim);
    }
    expected_size[dim] = total_size;
    TORCH_CHECK(out_tensor.sizes() == expected_size,
        "Expected out tensor to have shape ",
        at::IntArrayRef(expected_size),
        ", but got ",
        out_tensor.sizes(), PTA_ERROR(ErrCode::VALUE))
    return _gather_out_impl(tensors, out_tensor, dim);
}

at::Tensor gather(at::TensorList tensors, int64_t dim, c10::optional<int32_t> destination_index)
{
    TORCH_CHECK(!tensors.empty(), "Expected at least one tensor to gather from" + PTA_ERROR(ErrCode::VALUE));
    int64_t total_size = 0;
    auto& first = tensors.front();
    const auto first_size = first.sizes();
    dim = at::maybe_wrap_dim(dim, first);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<int64_t> expected_size(first_size.begin(), first_size.end());
    auto memory_format = first.suggest_memory_format();
    for (const auto i : c10::irange(tensors.size())) {
        const auto& tensor = tensors[i];
        TORCH_CHECK(
            torch_npu::utils::is_npu(tensor),
            "Expected all input tensors to be NPU tensors, but "
            "tensor at index ",
            i,
            " has device ",
            tensor.device(), PTA_ERROR(ErrCode::TYPE));
        TORCH_CHECK(
            tensor.ndimension() == static_cast<int64_t>(expected_size.size()),
            "Expected all input tensors to have the same number of dimensions, but ",
            "tensor at index ",
            i,
            "has ",
            tensor.ndimension(),
            " dimensions, (expected ",
            expected_size.size(),
            ")" + PTA_ERROR(ErrCode::VALUE));
        expected_size[dim] = tensor.size(dim);
        for (const auto dimension : c10::irange(expected_size.size())) {
            TORCH_CHECK(
                expected_size[dimension] == tensor.size(dimension),
                "Input tensor at index ",
                i,
                " has invalid shape ",
                tensor.sizes(),
                ", but expected ",
                at::IntArrayRef(expected_size), PTA_ERROR(ErrCode::VALUE));
        }
        total_size += tensor.size(dim);
        if (memory_format != MemoryFormat::Contiguous && tensor.suggest_memory_format() != memory_format) {
            memory_format = MemoryFormat::Contiguous;
        }
    }
    expected_size[dim] = total_size;
    at::Device device(DeviceType::CPU);
    if (!destination_index || *destination_index != -1) {
        device = at::Device(DeviceType::PrivateUse1, destination_index ? *destination_index : -1);
    }

    at::Tensor result = at::empty(expected_size, first.options().device(device), memory_format);
    return _gather_out_impl(tensors, result, dim);
}

} // namespace torch_npu
}
