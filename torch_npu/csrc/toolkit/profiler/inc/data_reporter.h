#pragma once

#include "torch_npu/csrc/profiler/containers.h"
#include "torch_npu/csrc/profiler/profiler_python.h"

#include <stdint.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <utility>

#include <ATen/core/ivalue.h>
#include <ATen/record_function.h>
#include <c10/core/ScalarType.h>

namespace torch_npu {
namespace toolkit {
namespace profiler {

template<typename T>
std::string to_string(T value)
{
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

template<typename T>
void encodeFixedData(const std::vector<T> &datas, std::vector<uint8_t> &result)
{
    for (auto data : datas) {
        for (size_t i = 0; i < sizeof(T); ++i) {
            result.push_back((static_cast<size_t>(data) >> (i * 8)) & 0xff);
        }
    }
}

inline void encodeStrData(uint16_t type, const std::string &data, std::vector<uint8_t> &result)
{
    for (size_t i = 0; i < sizeof(uint16_t); ++i) {
        result.push_back((type >> (i * 8)) & 0xff);
    }
    uint32_t length = data.size();
    for (size_t i = 0; i < sizeof(uint32_t); ++i) {
        result.push_back((length >> (i * 8)) & 0xff);
    }
    for (const auto &c : data) {
        result.push_back(c);
    }
}

inline void encodeStrArrayData(uint16_t type, const std::vector<std::string> &datas, std::vector<uint8_t> &result)
{
    std::string rst;
    for (auto str : datas) {
        rst += (str + ";");
    }
    if (!rst.empty()) {
        rst.pop_back();
    }
    encodeStrData(type, rst, result);
}

inline void encodeMapData(uint16_t type, const std::unordered_map<std::string, c10::IValue> &datas,
    std::vector<uint8_t> &result)
{
    std::string rst;
    for (auto &entry : datas) {
        rst += entry.first + ":" + to_string(entry.second) + ";";
    }
    if (!rst.empty()) {
        rst.pop_back();
    }
    encodeStrData(type, rst, result);
}

template<typename T>
void encode2DIntegerMatrixDatas(uint16_t type, std::vector<std::vector<T>> &datas, std::vector<uint8_t> &result)
{
    std::string rst;
    for (auto tensor : datas) {
        std::stringstream ss;
        copy(tensor.begin(), tensor.end(), std::ostream_iterator<T>(ss, ","));
        std::string str = ss.str();
        if (!str.empty()) {
            str.pop_back();
        }
        rst += (str + ";");
    }
    if (!rst.empty()) {
        rst.pop_back();
    }
    encodeStrData(type, rst, result);
}

class WeakTensor {
public:
    explicit WeakTensor(const at::Tensor &t) : weak_self_(t.getIntrusivePtr()) {}

    auto get() const
    {
        return (c10::TensorImpl *)(weak_self_._unsafe_get_target());
    }

private:
    c10::weak_intrusive_ptr<c10::TensorImpl> weak_self_;
};

struct TensorMetadata {
    TensorMetadata() = default;
    explicit TensorMetadata(const at::Tensor &t);

    c10::TensorImpl *impl_;
    const void *ptr_;
    std::string dtype_;
    uint64_t dtype_size_{ 0 };
    std::vector<int64_t> sizes_;
    std::vector<int64_t> strides_;
    int device_type_{ 0 };
    int device_index_{ -1 };
};

struct ModuleParam {
    std::string name_;
    TensorMetadata metadata_;
    c10::optional<TensorMetadata> grad_;
};

struct OptimizerParam {
    TensorMetadata metadata_;
    c10::optional<TensorMetadata> grad_;
    std::vector<std::pair<std::string, TensorMetadata>> state_;
};

inline void appendWithDelimiter(std::ostringstream &oss, const std::string &str, char delimiter = ';')
{
    oss << str << delimiter;
}

inline std::string removeLastCharAndReturn(std::ostringstream &oss)
{
    std::string str = oss.str();
    if (!str.empty()) {
        str.pop_back();
    }
    return str;
}

template<typename T>
std::string VectorToString(const std::vector<T> &vec)
{
    std::ostringstream oss;
    if (!vec.empty()) {
        std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(oss, ","));
        std::string result = oss.str();
        result.pop_back();
        return result;
    }
    return "";
}

inline void encodeTensor(const TensorMetadata &t, std::ostringstream &oss)
{
    // append impl, ptr
    appendWithDelimiter(oss, to_string(t.impl_));
    appendWithDelimiter(oss, (t.ptr_ == nullptr) ? "" : to_string(t.ptr_));

    // append dtype, dtype_size, sizes and strides
    appendWithDelimiter(oss, t.dtype_);
    appendWithDelimiter(oss, to_string(t.dtype_size_));
    appendWithDelimiter(oss, VectorToString(t.sizes_));
    appendWithDelimiter(oss, VectorToString(t.strides_));

    // append device info
    appendWithDelimiter(oss, to_string(t.device_type_));
    oss << to_string(t.device_index_);
}

inline void encodeTensors(uint16_t type, std::vector<TensorMetadata> tensors, std::vector<uint8_t> &result)
{
    std::ostringstream oss;
    for (auto &tensor : tensors) {
        encodeTensor(tensor, oss);
        oss << ")";
    }
    std::string str = removeLastCharAndReturn(oss);
    encodeStrData(type, str, result);
}

inline void encodeTensorLists(uint16_t type, std::vector<std::vector<TensorMetadata>> tensorlists,
    std::vector<uint8_t> &result)
{
    std::ostringstream oss;
    for (auto &tensorlist : tensorlists) {
        for (auto &tensor : tensorlist) {
            encodeTensor(tensor, oss);
            oss << ")";
        }
        oss << "}";
    }
    std::string str = removeLastCharAndReturn(oss);
    encodeStrData(type, str, result);
}

inline void encodeModuleParams(uint16_t type, std::vector<ModuleParam> params, std::vector<uint8_t> &result)
{
    std::ostringstream oss;
    for (auto &param : params) {
        appendWithDelimiter(oss, param.name_, ')');
        encodeTensor(param.metadata_, oss);
        oss << ")";
        if (param.grad_.has_value()) {
            encodeTensor(*(param.grad_), oss);
        }
        oss << "}";
    }
    std::string str = removeLastCharAndReturn(oss);
    encodeStrData(type, str, result);
}

inline void encodeOptimizerParams(uint16_t type, std::vector<OptimizerParam> params, std::vector<uint8_t> &result)
{
    std::ostringstream oss;
    for (auto &param : params) {
        encodeTensor(param.metadata_, oss);
        oss << ")";
        if (param.grad_.has_value()) {
            encodeTensor(*(param.grad_), oss);
        }
        oss << ")";
        for (auto s : param.state_) {
            appendWithDelimiter(oss, s.first, '>');
            encodeTensor(s.second, oss);
            oss << "]";
        }
        oss << "}";
    }
    std::string str = removeLastCharAndReturn(oss);
    encodeStrData(type, str, result);
}

struct BaseReportData {
    int32_t device_id{ 0 };
    std::string tag;
    BaseReportData(int32_t device_id, std::string tag) : device_id(device_id), tag(std::move(tag)) {}
    virtual ~BaseReportData() = default;
    virtual std::vector<uint8_t> encode() = 0;
};

enum class FwkDataType {
    OP_RANGE_DATA = 1,
    OP_MARK_DATA = 2,
    MEMORY_DATA = 3,
    PYTHON_TRACER_HASH_DATA = 4,
    PARAM_TENSOR_DATA = 5,
};

enum class OpRangeDataType {
    IS_ASYNC = 1,
    NAME = 2,
    INPUT_DTYPES = 3,
    INPUT_SHAPES = 4,
    INPUT_TENSORS = 5,
    INPUT_TENSORLISTS = 6,
    INPUT_SCALARS = 7,
    STACK = 8,
    MODULE_HIERARCHY = 9,
    EXTRA_ARGS = 10,
    RESERVED = 30,
};

struct OpRangeData : BaseReportData {
    int64_t start_ns{ 0 };
    int64_t end_ns{ 0 };
    int64_t sequence_number{ 0 };
    uint64_t process_id{ 0 };
    uint64_t start_thread_id{ 0 };
    uint64_t end_thread_id{ 0 };
    uint64_t forward_thread_id{ 0 };
    bool is_async{ false };
    std::string name;
    uint8_t scope{ 0 };
    std::vector<std::string> input_dtypes;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<TensorMetadata> input_tensors;
    std::vector<std::vector<TensorMetadata>> input_tensorlists;
    std::vector<std::string> input_scalars;
    std::vector<std::string> stack;
    std::vector<std::string> module_hierarchy;
    std::unordered_map<std::string, c10::IValue> extra_args;
    OpRangeData(int64_t start_ns, int64_t end_ns, int64_t sequence_number, uint64_t process_id,
        uint64_t start_thread_id, uint64_t end_thread_id, uint64_t forward_thread_id, bool is_async, std::string name)
        : BaseReportData(0, "torch.op_range"),
          start_ns(start_ns),
          end_ns(end_ns),
          sequence_number(sequence_number),
          process_id(process_id),
          start_thread_id(start_thread_id),
          end_thread_id(end_thread_id),
          forward_thread_id(forward_thread_id),
          is_async(is_async),
          name(std::move(name))
    {}
    std::vector<uint8_t> encode();
};

enum class OpMarkDataType {
    NAME = 1,
};

struct OpMarkData : BaseReportData {
    int64_t time_ns{ 0 };
    uint64_t category{ 0 };
    uint64_t correlation_id{ 0 };
    uint64_t thread_id{ 0 };
    uint64_t process_id{ 0 };
    std::string name;
    OpMarkData(int64_t time_ns, uint64_t category, uint64_t correlation_id, uint64_t thread_id, uint64_t process_id,
        const std::string &name)
        : BaseReportData(0, "torch.op_mark"),
          time_ns(time_ns),
          category(category),
          correlation_id(correlation_id),
          thread_id(thread_id),
          process_id(process_id),
          name(name)
    {}
    std::vector<uint8_t> encode();
};

struct MemoryData : BaseReportData {
    int64_t ptr{ 0 };
    int64_t time_ns{ 0 };
    int64_t alloc_size{ 0 };
    int64_t total_allocated{ 0 };
    int64_t total_reserved{ 0 };
    int64_t total_active{ 0 };
    int64_t stream_ptr{ 0 };
    int8_t device_type{ 0 };
    int8_t device_index{ 0 };
    uint8_t component_type{ 0 };
    uint8_t data_type{ 0 };
    uint8_t allocator_type{ 0 };
    uint64_t thread_id{ 0 };
    uint64_t process_id{ 0 };
    MemoryData(int64_t ptr, int64_t time_ns, int64_t alloc_size, int64_t total_allocated, int64_t total_reserved,
        int64_t total_active, int64_t stream_ptr, int8_t device_type, int8_t device_index, uint8_t component_type,
        uint8_t data_type, uint8_t allocator_type, uint64_t thread_id, uint64_t process_id)
        : BaseReportData(0, "torch.memory_usage"),
          ptr(ptr),
          time_ns(time_ns),
          alloc_size(alloc_size),
          total_allocated(total_allocated),
          total_reserved(total_reserved),
          total_active(total_active),
          stream_ptr(stream_ptr),
          device_type(device_type),
          device_index(device_index),
          component_type(component_type),
          data_type(data_type),
          allocator_type(allocator_type),
          thread_id(thread_id),
          process_id(process_id)
    {}
    std::vector<uint8_t> encode();
};

struct PythonTracerFuncData : BaseReportData {
    uint64_t process_id{ 0 };
    torch_npu::profiler::AppendOnlyList<torch_npu::profiler::python_tracer::TraceEvent> events;
    PythonTracerFuncData(uint64_t process_id,
        torch_npu::profiler::AppendOnlyList<torch_npu::profiler::python_tracer::TraceEvent> &&events)
        : BaseReportData(0, "torch.python_tracer_func"), process_id(process_id), events(std::move(events))
    {}
    std::vector<uint8_t> encode();
};

enum class PythonTracerHashDataType { VALUE = 1 };

struct PythonTracerHashData : BaseReportData {
    std::vector<std::pair<uint64_t, std::string>> hash_data;
    PythonTracerHashData(std::vector<std::pair<uint64_t, std::string>> hash_data)
        : BaseReportData(0, "torch.python_tracer_hash"), hash_data(std::move(hash_data))
    {}
    std::vector<uint8_t> encode();
};

enum class ParamTensorDataType { MODULE_PARAM = 1, OPTIMIZER_PARAM = 2 };

struct ParamTensorData : BaseReportData {
    std::vector<std::pair<uint64_t, std::vector<ModuleParam>>> module_param_data;
    std::vector<std::pair<uint64_t, std::vector<OptimizerParam>>> optimizer_param_data;
    ParamTensorData(std::vector<std::pair<uint64_t, std::vector<ModuleParam>>> module_param_data,
        std::vector<std::pair<uint64_t, std::vector<OptimizerParam>>> optimizer_param_data)
        : BaseReportData(0, "torch.param_tensor_info"),
          module_param_data(std::move(module_param_data)),
          optimizer_param_data(std::move(optimizer_param_data))
    {}
    std::vector<uint8_t> encode();
};
} // profiler
} // toolkit
} // torch_npu
