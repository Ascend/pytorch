#include "torch_npu/csrc/toolkit/profiler/inc/data_reporter.h"

namespace torch_npu {
namespace toolkit {
namespace profiler {
std::vector<uint8_t> OpRangeData::encode()
{
    std::vector<uint8_t> result;
    encodeFixedData<int64_t>({start_ns, end_ns, sequence_number}, result);
    encodeFixedData<uint64_t>({process_id, start_thread_id, end_thread_id, forward_thread_id}, result);
    result.push_back(is_async);
    encodeStrData(static_cast<uint16_t>(OpRangeDataType::NAME), name, result);
    if (!input_dtypes.empty()) {
        encodeStrArrayData(static_cast<uint16_t>(OpRangeDataType::INPUT_DTYPES), input_dtypes, result);
    }
    if (!input_shapes.empty()) {
        encode2DIntegerMatrixDatas<int64_t>(static_cast<uint16_t>(OpRangeDataType::INPUT_SHAPE), input_shapes, result);
    }
    if (!stack.empty()) {
        encodeStrArrayData(static_cast<uint16_t>(OpRangeDataType::STACK), stack, result);
    }
    if (!module_hierarchy.empty()) {
        encodeStrArrayData(static_cast<uint16_t>(OpRangeDataType::MODULE_HIERARCHY), module_hierarchy, result);
    }
    if (!extra_args.empty()) {
        encodeMapData(static_cast<uint16_t>(OpRangeDataType::EXTRA_ARGS), extra_args, result);
    }

    std::vector<uint8_t> resultTLV;
    uint16_t dataType = static_cast<uint16_t>(OpRangeDataType::OP_RANGE_DATA);
    for (size_t i = 0; i < sizeof(uint16_t); ++i) {
        resultTLV.push_back((dataType >> (i * 8)) & 0xff);
    }
    uint32_t length = result.size();
    for (size_t i = 0; i < sizeof(uint32_t); ++i) {
        resultTLV.push_back((length >> (i * 8)) & 0xff);
    }
    resultTLV.insert(resultTLV.end(), result.cbegin(), result.cend());
    return resultTLV;
}

std::vector<uint8_t> OpMarkData::encode()
{
    std::vector<uint8_t> result;
    encodeFixedData<int64_t>({time_ns}, result);
    encodeFixedData<uint64_t>({category, correlation_id, thread_id, process_id}, result);
    encodeStrData(static_cast<uint16_t>(OpMarkDataType::NAME), name, result);

    std::vector<uint8_t> resultTLV;
    uint16_t dataType = static_cast<uint16_t>(OpMarkDataType::OP_MARK_DATA);
    for (size_t i = 0; i < sizeof(uint16_t); ++i) {
        resultTLV.push_back((dataType >> (i * 8)) & 0xff);
    }
    uint32_t length = result.size();
    for (size_t i = 0; i < sizeof(uint32_t); ++i) {
        resultTLV.push_back((length >> (i * 8)) & 0xff);
    }
    resultTLV.insert(resultTLV.end(), result.cbegin(), result.cend());
    return resultTLV;
}

std::vector<uint8_t> MemoryData::encode() {
    std::vector<uint8_t> result;
    encodeFixedData<int64_t>({ptr, time_ns, alloc_size, total_allocated,
                              total_reserved, total_active, stream_ptr},
                             result);
    encodeFixedData<int8_t>({device_type}, result);
    encodeFixedData<uint8_t>({device_index, data_type}, result);
    encodeFixedData<uint64_t>({thread_id, process_id}, result);

    std::vector<uint8_t> resultTLV;
    uint16_t dataType = static_cast<uint16_t>(MemoryDataType::MEMORY_DATA);
    for (size_t i = 0; i < sizeof(uint16_t); ++i) {
        resultTLV.push_back((dataType >> (i * 8)) & 0xff);
    }
    uint32_t length = result.size();
    for (size_t i = 0; i < sizeof(uint32_t); ++i) {
        resultTLV.push_back((length >> (i * 8)) & 0xff);
    }
    resultTLV.insert(resultTLV.end(), result.cbegin(), result.cend());
    return resultTLV;
}

std::vector<uint8_t> PythonFuncCallData::encode()
{
    std::vector<uint8_t> result;
    encodeFixedData<uint64_t>({start_ns, thread_id, process_id}, result);
    encodeFixedData<uint8_t>({trace_tag}, result);
    encodeStrData(static_cast<uint16_t>(PythonFuncCallDataType::NAME), func_name, result);

    std::vector<uint8_t> resultTLV;
    uint16_t dataType = static_cast<uint16_t>(PythonFuncCallDataType::PYTHON_FUNC_CALL_DATA);
    for (size_t i = 0; i < sizeof(uint16_t); ++i) {
        resultTLV.push_back((dataType >> (i * 8)) & 0xff);
    }
    uint32_t length = result.size();
    for (size_t i = 0; i < sizeof(uint32_t); ++i) {
        resultTLV.push_back((length >> (i * 8)) & 0xff);
    }
    resultTLV.insert(resultTLV.end(), result.cbegin(), result.cend());
    return resultTLV;
}

std::vector<uint8_t> PythonModuleCallData::encode()
{
    std::vector<uint8_t> result;
    encodeFixedData<uint64_t>({idx, thread_id, process_id}, result);
    encodeStrData(static_cast<uint16_t>(PythonModuleCallDataType::MODULE_UID), module_uid, result);
    encodeStrData(static_cast<uint16_t>(PythonModuleCallDataType::MODULE_NAME), module_name, result);

    std::vector<uint8_t> resultTLV;
    uint16_t dataType = static_cast<uint16_t>(PythonModuleCallDataType::PYTHON_MODULE_CALL_DATA);
    for (size_t i = 0; i < sizeof(uint16_t); ++i) {
        resultTLV.push_back((dataType >> (i * 8)) & 0xff);
    }
    uint32_t length = result.size();
    for (size_t i = 0; i < sizeof(uint32_t); ++i) {
        resultTLV.push_back((length >> (i * 8)) & 0xff);
    }
    resultTLV.insert(resultTLV.end(), result.cbegin(), result.cend());
    return resultTLV;
}
} // profiler
} // toolkit
} // torch_npu
