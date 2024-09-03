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
        encode2DIntegerMatrixDatas<int64_t>(static_cast<uint16_t>(OpRangeDataType::INPUT_SHAPES), input_shapes, result);
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
    encodeFixedData<int8_t>({device_type, device_index}, result);
    encodeFixedData<uint8_t>({data_type}, result);
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

std::vector<uint8_t> PythonTracerFuncData::encode()
{
    std::vector<uint8_t> result;
    for (const auto& item : events) {
        encodeFixedData<uint64_t>({item.ts_, thread_id, process_id, item.key_}, result);
        encodeFixedData<uint8_t>({item.tag_}, result);
    }
    return result;
}

std::vector<uint8_t> PythonTracerHashData::encode()
{
    std::vector<uint8_t> result;
    for (const auto& item : hash_data) {
        std::vector<uint8_t> item_data;
        encodeFixedData<uint64_t>({item.first}, item_data);
        encodeStrData(static_cast<uint16_t>(PythonTracerHashDataType::VALUE), item.second, item_data);

        std::vector<uint8_t> tlv_data;
        uint16_t dataType = static_cast<uint16_t>(PythonTracerHashDataType::PYTHON_TRACER_HASH_DATA);
        for (size_t i = 0; i < sizeof(uint16_t); ++i) {
            tlv_data.push_back((dataType >> (i * 8)) & 0xff);
        }
        uint32_t length = item_data.size();
        for (size_t i = 0; i < sizeof(uint32_t); ++i) {
            tlv_data.push_back((length >> (i * 8)) & 0xff);
        }
        tlv_data.insert(tlv_data.end(), item_data.cbegin(), item_data.cend());
        result.insert(result.end(), tlv_data.cbegin(), tlv_data.cend());
    }
    return result;
}
} // profiler
} // toolkit
} // torch_npu
