#include <unistd.h>

#include <vector>
#include <map>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>

#include "torch_npu/csrc/toolkit/profiler/inc/data_dumper.h"
#include "torch_npu/csrc/toolkit/profiler/common/utils.h"
#include "torch_npu/csrc/core/npu/npu_log.h"

namespace torch_npu {
namespace toolkit {
namespace profiler {
DataDumper::DataDumper() : path_(""), start_(false), init_(false) {}

DataDumper::~DataDumper()
{
    UnInit();
}

void DataDumper::Init(const std::string &path, size_t capacity = kDefaultRingBuffer)
{
    path_ = path;
    data_chunk_buf_.Init(capacity);
    init_.store(true);
}

void DataDumper::UnInit()
{
    if (init_.load()) {
        data_chunk_buf_.UnInit();
        init_.store(false);
        start_.store(false);
        for (auto &f : fd_map_) {
            if (f.second != nullptr) {
                fclose(f.second);
                f.second = nullptr;
            }
        }
        fd_map_.clear();
    }
}

void DataDumper::Start()
{
    if (!init_.load() || Thread::Start() != 0) {
        return;
    }
    start_.store(true);
}

void DataDumper::Stop()
{
    if (start_.load() == true) {
        start_.store(false);
        Thread::Stop();
    }
    Flush();
}

void DataDumper::GatherAndDumpData()
{
    std::map<std::string, std::vector<uint8_t>> dataMap;
    uint64_t batchSize = 0;
    while (batchSize < kBatchMaxLen) {
        std::unique_ptr<BaseReportData> data = nullptr;
        if (!data_chunk_buf_.Pop(data) || data == nullptr) {
            break;
        }
        std::vector<uint8_t> encodeData = data->encode();
        batchSize += encodeData.size();
        const std::string &key = data->tag;
        auto iter = dataMap.find(key);
        if (iter == dataMap.end()) {
            dataMap.insert({ key, encodeData });
        } else {
            iter->second.insert(iter->second.end(), encodeData.cbegin(), encodeData.cend());
        }
    }
    if (dataMap.size() > 0) {
        static bool create_flag = true;
        if (create_flag) {
            create_flag = !Utils::CreateDir(this->path_);
        }
        Dump(dataMap);
    }
}

void DataDumper::Run()
{
    for (;;) {
        if (!start_.load()) {
            break;
        }
        if (data_chunk_buf_.Size() > kNotifyInterval) {
            GatherAndDumpData();
        } else {
            usleep(kMaxWaitTimeUs);
        }
    }
}

void DataDumper::Flush()
{
    while (data_chunk_buf_.Size() != 0) {
        GatherAndDumpData();
    }
}

void DataDumper::Report(std::unique_ptr<BaseReportData> data)
{
    if (C10_UNLIKELY(!start_.load() || data == nullptr)) {
        return;
    }
    data_chunk_buf_.Push(std::move(data));
}

void DataDumper::Dump(const std::map<std::string, std::vector<uint8_t>> &dataMap)
{
    for (auto &data : dataMap) {
        FILE *fd = nullptr;
        const std::string dump_file = path_ + "/" + data.first;
        auto iter = fd_map_.find(dump_file);
        if (iter == fd_map_.end()) {
            if (!Utils::IsFileExist(dump_file) && !Utils::CreateFile(dump_file)) {
                ASCEND_LOGE("DataDumper cerate file failed: %s", dump_file.c_str());
                continue;
            }
            fd = fopen(dump_file.c_str(), "ab");
            if (fd == nullptr) {
                ASCEND_LOGE("DataDumper open file failed: %s", dump_file.c_str());
                continue;
            }
            fd_map_.insert({ dump_file, fd });
        } else {
            fd = iter->second;
        }
        fwrite(reinterpret_cast<const char *>(data.second.data()), sizeof(char), data.second.size(), fd);
    }
}

TraceDataDumper::TraceDataDumper() : path_(""), start_(false), init_(false), trace_hash_data_(nullptr) {}

TraceDataDumper::~TraceDataDumper()
{
    UnInit();
}

void TraceDataDumper::Init(const std::string &path, size_t capacity)
{
    path_ = path;
    trace_data_buf_.Init(capacity);
    init_.store(true);
}

void TraceDataDumper::UnInit()
{
    if (init_.load()) {
        trace_data_buf_.UnInit();
        init_.store(false);
        start_.store(false);
        for (auto &f : fd_map_) {
            if (f.second != nullptr) {
                fclose(f.second);
                f.second = nullptr;
            }
        }
        fd_map_.clear();
    }
}

void TraceDataDumper::Start()
{
    if (!init_.load() || Thread::Start() != 0) {
        return;
    }
    start_.store(true);
}

void TraceDataDumper::Stop()
{
    if (start_.load() == true) {
        start_.store(false);
        Thread::Stop();
    }
    while (trace_data_buf_.Size() != 0) {
        FlushTraceData();
    }
    FlushHashData();
    FlushParamData();
}

void TraceDataDumper::Run()
{
    while (true) {
        if (!start_.load()) {
            break;
        }
        if (trace_data_buf_.Size() > 0) {
            FlushTraceData();
        } else {
            usleep(kMaxWaitTimeUs);
        }
    }
}

void TraceDataDumper::Report(std::unique_ptr<PythonTracerFuncData> data)
{
    if (C10_UNLIKELY(!start_.load() || data == nullptr)) {
        return;
    }
    trace_data_buf_.Push(std::move(data));
}

void TraceDataDumper::ReportHash(std::unique_ptr<PythonTracerHashData> data)
{
    if (C10_UNLIKELY(!start_.load() || data == nullptr)) {
        return;
    }
    trace_hash_data_ = std::move(data);
}

void TraceDataDumper::ReportParam(std::unique_ptr<ParamTensorData> data)
{
    if (C10_UNLIKELY(!start_.load() || data == nullptr)) {
        return;
    }
    param_data_ = std::move(data);
}

void TraceDataDumper::CreateDumpDir()
{
    static bool create_flag = true;
    if (create_flag) {
        create_flag = !Utils::CreateDir(this->path_);
    }
}

void TraceDataDumper::FlushTraceData()
{
    std::unique_ptr<PythonTracerFuncData> data = nullptr;
    if (!trace_data_buf_.Pop(data) || data == nullptr) {
        return;
    }
    auto encode_data = data->encode();
    if (!encode_data.empty()) {
        CreateDumpDir();
        Dump(data->tag, encode_data);
    }
}

void TraceDataDumper::FlushHashData()
{
    if (trace_hash_data_ == nullptr) {
        return;
    }
    auto encode_data = trace_hash_data_->encode();
    if (!encode_data.empty()) {
        CreateDumpDir();
        Dump(trace_hash_data_->tag, encode_data);
    }
    trace_hash_data_ = nullptr;
}

void TraceDataDumper::FlushParamData()
{
    if (param_data_ == nullptr) {
        return;
    }
    auto encode_data = param_data_->encode();
    if (!encode_data.empty()) {
        CreateDumpDir();
        Dump(param_data_->tag, encode_data);
    }
    param_data_ = nullptr;
}

void TraceDataDumper::Dump(const std::string &file_name, const std::vector<uint8_t> &encode_data)
{
    FILE *fd = nullptr;
    const std::string dump_file = path_ + "/" + file_name;
    auto iter = fd_map_.find(dump_file);
    if (iter == fd_map_.end()) {
        if (!Utils::IsFileExist(dump_file) && !Utils::CreateFile(dump_file)) {
            ASCEND_LOGE("TraceDataDumper cerate file failed: %s", dump_file.c_str());
            return;
        }
        fd = fopen(dump_file.c_str(), "ab");
        if (fd == nullptr) {
            ASCEND_LOGE("TraceDataDumper open file failed: %s", dump_file.c_str());
            return;
        }
        fd_map_.insert({ dump_file, fd });
    } else {
        fd = iter->second;
    }
    fwrite(reinterpret_cast<const char *>(encode_data.data()), sizeof(char), encode_data.size(), fd);
}
} // profiler
} // toolkit
} // torch_npu
