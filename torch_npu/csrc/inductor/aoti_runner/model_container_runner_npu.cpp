#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch_npu/csrc/inductor/aoti_runner/model_container_runner_npu.h>
#include <torch_npu/csrc/inductor/aoti_torch/oss_proxy_executor_npu.h>

#include <iostream>

#ifndef _WIN32
#include <sys/stat.h>
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

namespace {
bool file_exists(std::string& path)
{
#ifdef _WIN32
    return fs::exists(path);
#else
    struct stat rc{};
    return lstat(path.c_str(), &rc) == 0;
#endif
}
} // namespace

namespace torch::inductor {

AOTIModelContainerRunnerNpu::AOTIModelContainerRunnerNpu(const std::string& model_so_path, size_t num_models,
                                                         const std::string& device_str, const std::string& cubin_dir,
                                                         const bool run_single_threaded)
    : AOTIModelContainerRunner(model_so_path, num_models, device_str, cubin_dir, run_single_threaded)
{
    model_so_path_ = model_so_path;
    init_flag_ = false;
}

AOTIModelContainerRunnerNpu::~AOTIModelContainerRunnerNpu() = default;

void AOTIModelContainerRunnerNpu::init_proxy_executor()
{
    if (init_flag_)
        return;

    init_flag_ = true;
    size_t lastindex = model_so_path_.find_last_of('.');
    std::string json_filename = model_so_path_.substr(0, lastindex) + "_npu.json";
    if (file_exists(json_filename)) {
        proxy_executor_npu_ = std::make_unique<torch::aot_inductor::OSSProxyExecutorNpu>(json_filename, false);
        proxy_executor_handle_ = reinterpret_cast<AOTIProxyExecutorHandle>(proxy_executor_npu_.get());
    } else {
        proxy_executor_handle_ = nullptr;
    }
}

std::vector<at::Tensor> AOTIModelContainerRunnerNpu::run_impl(std::vector<AtenTensorHandle>& input_handles,
                                                              void* stream_handle)
{
    init_proxy_executor();
    c10_npu::NPUStream npu_stream = c10_npu::getCurrentNPUStream();
    return AOTIModelContainerRunner::run_impl(input_handles, reinterpret_cast<void*>(npu_stream.stream()));
}

std::vector<at::Tensor> AOTIModelContainerRunnerNpu::run_with_npu_stream(const std::vector<at::Tensor>& inputs,
                                                                         const c10_npu::NPUStream& npu_stream)
{
    init_proxy_executor();
    c10_npu::NPUStream cur_npu_stream = c10_npu::getCurrentNPUStream();
    return run(inputs, reinterpret_cast<void*>(cur_npu_stream.stream()));
}

namespace {
std::unique_ptr<AOTIModelContainerRunner> create_aoti_runner_npu(const std::string& model_so_path, size_t num_models,
                                                                 const std::string& device_str,
                                                                 const std::string& cubin_dir,
                                                                 const bool run_single_threaded)
{
    return std::make_unique<AOTIModelContainerRunnerNpu>(model_so_path, num_models, device_str, cubin_dir,
                                                         run_single_threaded);
}
} // namespace

RegisterAOTIModelRunner register_npu_runner("npu", &create_aoti_runner_npu);

} // namespace torch::inductor
#endif
