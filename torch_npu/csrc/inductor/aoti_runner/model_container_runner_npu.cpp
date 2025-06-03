#include <torch_npu/csrc/inductor/aoti_runner/model_container_runner_npu.h>
#include <torch_npu/csrc/inductor/aoti_torch/oss_proxy_executor_npu.h>

#ifndef _WIN32
#include <sys/stat.h>
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

namespace {
bool file_exists(std::string& path) {
#ifdef _WIN32
  return fs::exists(path);
#else
  struct stat rc {};
  return lstat(path.c_str(), &rc) == 0;
#endif
}
} // namespace

namespace torch::inductor {

AOTIModelContainerRunnerNpu::AOTIModelContainerRunnerNpu(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& cubin_dir)
    : AOTIModelContainerRunner(
          model_so_path,
          num_models,
          device_str,
          cubin_dir) {
  model_so_path_ = model_so_path;
  init_flag_ = false;
}

AOTIModelContainerRunnerNpu::~AOTIModelContainerRunnerNpu() = default;

void AOTIModelContainerRunnerNpu::init_proxy_executor() {
  if (init_flag_) return;

  init_flag_ = true;
  size_t lastindex = model_so_path_.find_last_of('.');
  std::string json_filename = model_so_path_.substr(0, lastindex) + "_npu.json";

  if (file_exists(json_filename)) {
    proxy_executor_npu_ = std::make_unique<torch::aot_inductor::OSSProxyExecutorNpu>(
        json_filename, false);
    proxy_executor_handle_ =
        reinterpret_cast<AOTIProxyExecutorHandle>(proxy_executor_npu_.get());
  } else {
    proxy_executor_handle_ = nullptr;
  }
}

std::vector<at::Tensor> AOTIModelContainerRunnerNpu::run(
  const std::vector<at::Tensor>& inputs, void* stream_handle) {
  init_proxy_executor();
  c10_npu::NPUStream npu_stream = c10_npu::getCurrentNPUStream();
  return AOTIModelContainerRunner::run(
      inputs, reinterpret_cast<AOTInductorStreamHandle>(npu_stream.stream()));
}

std::vector<at::Tensor> AOTIModelContainerRunnerNpu::run_with_npu_stream(
    std::vector<at::Tensor>& inputs,
    c10_npu::NPUStream npu_stream) {
  return AOTIModelContainerRunner::run(
      inputs, reinterpret_cast<AOTInductorStreamHandle>(npu_stream.stream()));
}

void AOTIModelContainerRunnerNpu::set_proxy_executor(AOTIProxyExecutorHandle handle) {
  proxy_executor_handle_ = handle;
  init_flag_ = true;
}

namespace {
std::unique_ptr<AOTIModelContainerRunner> create_aoti_runner_npu(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& cubin_dir) {
  return std::make_unique<AOTIModelContainerRunnerNpu>(
      model_so_path, num_models, device_str, cubin_dir);
}

RegisterAOTIModelRunner register_npu_runner("npu", &create_aoti_runner_npu);
}


} // namespace torch::inductor
