#pragma once

#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/inductor/aoti_runner/model_container_runner.h>

namespace torch::inductor {
// NOTICE: Following APIs are subject to change due to active development
// We provide NO BC guarantee for these APIs
class TORCH_API AOTIModelContainerRunnerNpu : public AOTIModelContainerRunner {
public:
    // @param device_str: cuda device string, e.g. "cuda", "cuda:0"
    AOTIModelContainerRunnerNpu(const std::string &model_so_path, size_t num_models = 1,
        const std::string &device_str = "npu", const std::string &cubin_dir = "");

    ~AOTIModelContainerRunnerNpu();

    std::vector<at::Tensor> run(const std::vector<at::Tensor> &inputs, void *stream_handle = nullptr) override;

    std::vector<at::Tensor> run_with_npu_stream(std::vector<at::Tensor> &inputs, c10_npu::NPUStream npu_stream);

    void init_proxy_executor();

    void set_proxy_executor(AOTIProxyExecutorHandle handle);

private:
    std::string model_so_path_;
    bool init_flag_;
    std::unique_ptr<torch::aot_inductor::ProxyExecutor> proxy_executor_npu_;
};
} // namespace torch::inductor
