#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch_npu/csrc/inductor/aoti_runner/model_container_runner_npu.h>

namespace torch::inductor {

AOTIModelContainerRunnerNpu::AOTIModelContainerRunnerNpu(const std::string& model_so_path, size_t num_models,
                                                         const std::string& device_str, const std::string& cubin_dir,
                                                         const bool run_single_threaded)
    : AOTIModelContainerRunner(model_so_path, num_models, device_str, cubin_dir, run_single_threaded) {}

AOTIModelContainerRunnerNpu::~AOTIModelContainerRunnerNpu() = default;

std::vector<at::Tensor> AOTIModelContainerRunnerNpu::run_impl(std::vector<AtenTensorHandle>& input_handles,
                                                              void* stream_handle)
{
    c10_npu::NPUStream npu_stream = c10_npu::getCurrentNPUStream();
    return AOTIModelContainerRunner::run_impl(input_handles, reinterpret_cast<void*>(npu_stream.stream()));
}

std::vector<at::Tensor> AOTIModelContainerRunnerNpu::run_with_npu_stream(const std::vector<at::Tensor>& inputs,
                                                                         const c10_npu::NPUStream& npu_stream)
{
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
