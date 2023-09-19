#include <ATen/ATen.h>
#include <torch_npu/csrc/core/NPUStorageImpl.h>
#include <torch_npu/csrc/framework/graph/execute/GraphExecutor.h>

namespace at_npu {
namespace native {

struct TensorInfo {
    TensorInfo(const std::vector<int64_t>& size, const std::vector<int64_t>& stride, const int64_t& offset,
               const caffe2::TypeMeta& type, const torch_npu::NPUStorageDesc& desc):
               sizes(size), strides(stride), storage_offset(offset), dtype(type),
               storage_desc(desc) {}

    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    int64_t storage_offset = 0;
    caffe2::TypeMeta dtype;
    torch_npu::NPUStorageDesc storage_desc;
};

struct AtTensorInfoAndMap {
    std::vector<TensorInfo> at_tensor_info;
    std::vector<int64_t> mapping;
    std::vector<torch_npu::NpuGraphDesc> graph_desc_info;
};

struct ReplayGraphInfo {
    uint32_t graph_id_ = 0;
    /* input struct */
    AtTensorInfoAndMap inputs;
    std::vector<ge::Tensor> graph_inputs_ge_tensors;
    /* output struct */
    AtTensorInfoAndMap assigned_outputs;
    AtTensorInfoAndMap returnable_outputs;
    AtTensorInfoAndMap inner_outputs;
    std::vector<ge::Tensor> graph_outputs_ge_tensors;
    std::vector<at::Tensor> inner_outputs_tensors;
};

class ReplayGraphImpl {
public:
    ReplayGraphImpl() = default;

    void GenerateGraph(const at::TensorList& inputs, at::TensorList assigned_outputs,
                        at::TensorList returnable_outputs, bool retain_inner_output);
    std::vector<at::Tensor> Replay(const at::TensorList& inputs, at::TensorList assigned_outputs);
    std::vector<at::Tensor> GetInnerOutputs(const at::TensorList& inputs);
    bool ReplayCacheHit(const at::TensorList& inputs);

private:
    bool retain_inner_output_ = false;
    std::unordered_map<hash_t, ReplayGraphInfo> replay_graph_cache_;
    int64_t FindMapping(const std::vector<int64_t>& graph_uid, const torch_npu::NpuGraphDesc& desc);
    void BuildReplayGraphInfo(const at::TensorList& tensors, AtTensorInfoAndMap& build_tensor_struct,
                              CombinedInfo& combinedinfo);
    void SetInnerOutput(CombinedInfo& outputcombinedinfo, ReplayGraphInfo& graphinfo);
    void GetInputUniqueId(CombinedInfo& input_infos);
    void GetOutputUniqueId(CombinedInfo& output_infos);
    void BuildReplayGraphInfoAll(const at::TensorList& inputs, at::TensorList assigned_outputs,
                                 at::TensorList returnable_outputs, CombinedInfo& input_infos,
                                 CombinedInfo& output_infos, ReplayGraphInfo& graphinfo);
    void ClearGraphExecutor();
    void SetInputGeTensor(ReplayGraphInfo& graphinfo, const at::TensorList& inputs);
    std::vector<at::Tensor> SetOutputGeTensor(ReplayGraphInfo& graphinfo,
                                              at::TensorList assigned_outputs);
    std::vector<at::Tensor> SetOutputGeTensorAndSetReturnable(ReplayGraphInfo& graphinfo,
                                                              AtTensorInfoAndMap& build_tensor_struct);
};

class ReplayGraph {
public:
    ReplayGraph() : replay_graph_(std::make_shared<ReplayGraphImpl>()) {};
    ~ReplayGraph() {
        replay_graph_ = nullptr;
    }

    void GenerateGraph(const at::TensorList& inputs, at::TensorList assigned_outputs,
                        at::TensorList returnable_outputs, bool retain_inner_output = false);
    std::vector<at::Tensor> Replay(const at::TensorList& inputs, at::TensorList assigned_outputs);
    std::vector<at::Tensor> GetInnerOutputs(const at::TensorList& inputs);
    bool ReplayCacheHit(const at::TensorList& inputs);

private:
    std::shared_ptr<ReplayGraphImpl> replay_graph_;
};
}
}