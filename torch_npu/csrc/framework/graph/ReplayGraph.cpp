// Copyright (c) 2020 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ReplayGraph.h"

#include <ATen/ATen.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/framework/graph/util/NPUGraphContextManager.h>
#include <torch_npu/csrc/framework/graph/execute/GraphExecutor.h>
#include <torch_npu/csrc/framework/graph/scalar/ScalarMemoryOps.h>
#include <torch_npu/csrc/framework/graph/util/ATenGeBridge.h>
#include <torch_npu/csrc/framework/graph/util/GraphUtils.h>
#include <torch_npu/csrc/framework/graph/util/NPUHashUtils.h>


namespace at_npu {
namespace native {
bool ReplayGraphImpl::ReplayCacheHit(const at::TensorList& inputs) {
    TORCH_CHECK(!inputs.empty(), "Input tensorlist must have one tensor at least");
    auto input_tensor_shape = inputs.front().sizes();
    if (replay_graph_cache_.empty() || (replay_graph_cache_.find(multi_hash(input_tensor_shape))
        == replay_graph_cache_.end())) {
            return false;
        }
    return true;
}

void ReplayGraphImpl::SetInputGeTensor(ReplayGraphInfo& graphinfo, const at::TensorList& inputs) {
    for (size_t i = 0; i < inputs.size(); ++i) {
        int64_t idx = graphinfo.inputs.mapping[i];
        if (idx >= 0) {
            ge::TensorDesc tensor_desc = ATenGeBridge::InferGeTenosrDesc(
                graphinfo.inputs.at_tensor_info[i].storage_desc,
                graphinfo.inputs.graph_desc_info[i].graph_value.GetRealDtype());
            TORCH_CHECK(idx < graphinfo.graph_inputs_ge_tensors.size(),
                        "replay model internal error, please feedback bug.");
            if (NpuUtils::check_match(&inputs[i])) {
                auto data_ptr = inputs[i].data_ptr();
                TORCH_CHECK(data_ptr != nullptr, "Input for replay graph must have data ptr");
                size_t numel = NPUNativeFunctions::get_storage_size(inputs[i]);
                graphinfo.graph_inputs_ge_tensors[idx] = ATenGeBridge::MakeGeTensor(tensor_desc,
                    data_ptr, numel * inputs[i].itemsize());
            } else {
                auto contiguous_input = NpuUtils::format_contiguous(inputs[i]);
                auto data_ptr = contiguous_input.data_ptr();
                TORCH_CHECK(data_ptr != nullptr, "Input for replay graph must have data ptr");
                size_t numel = NPUNativeFunctions::get_storage_size(contiguous_input);
                graphinfo.graph_inputs_ge_tensors[idx] = ATenGeBridge::MakeGeTensor(tensor_desc,
                    data_ptr, numel * inputs[i].itemsize());
            }
        }
    }
}

std::vector<at::Tensor> ReplayGraphImpl::SetOutputGeTensorAndSetReturnable(ReplayGraphInfo& graphinfo,
                                                                           AtTensorInfoAndMap& build_tensor_struct) {
    std::vector<at::Tensor> tmp_outputs;
    for (size_t i = 0; i < build_tensor_struct.at_tensor_info.size(); i++) {
        auto options = at::TensorOptions().dtype(build_tensor_struct.at_tensor_info[i].dtype)
                        .device(at_npu::key::NativeDeviceType);
        auto tensor = NPUNativeFunctions::empty_with_format(build_tensor_struct.at_tensor_info[i].sizes,
            optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(),
            options.pinned_memory_opt(), build_tensor_struct.at_tensor_info[i].storage_desc.npu_format_);
        int64_t idx = build_tensor_struct.mapping[i];
        size_t numel = NPUNativeFunctions::get_storage_size(tensor);
        ge::TensorDesc tensor_desc = ATenGeBridge::InferGeTenosrDesc(
            build_tensor_struct.at_tensor_info[i].storage_desc,
            build_tensor_struct.graph_desc_info[i].graph_value.GetRealDtype());
        TORCH_CHECK(idx < graphinfo.graph_outputs_ge_tensors.size(),
                    "replay model internal error, please feedback bug.");
        graphinfo.graph_outputs_ge_tensors[idx] = ATenGeBridge::MakeGeTensor(tensor_desc, tensor.data_ptr(),
                                                                             numel * tensor.itemsize());
        tmp_outputs.emplace_back(tensor);
    }
    return tmp_outputs;
}

std::vector<at::Tensor> ReplayGraphImpl::SetOutputGeTensor(ReplayGraphInfo& graphinfo,
                                                           at::TensorList assigned_outputs) {
    TORCH_CHECK(assigned_outputs.size() == graphinfo.assigned_outputs.at_tensor_info.size(),
                "replay model internal error, please feedback bug.");
    for (size_t i = 0; i < assigned_outputs.size(); i++) {
        int64_t idx = graphinfo.assigned_outputs.mapping[i];
        size_t numel = NPUNativeFunctions::get_storage_size(assigned_outputs[i]);
        auto data_ptr = assigned_outputs[i].data_ptr();
        ge::TensorDesc tensor_desc = ATenGeBridge::InferGeTenosrDesc(
            graphinfo.assigned_outputs.at_tensor_info[i].storage_desc,
            graphinfo.assigned_outputs.graph_desc_info[i].graph_value.GetRealDtype());
        TORCH_CHECK(idx < graphinfo.graph_outputs_ge_tensors.size(),
                    "replay model internal error, please feedback bug.");
        graphinfo.graph_outputs_ge_tensors[idx] = ATenGeBridge::MakeGeTensor(tensor_desc, data_ptr,
                                                                             numel * assigned_outputs[i].itemsize());
    }

    std::vector<at::Tensor> returnable_outputs = SetOutputGeTensorAndSetReturnable(graphinfo,
                                                                                   graphinfo.returnable_outputs);

    if (this->retain_inner_output_) {
        graphinfo.inner_outputs_tensors.clear();
        graphinfo.inner_outputs_tensors = SetOutputGeTensorAndSetReturnable(graphinfo, graphinfo.inner_outputs);
        }

    return returnable_outputs;
}

std::vector<at::Tensor> ReplayGraphImpl::Replay(const at::TensorList& inputs, at::TensorList assigned_outputs) {
    TORCH_CHECK(!inputs.empty(), "Input tensorlist must have one tensor at least");
    auto input_tensor_shape = inputs.front().sizes();
    auto cache = replay_graph_cache_.find(multi_hash(input_tensor_shape));
    TORCH_CHECK(cache != replay_graph_cache_.end(), "The graph is not captured when replay");
    auto& graphinfo = cache->second;
    TORCH_CHECK(inputs.size() == graphinfo.inputs.at_tensor_info.size(),
                "Replay must have same num of inputs with generate graph");
    TORCH_CHECK(assigned_outputs.size() == graphinfo.assigned_outputs.at_tensor_info.size(),
                "Replay must have same num of assigned outputs with generate graph");

    SetInputGeTensor(graphinfo, inputs);
    auto returnable_outputs = SetOutputGeTensor(graphinfo, assigned_outputs);

    GraphExecutor::GetInstance().RunGraph(graphinfo.graph_id_, graphinfo.graph_inputs_ge_tensors,
        graphinfo.graph_outputs_ge_tensors);

    return returnable_outputs;
}

int64_t ReplayGraphImpl::FindMapping(const std::vector<int64_t>& graph_uid,
                                     const torch_npu::NpuGraphDesc& desc) {
    int64_t uid = desc.unique_id;
    for (size_t i = 0L; i < graph_uid.size(); i++) {
        if (uid == graph_uid[i]) {
            return i;
        }
    }
    return -1;
}

void ReplayGraphImpl::BuildReplayGraphInfo(const at::TensorList& tensors, AtTensorInfoAndMap& build_tensor_struct,
                                           CombinedInfo& combinedinfo) {
    const auto& unique_ids = combinedinfo.unique_ids;
    for (const auto& tensor : tensors) {
        torch_npu::NPUStorageDesc& storage_desc = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_;
        build_tensor_struct.at_tensor_info.emplace_back(TensorInfo(tensor.sizes().vec(),
            tensor.strides().vec(), tensor.storage_offset(), tensor.dtype(), storage_desc));
        torch_npu::NpuGraphDesc& graph_desc = torch_npu::NPUBridge::
                                    GetNpuStorageImpl(tensor)->get_mutable_npu_graph_desc();
        build_tensor_struct.graph_desc_info.emplace_back(graph_desc);
        build_tensor_struct.mapping.emplace_back(FindMapping(unique_ids, graph_desc));
    }
}

void ReplayGraphImpl::BuildReplayGraphInfoAll(const at::TensorList& inputs, at::TensorList assigned_outputs,
                                              at::TensorList returnable_outputs, CombinedInfo& input_infos,
                                              CombinedInfo& output_infos, ReplayGraphInfo& graphinfo) {
    BuildReplayGraphInfo(inputs, graphinfo.inputs, input_infos);
    BuildReplayGraphInfo(assigned_outputs, graphinfo.assigned_outputs, output_infos);
    BuildReplayGraphInfo(returnable_outputs, graphinfo.returnable_outputs, output_infos);
}

void ReplayGraphImpl::SetInnerOutput(CombinedInfo& outputcombinedinfo, ReplayGraphInfo& graphinfo) {
    std::vector<int64_t> id_mask;
    const auto& out_ids = outputcombinedinfo.unique_ids;
    id_mask.resize(out_ids.size());
    for (auto map_id : graphinfo.returnable_outputs.mapping) {
        id_mask[map_id] = 1;
    }

    auto full_output_storages = NpuGraphContextManager::GetInstance().
                                GetAllStorageOfLiveTensors(c10_npu::current_device());
    std::vector<c10::StorageImpl*> output_storages;
    for (const auto& s : full_output_storages) {
        if (!(GraphUtils::IsTensorWithoutNode(s) || GraphUtils::IsDataTensor(s))) {
            output_storages.emplace_back(s);
        }
    }

    for (size_t i = 0UL; i < id_mask.size(); i++) {
        if (id_mask[i] == 1) {
            continue;
        }
        const auto inner_id = out_ids[i];
        for (size_t storage_idx = 0; storage_idx < output_storages.size(); storage_idx++) {
            auto& graph_desc = torch_npu::NPUBridge::
                        GetNpuStorageImpl(output_storages[storage_idx])->get_mutable_npu_graph_desc();
            if (GraphUtils::IsTensorWithoutNode(output_storages[storage_idx])) {
                continue;
            }
            if (graph_desc.unique_id == inner_id) {
                const auto& storage_desc = torch_npu::NPUBridge::
                                GetNpuStorageImpl(output_storages[storage_idx])->get_npu_desc();
                std::vector<int64_t> sizes;
                for (const auto& size : storage_desc.base_sizes_) {
                    sizes.emplace_back(size);
                }
                std::vector<int64_t> strides;
                for (const auto& stride : storage_desc.base_strides_) {
                    strides.emplace_back(stride);
                }
                graphinfo.inner_outputs.mapping.emplace_back(i);
                graphinfo.inner_outputs.graph_desc_info.emplace_back(graph_desc);
                graphinfo.inner_outputs.at_tensor_info.emplace_back(TensorInfo(sizes, strides,
                    storage_desc.base_offset_, storage_desc.data_type_, storage_desc));
                break;
            }
        }
    }

    graphinfo.inner_outputs_tensors.clear();
    for (size_t i = 0; i < graphinfo.inner_outputs.at_tensor_info.size(); i++) {
        int64_t idx = graphinfo.inner_outputs.mapping[i];
        TORCH_CHECK(idx < output_storages.size(), "replay model internal error, please feedback bug.");
        c10::intrusive_ptr<c10::StorageImpl> storage_impl = c10::intrusive_ptr<c10::StorageImpl>::
                                                unsafe_reclaim_from_nonowning(output_storages[idx]);
        auto tensor = at::detail::make_tensor<torch_npu::NPUTensorImpl>(storage_impl, storage_impl,
                                                                    graphinfo.inner_outputs.at_tensor_info[i].dtype);
        graphinfo.inner_outputs_tensors.emplace_back(tensor);
    }
}

void ReplayGraphImpl::GetInputUniqueId(CombinedInfo& input_infos) {
    auto input_storages = NpuGraphContextManager::GetInstance().GetAllInputStorages(c10_npu::current_device());
    for (const auto& input_storage : input_storages) {
        torch_npu::NpuGraphDesc& graph_desc =
                torch_npu::NPUBridge::GetNpuStorageImpl(input_storage)->get_mutable_npu_graph_desc();
        auto data_node = graph_desc.graph_value.GetDataNode();
        if (data_node.value()->GetOpType() == "Data") {
            input_infos.unique_ids.emplace_back(graph_desc.unique_id);
        } else {
            input_infos.unique_ids.emplace_back(-1);
        }
    }
    return;
}

void ReplayGraphImpl::GetOutputUniqueId(CombinedInfo& output_infos) {
    auto output_storages = NpuGraphContextManager::GetInstance().GetAllStorageOfLiveTensors(c10_npu::current_device());
    for (const auto& output_storage : output_storages) {
        if (!(GraphUtils::IsTensorWithoutNode(output_storage) ||
            GraphUtils::IsDataTensor(output_storage))) {
                torch_npu::NpuGraphDesc& graph_desc =
                    torch_npu::NPUBridge::GetNpuStorageImpl(output_storage)->get_mutable_npu_graph_desc();
                output_infos.unique_ids.emplace_back(graph_desc.unique_id);
            }
    }
    return;
}

void ReplayGraphImpl::ClearGraphExecutor() {
    ScalarMemContext::GetContext().Reset();
    GraphExecutor::GetInstance().ResetGraphOutputs();
    GraphExecutor::GetInstance().RefreshGraphInputs();
    GraphExecutor::GetInstance().ClearDataStore();
}


void ReplayGraphImpl::GenerateGraph(const at::TensorList& inputs, at::TensorList assigned_outputs,
                                    at::TensorList returnable_outputs, bool retain_inner_output) {
    TORCH_CHECK(!inputs.empty(), "Input tensorlist must have one tensor at least");
    auto input_tensor_shape = inputs.front().sizes();
    auto key = multi_hash(input_tensor_shape);
    auto cache = replay_graph_cache_.find(key);
    ReplayGraphInfo info;
    if (cache == replay_graph_cache_.end()) {
        replay_graph_cache_.insert(std::make_pair(key, info));
    } else {
        replay_graph_cache_[key] = info;
    }
    auto& graphinfo = replay_graph_cache_[key];

    GraphExecutor::GetInstance().CheckDeviceIdAndInit();
    ScalarMemContext::GetContext().ExecuteH2D(c10_npu::getCurrentNPUStream());
    auto input_info = GraphExecutor::GetInstance().GetInputCombinedInfo();
    auto output_info = GraphExecutor::GetInstance().GetOutputCombinedInfo();
    GetInputUniqueId(input_info);
    GetOutputUniqueId(output_info);

    graphinfo.graph_inputs_ge_tensors = input_info.tensors;
    graphinfo.graph_outputs_ge_tensors = output_info.tensors;

    TORCH_CHECK(input_info.unique_ids.size() == graphinfo.graph_inputs_ge_tensors.size(),
                "replay model internal error, please feedback bug.");
    TORCH_CHECK(output_info.unique_ids.size() == graphinfo.graph_outputs_ge_tensors.size(),
                "replay model internal error, please feedback bug.");

    BuildReplayGraphInfoAll(inputs, assigned_outputs, returnable_outputs, input_info, output_info, graphinfo);

    bool is_cache_hit = false;
    graphinfo.graph_id_ = GraphExecutor::GetInstance().GetGraphIdDependOnCompileTypeAndCache(
        input_info, output_info, is_cache_hit);
    if (retain_inner_output) {
        this->retain_inner_output_ = retain_inner_output;
        SetInnerOutput(output_info, graphinfo);
    }

    ClearGraphExecutor();
    return;
}

std::vector<at::Tensor> ReplayGraphImpl::GetInnerOutputs(const at::TensorList& inputs) {
    TORCH_CHECK(!inputs.empty(), "Input tensorlist must have one tensor at least");
    auto input_tensor_shape = inputs.front().sizes();
    auto cache = replay_graph_cache_.find(multi_hash(input_tensor_shape));
    TORCH_CHECK(cache != replay_graph_cache_.end(), "The graph is not captured when replay");
    auto& graphinfo = cache->second;
    TORCH_CHECK(this->retain_inner_output_ , "Get inner outputs should set retain_inner_output as true");
    return graphinfo.inner_outputs_tensors;
}

std::vector<at::Tensor> ReplayGraph::Replay(const at::TensorList& inputs, at::TensorList assigned_outputs) {
    TORCH_CHECK(this->replay_graph_ != nullptr, "replay_graph_ == nullptr !")
    return this->replay_graph_->Replay(inputs, assigned_outputs);
}

void ReplayGraph::GenerateGraph(const at::TensorList& inputs, at::TensorList assigned_outputs,
                                at::TensorList returnable_outputs, bool retain_inner_output) {
    TORCH_CHECK(this->replay_graph_ != nullptr, "replay_graph_ == nullptr !");
    this->replay_graph_->GenerateGraph(inputs, assigned_outputs, returnable_outputs, retain_inner_output);
    return;
}

std::vector<at::Tensor> ReplayGraph::GetInnerOutputs(const at::TensorList& inputs) {
    TORCH_CHECK(this->replay_graph_ != nullptr, "replay_graph_ == nullptr !");
    return this->replay_graph_->GetInnerOutputs(inputs);
}

bool ReplayGraph::ReplayCacheHit(const at::TensorList& inputs) {
    TORCH_CHECK(this->replay_graph_ != nullptr, "replay_graph_ == nullptr !");
    return this->replay_graph_->ReplayCacheHit(inputs);
}
}
}