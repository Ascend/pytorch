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

#include "GraphExecutor.h"

#include <aten/src/ATen/Utils.h>
#include <aten/src/ATen/native/npu/graph/util/ATenGeBridge.h>
#include <aten/src/ATen/native/npu/graph/util/GraphUtils.h>
#include <c10/npu/NPUCachingAllocator.h>
#include <c10/npu/NPUFunctions.h>
#include <c10/npu/NPUGraphContextManager.h>
#include <c10/npu/NPUStream.h>
#include <c10/npu/register/OptionRegister.h>
#include <third_party/acl/inc/op_proto/array_ops.h>

#include <stack>
namespace at {
namespace native {
namespace npu {
namespace {
const char* const kPytorchGraphName = "PytorchGraph";
const std::string kDataNodeType = "Data";
const std::string kDataAttrIndex = "index";

static ge::Tensor MakeGeTensor(
    const ge::TensorDesc& tensor_desc,
    void* device_ptr,
    const size_t nbytes) {
  ge::Tensor ge_tensor{tensor_desc};
  ge_tensor.SetData(
      reinterpret_cast<uint8_t*>(device_ptr), nbytes, [](uint8_t* device_ptr) {
        return;
      });
  return ge_tensor;
}
} // namespace

uint32_t GraphExecutor::graph_id = 0;

void GraphExecutor::RunGraph(
    uint32_t graph_id,
    CombinedInfo& inputs,
    CombinedInfo& outputs) {
  aclrtStream cal_stream =
      const_cast<aclrtStream>(c10::npu::getCurrentNPUStream().stream());

  auto ret = GraphExecutor::GetInstance().session_->RunGraphWithStreamAsync(
      graph_id, cal_stream, inputs.tensors, outputs.tensors);
  TORCH_CHECK(ret == 0, "Run Graph Failed!");
}

void GraphExecutor::ConstructAndExecuteGraph() {
  if (GraphExecutor::GetInstance().session_ == nullptr) {
    GraphExecutor::GetInstance().Init();
  }

  TORCH_CHECK(session_ != nullptr, "Undefined session before run graph.");
  CombinedInfo inputs = GetInputCombinedInfo();
  CombinedInfo outputs = GetOutputCombinedInfo();
  if (outputs.nodes.empty()) {
    return;
  }

  uint32_t cur_graph_id = graph_id + 1;
  auto cached_graph_id = cacher_.GetCacheGraphId(
      inputs.hash_of_topo_and_attr,
      inputs.hash_of_shape,
      outputs.hash_of_topo_and_attr,
      outputs.hash_of_shape,
      cur_graph_id);

  if (!cached_graph_id.has_value()) {
    ConstructOps(outputs);
    ge::Graph graph(kPytorchGraphName);
    graph.SetInputs(GetInputOps()).SetOutputs(GetOutputOps());

    TORCH_CHECK(
        session_->AddGraph(cur_graph_id, graph) == 0, "AddGraph failed!");
  } else {
    cur_graph_id = cached_graph_id.value();
  }

  RunGraph(cur_graph_id, inputs, outputs);
  ResetGraphOutputs();
  if (!cached_graph_id.has_value()) {
    // Data of new graph maybe inputs of old graphs,
    // GE will change its attr
    // so we need to refresh it
    RefreshGraphInputs();
  }
  ClearDataStore();
  return;
}

void GraphExecutor::Init() {
  auto device_id = std::to_string(c10::npu::current_device());
  std::map<ge::AscendString, ge::AscendString> config = {
      {ge::AscendString(ge::OPTION_EXEC_DEVICE_ID),
       ge::AscendString(device_id.data())},
      {ge::AscendString(ge::OPTION_GRAPH_RUN_MODE), "1"},
      {ge::AscendString(ge::PRECISION_MODE.data()), "allow_fp32_to_fp16"},
  };

  static std::map<const std::string, const std::string>
      STRING_TO_COMPILE_OPT_MAP = {
          {"ACL_OP_DEBUG_LEVEL", ge::OP_DEBUG_LEVEL},
          {"ACL_DEBUG_DIR", ge::DEBUG_DIR},
          {"ACL_OP_COMPILER_CACHE_MODE", ge::OP_COMPILER_CACHE_MODE},
          {"ACL_OP_COMPILER_CACHE_DIR", ge::OP_COMPILER_CACHE_DIR},
      };

  for (const auto& iter : STRING_TO_COMPILE_OPT_MAP) {
    auto val = c10::npu::GetOption(iter.first);
    if (val.has_value()) {
      config.emplace(iter.second.c_str(), val.value().c_str());
    }
  }

  auto ret = ge::GEInitialize(config);
  if (ret != 0) {
    AT_ERROR("GE init failed!");
  }
  config["ge.session_device_id"] = ge::AscendString(device_id.data());
  session_.reset(new ge::Session(config));
  C10_NPU_CHECK(aclrtSetDevice(c10::npu::current_device()));
  if (session_ == nullptr) {
    AT_ERROR("Create session failed!");
  }
}

void GraphExecutor::Finalize() {
  if (GraphExecutor::GetInstance().session_ != nullptr) {
    session_.release();
    session_ = nullptr;
  }
}

void GraphExecutor::ConstructOps(CombinedInfo& output) {
  std::set<NodePtr> searched_nodes;
  for (const auto& output_node : output.nodes) {
    if (searched_nodes.find(output_node) != searched_nodes.end()) {
      continue;
    }
    searched_nodes.insert(output_node);
    std::stack<NodePtr> stack_node;
    stack_node.push(output_node);
    while (!stack_node.empty()) {
      auto top_node = stack_node.top();
      ATenGeBridge::CheckAndBuildGeOpForNode(top_node);
      stack_node.pop();
      const auto& inputs = top_node->GetInputs();
      for (const auto& input : inputs) {
        ATenGeBridge::CheckAndBuildGeOpForNode(top_node);
        top_node->GetGeOp()->SetInput(
            input.input_index,
            *(input.peer_output_node->GetGeOp()),
            input.peer_output_index);
        if (searched_nodes.find(input.peer_output_node) !=
            searched_nodes.end()) {
          continue;
        }
        stack_node.push(input.peer_output_node);
        searched_nodes.insert(input.peer_output_node);
      }
    }
  }
}

std::vector<ge::Operator> GraphExecutor::GetInputOps() {
  std::vector<ge::Operator> ops;
  auto npu_device = c10::npu::current_device();
  auto input_storages = c10::npu::graph::NpuGraphContextManager::GetInstance()
                            .GetAllInputStorages(npu_device);
  for (size_t index = 0; index < input_storages.size(); ++index) {
    auto data_node = input_storages[index]
                         ->get_mutable_npu_graph_desc()
                         .graph_value.GetDataNode();
    auto op_ptr = data_node.value()->GetGeOp();
    if (data_node.value()->GetOpType() == kDataNodeType) {
      if (op_ptr == nullptr) {
        data_node.value()->SetGeOp(std::make_shared<ge::op::Data>());
        op_ptr = data_node.value()->GetGeOp();
      }
      op_ptr->SetAttr(kDataAttrIndex, static_cast<uint32_t>(index));
    }
    ops.push_back(*op_ptr);
  }
  return ops;
}

GeOutPutOpType GraphExecutor::GetOutputOps() {
  GeOutPutOpType ops_and_idx;
  auto npu_device = c10::npu::current_device();
  auto output_storages = c10::npu::graph::NpuGraphContextManager::GetInstance()
                             .GetAllStorageOfLiveTensors(npu_device);
  for (auto& output_storage : output_storages) {
    if (GraphUtils::IsTensorWithoutNode(output_storage) ||
        GraphUtils::IsDataTensor(output_storage)) {
      continue;
    }
    const auto& graph_value =
        output_storage->get_mutable_npu_graph_desc().graph_value;
    auto op_ptr = graph_value.GetCurNode()->GetGeOp();
    ops_and_idx.emplace_back(
        *op_ptr, std::vector<size_t>{graph_value.GetValueIndex()});
  }
  return ops_and_idx;
}

CombinedInfo GraphExecutor::GetInputCombinedInfo() {
  CombinedInfo input_infos;
  auto npu_device = c10::npu::current_device();
  auto input_storages = c10::npu::graph::NpuGraphContextManager::GetInstance()
                            .GetAllInputStorages(npu_device);
  for (size_t index = 0; index < input_storages.size(); ++index) {
    NpuGraphDesc& graph_desc =
        input_storages[index]->get_mutable_npu_graph_desc();
    auto data_node = graph_desc.graph_value.GetDataNode();
    TORCH_CHECK(data_node.has_value(), "Inputs Tensor must have data node");
    ge::TensorDesc tensor_desc = ATenGeBridge::InferGeTenosrDesc(
        input_storages[index]->get_npu_desc(), input_storages[index]->dtype());

    if (data_node.value()->GetOpType() == kDataNodeType) {
      ge::Tensor ge_tensor =
          PrepareInputTensor(input_storages[index], tensor_desc);
      input_infos.tensors.push_back(std::move(ge_tensor));
    }
    hash_t topo_hash =
        GraphCache::GetTensorTopoHash(graph_desc.graph_value, tensor_desc);
    input_infos.hash_of_topo_and_attr.push_back(topo_hash);
    hash_t shape_hash = GraphCache::GetTensorShapeHash(topo_hash, tensor_desc);
    input_infos.hash_of_shape.push_back(shape_hash);
  }
  return input_infos;
}

CombinedInfo GraphExecutor::GetOutputCombinedInfo() {
  CombinedInfo output_infos;
  auto npu_device = c10::npu::current_device();
  auto output_storages = c10::npu::graph::NpuGraphContextManager::GetInstance()
                             .GetAllStorageOfLiveTensors(npu_device);
  for (auto& output_storage : output_storages) {
    if (GraphUtils::IsTensorWithoutNode(output_storage) ||
        GraphUtils::IsDataTensor(output_storage)) {
      continue;
    }
    auto& graph_value =
        output_storage->get_mutable_npu_graph_desc().graph_value;
    TORCH_CHECK(graph_value.HashNode(), "output must have node!");
    output_infos.nodes.push_back(graph_value.GetCurNode());
    ge::TensorDesc tensor_desc =
        ATenGeBridge::InferGeTenosrDesc(output_storage->get_npu_desc(),
        output_storage->dtype());
    auto ge_tensor = PrepareOutputTenosr(output_storage, tensor_desc);
    output_infos.tensors.push_back(std::move(ge_tensor));
    hash_t topo_hash = GraphCache::GetTensorTopoHash(graph_value, tensor_desc);
    output_infos.hash_of_topo_and_attr.emplace_back(topo_hash);

    hash_t shape_hash = GraphCache::GetTensorShapeHash(topo_hash, tensor_desc);
    output_infos.hash_of_topo_and_attr.push_back(shape_hash);
  }
  return output_infos;
}

ge::Tensor GraphExecutor::PrepareInputTensor(
    const c10::StorageImpl* const storage,
    const ge::TensorDesc& desc) {
  NpuGraphDesc& graph_desc = storage->get_mutable_npu_graph_desc();
  auto device_ptr = storage->data();
  size_t nbytes = storage->capacity();
  return MakeGeTensor(desc, device_ptr, nbytes);
}

ge::Tensor GraphExecutor::PrepareOutputTenosr(
    StorageImpl* storage,
    const ge::TensorDesc& desc) {
  NpuGraphDesc& graph_desc = storage->get_mutable_npu_graph_desc();
  TORCH_CHECK(
      graph_desc.graph_value.HashNode(),
      "graph desc in storage must have node");
  size_t nbytes = at::prod_intlist(storage->get_npu_desc().storage_sizes_) *
      storage->itemsize();
  DataPtr data_ptr;

  // In the case of in-place operator
  // we can not call set_data_ptr
  // for this will cause the old data ptr to be released
  if (!graph_desc.graph_value.GetDataNode().has_value()) {
    data_ptr = c10::npu::NPUCachingAllocator::get()->allocate(nbytes);
    storage->set_data_ptr(std::move(data_ptr));
  }
  return MakeGeTensor(desc, storage->data(), nbytes);
}

void GraphExecutor::ResetGraphOutputs() {
  auto npu_device = c10::npu::current_device();
  auto output_storages = c10::npu::graph::NpuGraphContextManager::GetInstance()
                             .GetAllStorageOfLiveTensors(npu_device);
  std::for_each(
      output_storages.begin(), output_storages.end(), [](StorageImpl* x) {
        if (!GraphUtils::IsTensorWithoutNode(x) ||
            !GraphUtils::IsDataTensor(x)) {
          GraphUtils::ResetOp(x);
        }
      });
}

void GraphExecutor::RefreshGraphInputs() {
  auto npu_device = c10::npu::current_device();
  auto input_storages = c10::npu::graph::NpuGraphContextManager::GetInstance()
                            .GetAllInputStorages(npu_device);
  std::for_each(
      input_storages.begin(), input_storages.end(), [&](StorageImpl* x) {
        GraphUtils::ResetOp(x);
      });
}

void GraphExecutor::ClearDataStore() {
  auto npu_device = c10::npu::current_device();
  c10::npu::graph::NpuGraphContextManager::GetInstance().EraseInputStorage(
      npu_device);
}
} // namespace npu
} // namespace native
} // namespace at