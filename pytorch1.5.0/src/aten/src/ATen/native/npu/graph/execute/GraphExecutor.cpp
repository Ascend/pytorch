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
#include <c10/npu/interface/AclInterface.h>
#include <c10/npu/NPUCachingAllocator.h>
#include <c10/npu/NPUFunctions.h>
#include <c10/npu/NPUGraphContextManager.h>
#include <c10/npu/register/OptionRegister.h>
#include <torch/csrc/autograd/record_function.h>
#include <ATen/native/npu/graph/scalar/ScalarMemoryOps.h>
#include <third_party/acl/inc/op_proto/array_ops.h>

#include <stack>
namespace at {
namespace native {
namespace npu {
namespace {
const char* kPytorchGraphName = "PytorchGraph";
const std::string kDataNodeType = "Data";
const char* kDataAttrIndex = "index";

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
  RECORD_HOST_FUNCTION("RunGraph", std::vector<c10::IValue>({}));
  aclrtStream cal_stream =
      const_cast<aclrtStream>(c10::npu::getCurrentNPUStream().stream());

  auto ret = session_->RunGraphWithStreamAsync(graph_id,
                                               cal_stream,
                                               inputs.tensors,
                                               outputs.tensors);
  TORCH_CHECK(ret == 0, "Run Graph Failed!");
}

void GraphExecutor::ConstructAndExecuteGraph() {
  RECORD_HOST_FUNCTION("ConstructAndExecuteGraph", std::vector<c10::IValue>({}));
  auto ret = CheckDeviceIdAndInit();
  if (!ret) {
    return;
  }
  TORCH_CHECK(session_ != nullptr, "Undefined session before run graph.");
  // before construct graph and tensor, do H2D copy for scalar.
  ScalarMemContext::GetContext().ExecuteH2D(c10::npu::getCurrentNPUStream());
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
    RECORD_HOST_FUNCTION("ConstructGraph", std::vector<c10::IValue>({}));
    ConstructOps(outputs);
    ge::Graph graph(kPytorchGraphName);
    graph.SetInputs(GetInputOps()).SetOutputs(GetOutputOps());

    TORCH_CHECK(
        session_->AddGraph(cur_graph_id, graph) == 0, "AddGraph failed!");
    graph_id = cur_graph_id;
  } else {
    cur_graph_id = cached_graph_id.value();
  }

  RunGraph(cur_graph_id, inputs, outputs);
  ScalarMemContext::GetContext().Reset();  
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
  auto device_id = std::to_string(init_device_id_);
  std::map<ge::AscendString, ge::AscendString> config = {
      {ge::AscendString(ge::OPTION_EXEC_DEVICE_ID),
       ge::AscendString(device_id.data())},
      {ge::AscendString(ge::OPTION_GRAPH_RUN_MODE), "0"},
      {ge::AscendString(ge::PRECISION_MODE.data()), "allow_fp32_to_fp16"},
      {ge::AscendString(ge::VARIABLE_MEMORY_MAX_SIZE), "1048576"}
  };

  static std::map<const std::string, const std::string>
      STRING_TO_COMPILE_OPT_MAP = {
          {"ACL_OP_DEBUG_LEVEL", ge::OP_DEBUG_LEVEL},
          {"ACL_DEBUG_DIR", ge::DEBUG_DIR},
          {"ACL_OP_COMPILER_CACHE_MODE", ge::OP_COMPILER_CACHE_MODE},
          {"ACL_OP_COMPILER_CACHE_DIR", ge::OP_COMPILER_CACHE_DIR},
          {"ACL_OP_SELECT_IMPL_MODE", ge::OP_SELECT_IMPL_MODE},
          {"ACL_OPTYPELIST_FOR_IMPLMODE", ge::OPTYPELIST_FOR_IMPLMODE}
      };

  for (const auto& iter : STRING_TO_COMPILE_OPT_MAP) {
    auto val = c10::npu::GetOption(iter.first);
    if (val.has_value() && (!val.value().empty())) {
      config.emplace(iter.second.data(), val.value().data());
    }
  }

  auto soc_name = c10::npu::acl::AclGetSocName();
  if (soc_name != nullptr) {
    config.emplace(ge::AscendString(ge::SOC_VERSION.data()), soc_name);
  }

  if (c10::npu::acl::IsExistQueryEventRecordedStatus()) {
    static const std::string HCOM_OPTIONS = "ge.exec.isUseHcom";
    config.emplace(HCOM_OPTIONS.data(), "1");
  }

  config["ge.session_device_id"] = ge::AscendString(device_id.data());
  config["ge.exec.reuseZeroCopyMemory"] = ge::AscendString("1");
  session_ = std::make_unique<ge::Session>(config);
  C10_NPU_CHECK(aclrtSetDevice(init_device_id_));
  if (session_ == nullptr) {
    AT_ERROR("Create session failed!");
  }
}

void GraphExecutor::Finalize() {
  if (GraphExecutor::GetInstance().session_ != nullptr) {
    session_.reset();
    session_ = nullptr;
  }
}

void GraphExecutor::ConstructOps(CombinedInfo& output) {
  RECORD_HOST_FUNCTION("ConstructOps", std::vector<c10::IValue>({}));
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
        ATenGeBridge::CheckAndBuildGeOpForNode(input.peer_output_node);
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
  auto input_storages = c10::npu::graph::NpuGraphContextManager::GetInstance()
                            .GetAllInputStorages(init_device_id_);
  for (size_t index = 0; index < input_storages.size(); ++index) {
    auto &graph_desc = input_storages[index]->get_mutable_npu_graph_desc();
    auto data_node = graph_desc.graph_value.GetDataNode();
    auto op_ptr = data_node.value()->GetGeOp();
    if (data_node.value()->GetOpType() == kDataNodeType) {
      if (op_ptr == nullptr) {
        data_node.value()->SetGeOp(std::make_shared<ge::op::Data>());
        op_ptr = data_node.value()->GetGeOp();
      }
      auto op_desc = ATenGeBridge::InferGeTenosrDesc(
          input_storages[index]->get_npu_desc(),
          input_storages[index]->dtype(),
          graph_desc.graph_value.GetRealDtype(),
          true);
      // x and y are the input and output names of Data IR
      op_ptr->UpdateInputDesc("x", op_desc);
      op_ptr->UpdateOutputDesc("y", op_desc);
      op_ptr->SetAttr(kDataAttrIndex, static_cast<uint32_t>(index));
    }
    ops.push_back(*op_ptr);
  }
  return ops;
}

GeOutPutOpType GraphExecutor::GetOutputOps() {
  GeOutPutOpType ops_and_idx;
  auto output_storages = c10::npu::graph::NpuGraphContextManager::GetInstance()
                             .GetAllStorageOfLiveTensors(init_device_id_);
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
  RECORD_HOST_FUNCTION("GetInputCombinedInfo", std::vector<c10::IValue>({}));
  CombinedInfo input_infos;
  auto input_storages = c10::npu::graph::NpuGraphContextManager::GetInstance()
                            .GetAllInputStorages(init_device_id_);
  for (size_t index = 0; index < input_storages.size(); ++index) {
    NpuGraphDesc& graph_desc =
        input_storages[index]->get_mutable_npu_graph_desc();
    auto data_node = graph_desc.graph_value.GetDataNode();
    TORCH_CHECK(data_node.has_value(), "Inputs Tensor must have data node");
    ge::TensorDesc tensor_desc = ATenGeBridge::InferGeTenosrDesc(
        input_storages[index]->get_npu_desc(),
        input_storages[index]->dtype(),
        graph_desc.graph_value.GetRealDtype());

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
  RECORD_HOST_FUNCTION("GetOutputCombinedInfo", std::vector<c10::IValue>({}));
  CombinedInfo output_infos;
  auto output_storages = c10::npu::graph::NpuGraphContextManager::GetInstance()
                             .GetAllStorageOfLiveTensors(init_device_id_);
  for (auto& output_storage : output_storages) {
    if (GraphUtils::IsTensorWithoutNode(output_storage) ||
        GraphUtils::IsDataTensor(output_storage)) {
      NpuGraphDesc graph_desc = output_storage->get_npu_graph_desc();
      // the tensor of scalar_merge_copy will enter here because is has't node,
      // only the length of the out queue is increased, nothing else.
      if ((output_storage->data() == nullptr) &&
          (!graph_desc.graph_value.GetScalarMemOffset().has_value())) {
        size_t nbytes = prod_intlist(output_storage->get_npu_desc().storage_sizes_) *
                        output_storage->itemsize();
        DataPtr data_ptr = c10::npu::NPUCachingAllocator::get()->allocate(nbytes);
        output_storage->set_data_ptr(std::move(data_ptr));
      }
      continue;
    }
    auto& graph_value =
        output_storage->get_mutable_npu_graph_desc().graph_value;
    TORCH_CHECK(graph_value.HashNode(), "output must have node!");
    output_infos.nodes.push_back(graph_value.GetCurNode());
    ge::TensorDesc tensor_desc = ATenGeBridge::InferGeTenosrDesc(
        output_storage->get_npu_desc(),
        output_storage->dtype(),
        graph_value.GetRealDtype());
    auto ge_tensor = PrepareOutputTenosr(output_storage, tensor_desc);
    output_infos.tensors.push_back(std::move(ge_tensor));
    hash_t topo_hash = GraphCache::GetTensorTopoHash(graph_value, tensor_desc);
    output_infos.hash_of_topo_and_attr.emplace_back(topo_hash);

    hash_t shape_hash = GraphCache::GetTensorShapeHash(topo_hash, tensor_desc);
    output_infos.hash_of_shape.push_back(shape_hash);
  }
  return output_infos;
}

ge::Tensor GraphExecutor::PrepareInputTensor(
    const c10::StorageImpl* const storage,
    const ge::TensorDesc& desc) {
  NpuGraphDesc& graph_desc = storage->get_mutable_npu_graph_desc();
  auto device_ptr = storage->data();
  size_t nbytes = storage->capacity();
  auto addr_offset = graph_desc.graph_value.GetScalarMemOffset();
  if (addr_offset.has_value()) {
    device_ptr = ScalarMemContext::GetContext().GetDeviceMemBuffer() + addr_offset.value();
  }
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
  // and if one value have data node which has no device memory
  // we should malloc for it
  if (!(graph_desc.graph_value.GetDataNode().has_value() &&
        storage->data() != nullptr)) {
    data_ptr = c10::npu::NPUCachingAllocator::get()->allocate(nbytes);
    storage->set_data_ptr(std::move(data_ptr));
  }
  return MakeGeTensor(desc, storage->data(), nbytes);
}

void GraphExecutor::ResetGraphOutputs() {
  RECORD_HOST_FUNCTION("ResetGraphOutputs", std::vector<c10::IValue>({}));
  auto output_storages = c10::npu::graph::NpuGraphContextManager::GetInstance()
                             .GetAllStorageOfLiveTensors(init_device_id_);
  std::for_each(
      output_storages.begin(), output_storages.end(), [](StorageImpl* x) {
        if (!GraphUtils::IsTensorWithoutNode(x) &&
            !GraphUtils::IsDataTensor(x)) {
          GraphUtils::ResetOp(x);
        }
      });
}

void GraphExecutor::RefreshGraphInputs() {
  RECORD_HOST_FUNCTION("RefreshGraphInputs", std::vector<c10::IValue>({}));
  auto input_storages = c10::npu::graph::NpuGraphContextManager::GetInstance()
                            .GetAllInputStorages(init_device_id_);
  std::for_each(
      input_storages.begin(), input_storages.end(), [&](StorageImpl* x) {
        GraphUtils::SetDataOp(x);
      });
}

void GraphExecutor::ClearDataStore() {
  RECORD_HOST_FUNCTION("ClearDataStore", std::vector<c10::IValue>({}));
  c10::npu::graph::NpuGraphContextManager::GetInstance().EraseInputStorage(
      init_device_id_);
}

bool GraphExecutor::CheckDeviceIdAndInit() {
  RECORD_HOST_FUNCTION("CheckDeviceIdAndInit", std::vector<c10::IValue>({}));
  auto devices_has_input =
      c10::npu::graph::NpuGraphContextManager::GetInstance()
          .GetDevicesHasLiveTensor();
  if (devices_has_input.empty()) {
    return false;
  } else if (devices_has_input.size() > 1) {
    AT_ERROR("In graph mode, you can not construct graph in different device");
  }

  init_device_id_ = devices_has_input.front();
  if (session_ == nullptr) {
    Init();
  }

  if (init_device_id_ != devices_has_input.front()) {
    AT_ERROR(
        "In graph mode, you can not change "
        "device id after first graph launch");
  }
  return true;
}

} // namespace npu
} // namespace native
} // namespace at
