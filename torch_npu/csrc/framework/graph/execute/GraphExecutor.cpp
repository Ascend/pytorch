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
#include <Python.h>
#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include "torch_npu/csrc/framework/graph/util/ATenGeBridge.h"
#include "torch_npu/csrc/framework/graph/util/GraphUtils.h"
#include "torch_npu/csrc/framework/interface/AclInterface.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include <torch_npu/csrc/framework/graph/util/NPUGraphContextManager.h>
#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "torch_npu/csrc/framework/graph/scalar/ScalarMemoryOps.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include <third_party/acl/inc/graph/operator_reg.h>
#include <stack>

// wait RECORD_HOST_FUNCTION to be added into plugin
#define RECORD_HOST_FUNCTION(a, b) ;
namespace at_npu {
namespace native {
namespace {
const char* kPytorchGraphName = "PytorchGraph";
const std::string kDataNodeType = "Data";
const char* kDataAttrIndex = "index";
const std::string kEnqueNodeType = "OutfeedEnqueueOpV2";

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
    const std::vector<ge::Tensor>& inputs,
    std::vector<ge::Tensor>& outputs) {
  RECORD_FUNCTION("RunGraph", std::vector<c10::IValue>({}));
  aclrtStream cal_stream =
      const_cast<aclrtStream>(c10_npu::getCurrentNPUStream().stream());

  auto start_time = std::chrono::steady_clock::now();
  NPU_CHECK_ERROR(session_->RunGraphWithStreamAsync(graph_id,
                                                    cal_stream,
                                                    inputs,
                                                    outputs));
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - start_time);
  if (verbose_) {
    ASCEND_LOGI("RunGraph Time: duration = %.3f ms", static_cast<double>(duration.count()) *
                                                 std::chrono::microseconds::period::num /
                                                 std::chrono::milliseconds::period::den);
  }
}

void GraphExecutor::ConstructAndExecuteGraph() {
  RECORD_FUNCTION("ConstructAndExecuteGraph", std::vector<c10::IValue>({}));
  auto ret = CheckDeviceIdAndInit();
  if (!ret) {
    return;
  }
  TORCH_CHECK(session_ != nullptr, "Undefined session before run graph.");
  // before construct graph and tensor, do H2D copy for scalar.
  ScalarMemContext::GetContext().ExecuteH2D(c10_npu::getCurrentNPUStream());
  CombinedInfo inputs = GetInputCombinedInfo();
  CombinedInfo outputs = GetOutputCombinedInfo();
  if ((outputs.nodes.empty()) && (outputs.none_output_nodes.empty())) {
    return;
  }

  bool is_cache_hit = false;
  auto cur_graph_id = GetGraphIdDependOnCompileTypeAndCache(inputs, outputs, is_cache_hit);

  size_t input_number = inputs.tensors.size();
  size_t output_number = outputs.tensors.size();
  if (verbose_) {
    string is_cache = is_cache_hit ? "true" : "false";
    ASCEND_LOGI("Using Graph Mode: current graph id = %u, cache hit = %s,"
                "input number = %zu, output number = %zu",
                cur_graph_id, is_cache.c_str(), input_number, output_number);
  }
  // Release GIL to avoid deadlocks.
  if (PyGILState_Check()) {
    Py_BEGIN_ALLOW_THREADS
    RunGraph(cur_graph_id, inputs.tensors, outputs.tensors);
    Py_END_ALLOW_THREADS
  } else {
    RunGraph(cur_graph_id, inputs.tensors, outputs.tensors);
  }
  ScalarMemContext::GetContext().Reset();
  ResetGraphOutputs();
  if (!is_cache_hit) {
    // Data of new graph maybe inputs of old graphs,
    // GE will change its attr
    // so we need to refresh it
    RefreshGraphInputs();
  }
  ClearDataStore();
  return;
}

uint32_t GraphExecutor::GetGraphIdDependOnCompileTypeAndCache(const CombinedInfo& inputs,
                                                              CombinedInfo& outputs,
                                                              bool& is_cache_hit) {
  uint32_t cur_graph_id = graph_id + 1;
  auto cached_graph_id = cacher_.GetCacheGraphId(
      inputs.hash_of_topo_and_attr,
      inputs.hash_of_shape,
      outputs.hash_of_topo_and_attr,
      outputs.hash_of_shape,
      cur_graph_id);
  if (!cached_graph_id.has_value()) {
    RECORD_FUNCTION("ConstructGraph", std::vector<c10::IValue>({}));
    std::vector<ge::Operator> const_input_ops;
    ConstructOpsAndAddEdge(outputs, const_input_ops);
    ge::Graph graph("PytorchGraph");
    std::vector<ge::Operator> input_ops = GetInputOps();
    input_ops.insert(input_ops.end(),
                     const_input_ops.begin(),
                     const_input_ops.end());
    graph.SetInputs(input_ops).SetOutputs(GetOutputOps());

    NPU_CHECK_ERROR(session_->AddGraph(cur_graph_id, graph));
    graph_id = cur_graph_id;
  } else {
    cur_graph_id = cached_graph_id.value();
  }
  is_cache_hit = cached_graph_id.has_value();
  return cur_graph_id;
}

void GraphExecutor::Init() {
  auto device_id = std::to_string(init_device_id_);
  std::map<ge::AscendString, ge::AscendString> config = {
      {ge::AscendString(ge::OPTION_EXEC_DEVICE_ID),
       ge::AscendString(device_id.data())},
      {ge::AscendString(ge::OPTION_GRAPH_RUN_MODE), "0"},
      {ge::AscendString(ge::PRECISION_MODE.data()), "allow_fp32_to_fp16"},
      {ge::AscendString(ge::VARIABLE_MEMORY_MAX_SIZE), "1048576"},
      {ge::AscendString(ge::OPTION_TOPOSORTING_MODE), "2"}
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
    auto val = c10_npu::option::GetOption(iter.first);
    if (val.has_value() && (!val.value().empty())) {
      config.emplace(iter.second.data(), val.value().data());
    }
  }

  auto soc_name = c10_npu::acl::AclGetSocName();
  if (soc_name != nullptr) {
    config.emplace(ge::AscendString(ge::SOC_VERSION.data()), soc_name);
  }

  if (c10_npu::acl::IsExistQueryEventRecordedStatus()) {
    static const std::string HCOM_OPTIONS = "ge.exec.isUseHcom";
    config.emplace(HCOM_OPTIONS.data(), "1");
  }

  config["ge.session_device_id"] = ge::AscendString(device_id.data());
  config["ge.exec.reuseZeroCopyMemory"] = ge::AscendString("1");
  config["GE_USE_STATIC_MEMORY"] = ge::AscendString("2");
  session_ = std::make_unique<ge::Session>(config);
  NPU_CHECK_ERROR(aclrtSetDevice(init_device_id_));
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

void GraphExecutor::ConstructOpsAndAddEdge(
    const CombinedInfo& output,
    std::vector<ge::Operator>& const_input_ops) {
  RECORD_FUNCTION("ConstructOpsAndAddEdge", std::vector<c10::IValue>({}));

  std::vector<NodePtr> out_nodes = output.nodes;
  const std::vector<NodePtr>& none_output_nodes = output.none_output_nodes;

  NodePtr front_enque = nullptr;
  for (auto& node_ptr : none_output_nodes) {
    if (node_ptr->GetOpType() == kEnqueNodeType) {
      ATenGeBridge::CheckAndBuildGeOpForNode(node_ptr, const_input_ops);
      if (front_enque != nullptr) {
        node_ptr->GetGeOp()->AddControlInput(*(front_enque->GetGeOp()));
      }
      front_enque = node_ptr;
    }
    out_nodes.emplace_back(node_ptr);
  }

  std::set<NodePtr> searched_nodes;
  for (const auto& output_node : out_nodes) {
    if (searched_nodes.find(output_node) != searched_nodes.end()) {
      continue;
    }
    searched_nodes.insert(output_node);
    std::stack<NodePtr> stack_node;
    stack_node.push(output_node);
    while (!stack_node.empty()) {
      auto top_node = stack_node.top();
      ATenGeBridge::CheckAndBuildGeOpForNode(top_node, const_input_ops);
      stack_node.pop();
      const auto& inputs = top_node->GetInputs();
      for (const auto& input : inputs) {
        ATenGeBridge::CheckAndBuildGeOpForNode(input.peer_output_node,
                                               const_input_ops);
        top_node->GetGeOp()->SetInput(
            input.input_index,
            *(input.peer_output_node->GetGeOp()),
            input.peer_output_index);
        /*
        *        public_input
        *           /     \
        *          /       \
        *         /         \
        *       op ---------> inplace_op
        *         control_edge
        *
        *  Add control edges to ensure that
        *  the operators will be executed in the correct order
        *  in inplace-related cases.
        */
        auto inplace_node_ptr =
          input.peer_output_node->GetInplaceNode(input.peer_output_index);
        if ((inplace_node_ptr.has_value()) &&
            (!inplace_node_ptr.value().expired()) &&
            (!top_node->IsInplace())) {
          auto inplace_node = inplace_node_ptr.value().lock();
          if (inplace_node != nullptr) {
            ATenGeBridge::CheckAndBuildGeOpForNode(inplace_node,
                                                   const_input_ops);
            inplace_node->GetGeOp()->AddControlInput(*(top_node->GetGeOp()));
          }
        }
        
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
  auto input_storages = NpuGraphContextManager::GetInstance()
                            .GetAllInputStorages(init_device_id_);
  for (size_t index = 0; index < input_storages.size(); ++index) {
    auto &graph_desc = torch_npu::NPUBridge::GetNpuStorageImpl(input_storages[index])->get_mutable_npu_graph_desc();
    auto data_node = graph_desc.graph_value.GetDataNode();
    auto op_ptr = data_node.value()->GetGeOp();
    if (data_node.value()->GetOpType() == "Data") {
      if (op_ptr == nullptr) {
        data_node.value()->SetGeOp(std::make_shared<ge::Operator>());
        op_ptr = data_node.value()->GetGeOp();
      }
      auto op_desc = ATenGeBridge::InferGeTenosrDesc(
          torch_npu::NPUBridge::GetNpuStorageImpl(input_storages[index])->get_npu_desc(),
          graph_desc.graph_value.GetRealDtype(),
          true);
      // x and y are the input and output names of Data IR
      op_ptr->UpdateInputDesc("x", op_desc);
      op_ptr->UpdateOutputDesc("y", op_desc);
      op_ptr->SetAttr("index", static_cast<uint32_t>(index));
    }
    ops.push_back(*op_ptr);
  }
  return ops;
}

GeOutPutOpType GraphExecutor::GetOutputOps() {
  GeOutPutOpType ops_and_idx;
  auto output_storages = NpuGraphContextManager::GetInstance()
                             .GetAllStorageOfLiveTensors(init_device_id_);
  for (auto& output_storage : output_storages) {
    if (GraphUtils::IsTensorWithoutNode(output_storage) ||
        GraphUtils::IsDataTensor(output_storage)) {
      continue;
    }
    const auto& graph_value =
        torch_npu::NPUBridge::GetNpuStorageImpl(output_storage)->get_mutable_npu_graph_desc().graph_value;
    auto op_ptr = graph_value.GetCurNode()->GetGeOp();
    ops_and_idx.emplace_back(
        *op_ptr, std::vector<size_t>{graph_value.GetValueIndex()});
  }
  return ops_and_idx;
}

CombinedInfo GraphExecutor::GetInputCombinedInfo() {
  RECORD_FUNCTION("GetInputCombinedInfo", std::vector<c10::IValue>({}));
  CombinedInfo input_infos;
  auto input_storages = NpuGraphContextManager::GetInstance()
                            .GetAllInputStorages(init_device_id_);
  for (size_t index = 0; index < input_storages.size(); ++index) {
    torch_npu::NpuGraphDesc& graph_desc =
        torch_npu::NPUBridge::GetNpuStorageImpl(input_storages[index])->get_mutable_npu_graph_desc();
    auto data_node = graph_desc.graph_value.GetDataNode();
    TORCH_CHECK(data_node.has_value(), "Inputs Tensor must have data node");
    ge::TensorDesc tensor_desc = ATenGeBridge::InferGeTenosrDesc(
        torch_npu::NPUBridge::GetNpuStorageImpl(input_storages[index])->get_npu_desc(),
        graph_desc.graph_value.GetRealDtype());

    if (data_node.value()->GetOpType() == "Data") {
      size_t tensor_capacity = GraphUtils::GetTensorCapacity(input_storages[index]);
      ge::Tensor ge_tensor =
          PrepareInputTensor(input_storages[index], tensor_desc, tensor_capacity);
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
  RECORD_FUNCTION("GetOutputCombinedInfo", std::vector<c10::IValue>({}));
  CombinedInfo output_infos;
  auto output_storages = NpuGraphContextManager::GetInstance()
                             .GetAllStorageOfLiveTensors(init_device_id_);
  for (auto& output_storage : output_storages) {
    if (GraphUtils::IsTensorWithoutNode(output_storage) ||
        GraphUtils::IsDataTensor(output_storage)) {
      torch_npu::NpuGraphDesc graph_desc = torch_npu::NPUBridge::GetNpuStorageImpl(output_storage)->get_npu_graph_desc();
      // the tensor of scalar_merge_copy will enter here because is has't node,
      // only the length of the out queue is increased, nothing else.
      if ((output_storage->data() == nullptr) &&
          (!graph_desc.graph_value.GetScalarMemOffset().has_value())) {
        size_t nbytes = GraphUtils::GetTensorCapacity(output_storage);
        GraphUtils::SetDataPtrAndNbytes(output_storage, nbytes);
      }
      continue;
    }
    auto& graph_value =
        torch_npu::NPUBridge::GetNpuStorageImpl(output_storage)->get_mutable_npu_graph_desc().graph_value;
    TORCH_CHECK(graph_value.HashNode(), "output must have node!");
    output_infos.nodes.push_back(graph_value.GetCurNode());
    ge::TensorDesc tensor_desc = ATenGeBridge::InferGeTenosrDesc(
        torch_npu::NPUBridge::GetNpuStorageImpl(output_storage)->get_npu_desc(),
        graph_value.GetRealDtype());
    auto ge_tensor = PrepareOutputTenosr(output_storage, tensor_desc);
    output_infos.tensors.push_back(std::move(ge_tensor));
    hash_t topo_hash = GraphCache::GetTensorTopoHash(graph_value, tensor_desc);
    output_infos.hash_of_topo_and_attr.emplace_back(topo_hash);

    hash_t shape_hash = GraphCache::GetTensorShapeHash(topo_hash, tensor_desc);
    output_infos.hash_of_shape.push_back(shape_hash);
  }

  std::vector<NodePtr> none_output_nodes =
    NpuGraphContextManager::GetInstance().
    GetNoneOutputNode(init_device_id_);
  for (auto& node_ptr : none_output_nodes) {
    output_infos.none_output_nodes.emplace_back(node_ptr);
    output_infos.hash_of_topo_and_attr.emplace_back(node_ptr->GetNodeHash());
  }
  return output_infos;
}

ge::Tensor GraphExecutor::PrepareInputTensor(
    const c10::StorageImpl* const storage,
    const ge::TensorDesc& desc,
    size_t capacity) {
  torch_npu::NpuGraphDesc& graph_desc = torch_npu::NPUBridge::GetNpuStorageImpl(const_cast<c10::StorageImpl*>(storage))->get_mutable_npu_graph_desc();
  auto device_ptr = storage->data();
  size_t nbytes = capacity;
  auto addr_offset = graph_desc.graph_value.GetScalarMemOffset();
  if (addr_offset.has_value()) {
    device_ptr = ScalarMemContext::GetContext().GetDeviceMemBuffer() + addr_offset.value();
  }
  return ATenGeBridge::MakeGeTensor(desc, device_ptr, nbytes);
}

ge::Tensor GraphExecutor::PrepareOutputTenosr(
    c10::StorageImpl* storage,
    const ge::TensorDesc& desc) {
  torch_npu::NpuGraphDesc& graph_desc = torch_npu::NPUBridge::GetNpuStorageImpl(storage)->get_mutable_npu_graph_desc();
  TORCH_CHECK(
      graph_desc.graph_value.HashNode(),
      "graph desc in storage must have node");
  size_t nbytes = GraphUtils::GetTensorCapacity(storage);

  // In the case of in-place operator
  // we can not call set_data_ptr
  // for this will cause the old data ptr to be released
  // and if one value have data node which has no device memory
  // we should malloc for it
  // After decoupling, we cannot simply set nbytes for NPUStorageImpl
  // by calling set_data_ptr. Instead, we need to call set_nbytes
  if (!(graph_desc.graph_value.GetDataNode().has_value() &&
        storage->data() != nullptr)) {
    GraphUtils::SetDataPtrAndNbytes(storage, nbytes);
  }
  return ATenGeBridge::MakeGeTensor(desc, storage->data(), nbytes);
}

void GraphExecutor::ResetGraphOutputs() {
  RECORD_FUNCTION("ResetGraphOutputs", std::vector<c10::IValue>({}));
  auto output_storages = NpuGraphContextManager::GetInstance()
                             .GetAllStorageOfLiveTensors(init_device_id_);
  std::for_each(
      output_storages.begin(), output_storages.end(), [](c10::StorageImpl* x) {
        if (!GraphUtils::IsTensorWithoutNode(x) &&
            !GraphUtils::IsDataTensor(x)) {
          GraphUtils::ResetOp(x);
        }
      });
  NpuGraphContextManager::GetInstance().EraseNoneOutputNode(init_device_id_);
}

void GraphExecutor::RefreshGraphInputs() {
  RECORD_FUNCTION("RefreshGraphInputs", std::vector<c10::IValue>({}));
  auto input_storages = NpuGraphContextManager::GetInstance()
                            .GetAllInputStorages(init_device_id_);
  std::for_each(
      input_storages.begin(), input_storages.end(), [&](c10::StorageImpl* x) {
        GraphUtils::SetDataOp(x);
      });
}

void GraphExecutor::ClearDataStore() {
  RECORD_FUNCTION("ClearDataStore", std::vector<c10::IValue>({}));
  NpuGraphContextManager::GetInstance().EraseInputStorage(
      init_device_id_);
}

bool GraphExecutor::CheckDeviceIdAndInit() {
  RECORD_FUNCTION("CheckDeviceIdAndInit", std::vector<c10::IValue>({}));
  auto devices_has_input =
      NpuGraphContextManager::GetInstance()
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
} // namespace native
} // namespace at_npu

