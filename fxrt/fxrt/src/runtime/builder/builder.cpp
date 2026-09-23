#include "runtime/builder/builder.h"
#include <algorithm>
#include <iterator>
#include <unordered_set>
#include <memory>
#include <utility>
#include "runtime/executor/executor.h"
#include "ops/utils/op_support.h"
#include "ops/op_register.h"
#include "hardware/hardware_abstract/device_context_manager.h"
#include "hardware/hardware_abstract/collective/collective_manager.h"

namespace fxrt {
namespace runtime {

namespace {

hardware::Device GetOpDeviceType(const ir::NodePtr& opNode) {
  std::vector<ir::ValuePtr> inputValues;
  inputValues.reserve(opNode->inputs.size());
  std::transform(opNode->inputs.begin(), opNode->inputs.end(), std::back_inserter(inputValues), [](const auto& in) {
    return in != nullptr ? in->output : nullptr;
  });
  return GetDeviceFromOutputAndInputs(opNode->output, inputValues);
}

void VisitAllNodes(const ir::GraphPtr& graph, std::function<void(const ir::NodePtr&)> visitor) {
  for (auto& input : graph->inputs) {
    visitor(input);
  }
  for (auto& param : graph->parameters) {
    visitor(param);
  }
  for (auto& node : graph->nodes) {
    visitor(node);
  }
}
} // namespace

std::unique_ptr<Executor> Builder::BuildExecutor() {
  SetupOpRunners();
  return std::make_unique<Executor>(opRunners_, deviceContexts_, GetGraphOutput());
}

const ir::ValuePtr& Builder::GetGraphOutput() const {
  CHECK_IF_FAIL(graph_->nodes.size() > 0);
  auto& returnNode = graph_->nodes.back();
  CHECK_IF_NULL(returnNode);
  CHECK_IF_FAIL(returnNode->op == ops::Op_return);
  return returnNode->output;
}

void Builder::SetupOpRunners() {
  CreateOpRunners();
  UpdateRefNodeOutputValue();
  RecordTensorUpdatePoint();
  RecordStorageFreePoint();
}

void Builder::UpdateRefNodeOutputValue() {
  auto& opRunners = *opRunners_;
  for (auto& opRunner : opRunners) {
    opRunner.UpdateRefNodeOutputValue();
  }
}

void Builder::RecordTensorUpdatePoint() {
  if (graph_ == nullptr || graph_->nodes.empty()) {
    return;
  }

  std::unordered_set<ir::TensorPtr> inputTensors;
  for (auto& node : graph_->inputs) {
    ir::VisitAllTensors(node->output, [&](const ir::TensorPtr& tensor) { (void)inputTensors.insert(tensor); });
  }
  for (auto& node : graph_->parameters) {
    ir::VisitAllTensors(node->output, [&](const ir::TensorPtr& tensor) { (void)inputTensors.insert(tensor); });
  }

  std::unordered_map<ir::Node*, std::vector<ir::Tensor*>> tensorsToUpdate;
  for (auto& node : graph_->nodes) {
    if (IsSkipBuildOpRunner(node.get())) {
      continue;
    }

    // Tensors should be updated lazily right before its first consumer op.
    for (auto& inputNode : node->inputs) {
      ir::VisitAllTensors(inputNode->output, [&](const ir::TensorPtr& tensor) {
        if (inputTensors.find(tensor) == inputTensors.end()) {
          return;
        }
        (void)inputTensors.erase(tensor);
        (void)tensorsToUpdate[node.get()].emplace_back(tensor.get());
      });
    }
  }

  for (auto& item : tensorsToUpdate) {
    auto& node = item.first;
    auto& tensors = item.second;
    auto iter = nodeToOpRunner_.find(node);
    if (iter == nodeToOpRunner_.end()) {
      RT_GLOG(EXCEPTION) << "Can not find OpRunner for op: " << ops::ToStr(node->op);
    }
    auto* opRunner = iter->second;
    opRunner->SetTensorsToUpdate(std::move(tensors));
  }
}

void Builder::RecordStorageFreePoint() {
  if (graph_ == nullptr) {
    return;
  }

  // Step 1.
  // First, create a map to record the first node that sees each storage (the storage's owner)
  std::unordered_map<ir::Storage*, ir::Node*> storageToOwner; // key: storage pointer, value: owning node
                                                              // Visit all inputs and parameters in the graph

  VisitAllNodes(graph_, [&](const ir::NodePtr& node) {
    ir::VisitAllTensors(node->output, [&](const ir::TensorPtr& tensor) {
      auto storage = tensor->GetStorage().get();
      CHECK_IF_NULL(storage);
      (void)storageToOwner.insert({storage, node.get()});
    });
  });

  // Step 2.
  // The storages that can be safely freed once the last consumer node is done.
  std::unordered_map<ir::Node*, std::vector<ir::Storage*>> storagesToFree;
  // The storages that should be allocated by each node.
  std::unordered_map<ir::Node*, std::vector<ir::Storage*>> storagesToAlloc;
  std::unordered_set<ir::Storage*> recordedStorages;

  // The output of graph should not be freed by any node.
  auto& graphOutput = graph_->nodes.back()->output;
  ir::VisitAllTensors(graphOutput, [&](const ir::TensorPtr& tensor) {
    auto storage = tensor->GetStorage().get();
    CHECK_IF_NULL(storage);
    RT_VLOG(VL_RUNTIME) << "Record graph output Storage: " << storage;
    (void)recordedStorages.insert(storage);
  });

  // Traverse in reverse execution order
  for (auto iter = graph_->nodes.rbegin(); iter != graph_->nodes.rend(); ++iter) {
    RT_VLOG(VL_RUNTIME) << "Node: " << *iter;
    auto currentNode = iter->get();
    CHECK_IF_NULL(currentNode);
    if (IsSkipBuildOpRunner(currentNode)) {
      continue;
    }

    // Each op node is responsible for freeing the storage of its inputs.
    for (auto& inputNode : currentNode->inputs) {
      RT_VLOG(VL_RUNTIME) << "Input: " << inputNode;
      ir::VisitAllTensors(inputNode->output, [&](const ir::TensorPtr& tensor) {
        auto storage = tensor->GetStorage().get();
        CHECK_IF_NULL(storage);
        // No longer check storage->CheckOwnsData()
        // First encounter, meaning current node is the last consumer
        // and is responsible for freeing the storage.
        if (recordedStorages.find(storage) == recordedStorages.end()) {
          RT_VLOG(VL_RUNTIME) << "Record node input Storage: " << storage;
          (void)recordedStorages.insert(storage);
          auto storageToOwnerIter = storageToOwner.find(storage);
          CHECK_IF_FAIL(storageToOwnerIter != storageToOwner.end());
          auto* ownerNode = storageToOwnerIter->second;
          CHECK_IF_NULL(ownerNode);
          if (ownerNode->op != ops::Op_End) {
            (void)storagesToFree[currentNode].emplace_back(storage);
          }
        }
      });
    }

    // Current output should freed by itself if it is not freed by later nodes.
    ir::VisitAllTensors(currentNode->output, [&](const ir::TensorPtr& tensor) {
      auto storage = tensor->GetStorage().get();
      CHECK_IF_NULL(storage);
      if (recordedStorages.find(storage) == recordedStorages.end()) {
        RT_VLOG(VL_RUNTIME) << "Record node output Storage: " << storage;
        (void)recordedStorages.insert(storage);
        auto storageToOwnerIter = storageToOwner.find(storage);
        CHECK_IF_FAIL(storageToOwnerIter != storageToOwner.end());
        auto* ownerNode = storageToOwnerIter->second;
        CHECK_IF_NULL(ownerNode);
        if (ownerNode->op != ops::Op_End) {
          (void)storagesToFree[currentNode].emplace_back(storage);
        }
      }
    });

    // Check if this node owns any of its output storages and record them for allocation
    ir::VisitAllTensors(currentNode->output, [&](const ir::TensorPtr& tensor) {
      auto storage = tensor->GetStorage().get();
      CHECK_IF_NULL(storage);
      // If the storage's owner is the current node, record it for allocation
      if (storageToOwner[storage] == currentNode) {
        (void)storagesToAlloc[currentNode].emplace_back(storage);
      }
    });
  }

  // Step 3.
  for (auto& item : storagesToFree) {
    auto& node = item.first;
    auto& storages = item.second;
    auto iter = nodeToOpRunner_.find(node);
    if (iter == nodeToOpRunner_.end()) {
      RT_GLOG(EXCEPTION) << "Can not find OpRunner for op: " << ops::ToStr(node->op);
    }
    auto* opRunner = iter->second;
    opRunner->SetStoragesToFree(std::move(storages));
  }

  for (auto& item : storagesToAlloc) {
    auto& node = item.first;
    auto& storages = item.second;
    auto iter = nodeToOpRunner_.find(node);
    if (iter == nodeToOpRunner_.end()) {
      RT_GLOG(EXCEPTION) << "Can not find OpRunner for op: " << ops::ToStr(node->op);
    }
    auto* opRunner = iter->second;
    opRunner->SetStoragesToAlloc(std::move(storages));
  }
}

void Builder::CreateOpRunners() {
  const size_t nodeNum = graph_->nodes.size();
  opRunners_ = std::make_shared<std::vector<OpRunner>>();
  opRunners_->reserve(nodeNum);
  for (auto& node : graph_->nodes) {
    CHECK_IF_NULL(node);
    if (IsSkipBuildOpRunner(node)) {
      continue;
    }

    auto device = GetOpDeviceType(node);
    auto operatorPtr = ops::CreateOperator(ops::ToStr(node->op), device.type);
    if (operatorPtr == nullptr) {
      RT_GLOG(EXCEPTION) << "Create operator for: " << ops::ToStr(node->op)
                         << " failed, please register it on platform: " << hardware::GetDeviceNameByType(device.type);
    }
    std::vector<const ir::Value*> inputs;
    for (auto& input : node->inputs) {
      const ir::Value* value = input != nullptr ? input->output.get() : nullptr;
      (void)inputs.emplace_back(value);
    }
    operatorPtr->Init(inputs, node->output.get());

    device::DeviceContext* deviceContext;
    if (auto iter = deviceContexts_.find(device.type); iter != deviceContexts_.end()) {
      deviceContext = iter->second;
    } else {
      auto deviceId = fxrt::collective::CollectiveManager::Instance().local_rank_id();
      device::DeviceContextKey deviceContextKey = {hardware::GetDeviceNameByType(device.type), deviceId};
      deviceContext = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(deviceContextKey);
      CHECK_IF_NULL(deviceContext);
      (void)deviceContexts_.emplace(device.type, deviceContext);
    }
    void* stream = deviceContext->deviceResManager_->GetCurrentStream();
    if (device.type != hardware::DeviceType::CPU) {
      CHECK_IF_NULL(stream);
    }
    // TODO: need to support ascend stream creation and getting real dynamic shape info.  // NOLINT(readability/todo)
    (void)(opRunners_->emplace_back(
        node->op,
        node->inputs,
        node->output,
        std::move(operatorPtr),
        stream,
        device,
        deviceContext,
        true /*isDynamicShape*/));
    nodeToOpRunner_.emplace(node.get(), &(opRunners_->back()));
  }
}
} // namespace runtime
} // namespace fxrt
