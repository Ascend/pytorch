#include "runtime/executor/executor.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ops/kernel_lib.h"
#include "ops/op_def/ops_name.h"
#include "runtime/executor/ir_graph_export.h"
#include "ops/utils/async.h"
#include "runtime/builder/builder.h"
#include "runtime/builder/pipeline/pipeline_builder.h"
#include "runtime/builder/kernel_launch_group/kernel_launch_group_builder.h"
#include "runtime/builder/kernel_capture/kernel_capture_builder.h"
#include "runtime/utils/utils.h"

namespace fxrt {
namespace runtime {
using BuilderCreationFunc = std::function<std::unique_ptr<Builder>(const ir::GraphPtr&)>;
static std::vector<BuilderCreationFunc> builderCreators = {
    [](const ir::GraphPtr& graph) { return std::make_unique<Builder>(graph); },
    [](const ir::GraphPtr& graph) { return std::make_unique<PipelineBuilder>(graph); },
    [](const ir::GraphPtr& graph) { return std::make_unique<KernelLaunchGroupBuilder>(graph); },
    [](const ir::GraphPtr& graph) { return std::make_unique<KernelCaptureBuilder>(graph); }}; // namespace runtime

namespace {
ExecutionMode GetExecutionMode() {
  static const char kernelLaunchGroupNum[] = "FXRT_KERNEL_LAUNCH_GROUP_NUM";
  const char* enableGroupLaunchCStr = std::getenv(kernelLaunchGroupNum);
  const bool enableGroupLaunch = (enableGroupLaunchCStr != nullptr) && !std::string_view(enableGroupLaunchCStr).empty();
  const bool enableAclGraph = IsAclGraphEnabled();

  ExecutionMode executionMode = Base;
  if (ops::IsEnablePipeline()) {
    executionMode = Pipeline;
  }
  if (enableGroupLaunch) {
    executionMode = GroupLaunch;
  }
  if (enableAclGraph) {
    executionMode = AclGraph;
  }
  return executionMode;
}

const std::vector<std::string> GetEnvKernelLibPaths() {
  std::vector<std::string> kernelLibPaths{};
  constexpr char kKernelLibPathsEnvName[] = "DART_KERNEL_LIB_PATH";
  const char* pathsCStr = std::getenv(kKernelLibPathsEnvName);
  if (pathsCStr == nullptr) {
    return kernelLibPaths;
  }

  size_t pathLen = 0;
  while (pathsCStr[pathLen] != '\0') {
    if (pathsCStr[pathLen] == ',') {
      (void)kernelLibPaths.emplace_back(std::string(pathsCStr, pathLen));
      pathsCStr += pathLen + 1;
      pathLen = 0;
    } else {
      ++pathLen;
    }
  }
  (void)kernelLibPaths.emplace_back(std::string(pathsCStr, pathLen));
  return kernelLibPaths;
}

void ProcessMakeTuple(ir::NodePtr node) {
  CHECK_IF_NULL(node);
  std::vector<ir::ValuePtr> elements;
  for (auto& input : node->inputs) {
    (void)elements.emplace_back(input->output);
  }
  node->output = ir::MakeIntrusive<ir::Value>(ir::MakeIntrusive<ir::Tuple>(std::move(elements)));
}

void ProcessTupleGetItem(ir::NodePtr node) {
  CHECK_IF_NULL(node);
  auto index = node->inputs[kSecondInput]->output->ToInt();
  auto tuple = node->inputs[kFirstInput]->output->ToTuple();
  CHECK_IF_FAIL(static_cast<size_t>(index) < tuple->Size());
  node->output = (*tuple)[index];
}
} // namespace

GraphExecutor::GraphExecutor() {
  for (auto&& path : GetEnvKernelLibPaths()) {
    ops::KernelLibRegistry::Instance().Load(path);
  }
}

GraphExecutor::~GraphExecutor() {
  for (auto& kernelPair : kernels_) {
    CHECK_IF_NULL(kernelPair.second);
    delete kernelPair.second;
  }
}

// Start building graph.
void GraphExecutor::BeginGraph(const std::string& name) {
  RT_VLOG(VL_RUNTIME) << "Begin graph building";
  CHECK_IF_FAIL(graph_ == nullptr);
  graph_ = ir::MakeIntrusive<ir::Graph>();
  name_ = name;
}

// Finish building graph.
void GraphExecutor::EndGraph() {
  RT_VLOG(VL_RUNTIME) << "End graph building";
  CHECK_IF_NULL(graph_);
}

void GraphExecutor::OptGraph() {
  RT_VLOG(VL_RUNTIME) << "Opt graph";
  CHECK_IF_NULL(graph_);
  // clang-format off
  pass::TensorCreator tensorCreator =
    std::bind((ir::NodePtr(GraphExecutor::*)(ops::Op, const std::vector<ir::NodePtr> &, const ir::ValuePtr &)) &
                GraphExecutor::AddOpNode,
              this, std::placeholders::_1, std::placeholders::_2, nullptr);
  pass::PassManager::Instance().Run(graph_, tensorCreator);
}

// Add a parameter node for graph.
ir::NodePtr GraphExecutor::AddParameterNode(const ir::ValuePtr &value) {
  RT_VLOG(VL_RUNTIME) << "Add parameter node: " << value;
  auto node = ir::MakeIntrusive<ir::Node>();
  node->op = ops::Op_End;
  node->output = value == nullptr ? ir::MakeIntrusive<ir::Value>() : value;
  CHECK_IF_NULL(graph_);
  (void)graph_->parameters.emplace_back(node);
  return node;
}

// Add an input node for graph.
ir::NodePtr GraphExecutor::AddInputNode(const ir::ValuePtr &value) {
  RT_VLOG(VL_RUNTIME) << "Add input node: " << value;
  auto node = ir::MakeIntrusive<ir::Node>();
  node->op = ops::Op_End;
  node->output = value == nullptr ? ir::MakeIntrusive<ir::Value>() : value;
  CHECK_IF_NULL(graph_);
  (void)graph_->inputs.emplace_back(node);
  return node;
}

// Add a value node.
ir::NodePtr GraphExecutor::AddValueNode(const ir::ValuePtr &value) {
  RT_VLOG(VL_RUNTIME) << "Add value node: " << value;
  auto node = ir::MakeIntrusive<ir::Node>();
  node->op = ops::Op_End;
  node->output = value == nullptr ? ir::MakeIntrusive<ir::Value>() : value;
  CHECK_IF_NULL(graph_);
  (void)graph_->nodes.emplace_back(node);
  return node;
}

// Add an operation node.
ir::NodePtr GraphExecutor::AddOpNode(ops::Op op, const std::vector<ir::NodePtr> &inputs, const ir::ValuePtr &output) {
  RT_VLOG(VL_RUNTIME) << "Add operation node";
  RT_VLOG(VL_RUNTIME) << "operation input size: " << inputs.size();
  auto node = ir::MakeIntrusive<ir::Node>();
  CHECK_IF_NULL(node);
  node->op = op;
  node->inputs = inputs;
  node->output = output == nullptr ? ir::MakeIntrusive<ir::Value>() : output;
  CHECK_IF_NULL(graph_);
  (void)graph_->nodes.emplace_back(node);
  return node;
}

// Add return node.
void GraphExecutor::AddReturnNode(const ir::NodePtr &node) {
  RT_VLOG(VL_RUNTIME) << "Add return node: " << node;
  CHECK_IF_NULL(graph_);
  CHECK_IF_NULL(node);
  auto returnNode = ir::MakeIntrusive<ir::Node>();
  CHECK_IF_NULL(returnNode);
  returnNode->op = ops::Op_return;
  returnNode->output = node->output;
  (void)returnNode->inputs.emplace_back(node);
  (void)graph_->nodes.emplace_back(returnNode);
}

// Run a single node
void GraphExecutor::RunNode(ir::NodePtr node) {
  if (node->op == ops::Op_End) {
    return;
  }

  if (node->op == ops::Op_make_tuple) {
    ProcessMakeTuple(node);
    return;
  }

  if (node->op == ops::Op_tuple_getitem) {
    ProcessTupleGetItem(node);
    return;
  }

  if (auto it = opsOutputFromInputIndex.find(node->op); it != opsOutputFromInputIndex.end()) {
    node->output = node->inputs[it->second]->output;
    return;
  }

  auto iter = kernels_.find(node);
  if (iter == kernels_.end()) {
    RT_GLOG(ERROR) << "kernel not found: " << node;
    exit(EXIT_FAILURE);
  }
  auto kernel = iter->second;

  if (isDynamic_) {
    kernel->InferShape();
    kernel->Resize();
  } else if (IsDAKernelNeedForceResize(node)) {
    kernel->Resize();
  }

  if (auto it = opsOutputValueFromInputIndex.find(node->op); it != opsOutputValueFromInputIndex.end()) {
    RT_VLOG(VL_RUNTIME) << "Skip launch kernel for node" << node;
    auto outputTensor = node->output->ToTensor();
    auto inputStorage = node->inputs[it->second]->output->ToTensor()->GetStorage();
    node->output = ir::MakeIntrusive<ir::Value>(
      ir::MakeIntrusive<ir::Tensor>(inputStorage, outputTensor->Shape(), outputTensor->Dtype()));
  } else {
    kernel->Launch();
  }

  if (node->op != ops::Op_return) {
    // keep outputs memory until consumed.
    recycler_->FreeUnusedNodes(node);
  }
}

// Free the memory of graph outputs
void GraphExecutor::FreeGraphOutputs() {
  CHECK_IF_NULL(graph_);
  CHECK_IF_NULL(recycler_);
  auto returnNode = graph_->nodes[graph_->nodes.size() - 1];
  CHECK_IF_FAIL(returnNode->op == ops::Op_return);
  recycler_->FreeUnusedNodes(returnNode);
  recycler_->PrintRunningRefCounts();
}

// Record tensor refCount
void GraphExecutor::RecordTensorRefCount() {
  CHECK_IF_NULL(recycler_);
  CHECK_IF_NULL(graph_);

  for (auto &node : graph_->nodes) {
    recycler_->ForwardRecordInputsRefCounts(node);
  }
}

// Run the built graph.
void GraphExecutor::RunGraph(bool isDynamic) {
  CHECK_IF_NULL(executor_);
  RT_VLOG(VL_RUNTIME) << "Start run graph: " << name_ << ", isDynamic: " << isDynamic;
  executor_->Run(isDynamic);
  RT_VLOG(VL_RUNTIME) << "End run graph: " << name_;
}

ir::ValuePtr GraphExecutor::GetOutput() const { return executor_->GetOutput(); }


IrGraphExport GraphExecutor::ExportIrGraph() const {
  CHECK_IF_NULL(graph_);
  // Call free function (same name as this method) via explicit qualification + cast.
  return ::fxrt::runtime::ExportIrGraph(static_cast<const ir::Graph *>(graph_.get()), name_);
}


#ifdef DUMP
// Run the built graph.
std::string GraphExecutor::DumpGraph(bool printStdout) {
  RT_VLOG(VL_RUNTIME) << "Run graph";
  CHECK_IF_NULL(graph_);

  auto fnOutputGraph = [this](std::ostream &outStream) {
    constexpr auto paramPrefix = "param_";
    constexpr auto inputPrefix = "input_";
    outStream << "graph{" << name_ << "}(";
    for (size_t i = 0; i < graph_->inputs.size(); ++i) {
      auto input = graph_->inputs[i];
      (void)inputNumMap_.emplace(input, i);
      outStream << inputPrefix << i;
      if (i < graph_->inputs.size() - 1 || !graph_->parameters.empty()) {
        outStream << ", ";
      }
    }
    for (size_t i = 0; i < graph_->parameters.size(); ++i) {
      auto para = graph_->parameters[i];
      (void)paraNumMap_.emplace(para, i);
      outStream << paramPrefix << i;
      if (i < graph_->parameters.size() - 1) {
        outStream << ", ";
      }
    }
    outStream << ")" << std::endl;
    for (size_t i = 0; i < graph_->inputs.size(); ++i) {
      outStream << std::setw(10) << "// " << inputPrefix << i << " = " << graph_->inputs[i]->output << std::endl;
    }
    for (size_t i = 0; i < graph_->parameters.size(); ++i) {
      outStream << std::setw(10) << "// " << paramPrefix << i << " = " << graph_->parameters[i]->output << std::endl;
    }
    outStream << "{" << std::endl;

    for (size_t i = 0; i < graph_->nodes.size(); ++i) {
      (void)nodeNumMap_.emplace(graph_->nodes[i], i);
    }

    // Run all tensor nodes.
    ir::NodePtr tensorNode{nullptr};
    for (size_t i = 0; i < graph_->nodes.size(); ++i) {
      tensorNode = graph_->nodes[i];
      size_t inputSize = tensorNode->inputs.size();
      std::stringstream ss;
      for (size_t j = 0; j < inputSize; ++j) {
        auto input = tensorNode->inputs[j];
        // Find node number firstly.
        if (auto nodeIt = nodeNumMap_.find(input); nodeIt != nodeNumMap_.cend()) {
          ss << "%" << nodeIt->second;
        } else if (auto inputIt = inputNumMap_.find(input); inputIt != inputNumMap_.cend()) {
          ss << inputPrefix << inputIt->second;
        } else if (auto paraIt = paraNumMap_.find(input); paraIt != paraNumMap_.cend()) {
          ss << paramPrefix << paraIt->second;
        } else {
          ss << "<ERR>";
        }
        if (j != inputSize - 1) {
          ss << ", ";
        }
      }

      if (nodeNumMap_.count(tensorNode) == 0) {
        RT_GLOG(ERROR) << "Failed to find tensor number for " << tensorNode;
        exit(EXIT_FAILURE);
      }
      outStream << "  %" << nodeNumMap_[tensorNode];
      outStream << " = ops." << ops::ToStr(tensorNode->op) << "(" << ss.str() << ")";
      outStream << std::setw(10) << "// " << tensorNode->output << std::endl;
    }

    outStream << "}" << std::endl;
  };

  if (printStdout) {
    fnOutputGraph(std::cout);
    return "";
  } else {
    std::ostringstream oss;
    fnOutputGraph(oss);
    return oss.str();
  }
}
#endif

void GraphExecutor::BuildExecutor() {
  CHECK_IF_FAIL(nullptr == builder_);
  CHECK_IF_FAIL(nullptr == executor_);
  ExecutionMode executionMode = GetExecutionMode();
  CHECK_IF_FAIL(static_cast<size_t>(executionMode) < builderCreators.size());
  builder_ = builderCreators.at(static_cast<size_t>(executionMode))(graph_);
  CHECK_IF_NULL(builder_);
  executor_ = builder_->BuildExecutor();
  CHECK_IF_NULL(executor_);
}

void Executor::Run(bool isDynamic) {
  for (const auto &item : deviceContexts_) {
    auto *deviceContext = item.second;
    CHECK_IF_NULL(deviceContext);
    CHECK_IF_NULL(deviceContext->deviceResManager_);
    deviceContext->deviceResManager_->BindDeviceToCurrentThread(false);
  }

  OpRunner *opRunners = opRunners_->data();
  size_t opNum = opRunners_->size();
  for (size_t i = 0; i < opNum; i++) {
    OpRunner &opRunner = opRunners[i];
    opRunner.UpdateTensors();
    if (auto errNo = opRunner.InferShape() != ops::SUCCESS) {
      RT_GLOG(EXCEPTION) << "Infer shape failed for operator " << opRunner.GetOpName() << "Errno: " << errNo;
    }
    opRunner.AllocateMemory();
    if (auto errNo = opRunner.CalcWorkspace() != ops::SUCCESS) {
      RT_GLOG(EXCEPTION) << "CalcWorkspace shape failed for operator " << opRunner.GetOpName() << "Errno: " << errNo;
    }
    opRunner.AllocateWorkspaceMemory();
    opRunner.FreeMemory();

    if (!opRunner.NeedLaunch()) {
      continue;
    }

    if (auto errNo = opRunner.Launch() != ops::SUCCESS) {
      RT_GLOG(EXCEPTION) << "Launch failed for operator " << opRunner.GetOpName() << "Errno: " << errNo;
    }
  }
}

const ir::ValuePtr &Executor::GetOutput() const { return output_; }
}  // namespace runtime
}  // namespace fxrt
