#include "runtime/builder/kernel_launch_group/kernel_launch_group_builder.h"
#include <unordered_set>
#include <utility>
#include "runtime/executor/kernel_launch_group/kernel_launch_group_executor.h"

namespace fxrt {
namespace runtime {
KernelLaunchGroupBuilder::KernelLaunchGroupBuilder(const ir::GraphPtr& graph) : Builder(graph) {}

std::unique_ptr<Executor> KernelLaunchGroupBuilder::BuildExecutor() {
  RT_VLOG(VL_RUNTIME) << "Begin build kernel launch group executor.";
  SetupOpRunners();

  CheckGroupLaunchRequirements();
  PartitionKernelLaunchGroups();
  RecordGraphInputs();
  RecordGraphOutputs();

  auto executor = std::make_unique<KernelLaunchGroupExecutor>(
      opRunners_,
      deviceContexts_,
      opRunnerGroups_,
      serialLaunchOps_,
      graphInputTensors_,
      graphInputTensorsWithDynamicShape_,
      graphOutputs_,
      parallelDispatchNum_,
      parallelSliceNum_,
      GetGraphOutput());
  executor->Initialize();
  RT_VLOG(VL_RUNTIME) << "End build kernel launch group executor.";
  return executor;
}

void KernelLaunchGroupBuilder::CheckGroupLaunchRequirements() const {
  auto it = std::find_if(opRunners_->begin(), opRunners_->end(), [](const OpRunner& opRunner) {
    return opRunner.GetDevice().type == hardware::DeviceType::CPU;
  });
  if (it != opRunners_->end()) {
    RT_GLOG(EXCEPTION)
        << "Find CPU op: " << it->GetOpName()
        << ". The kernel group launch(parallel launch) feature can not work in this case. Please eliminate "
           "heterogeneous(CPU) ops or disable kernel group launch feature.";
  }
}

void KernelLaunchGroupBuilder::PartitionKernelLaunchGroups() {
  static const char kernelLaunchThreadNum[] = "FXRT_KERNEL_LAUNCH_THREAD_NUM";
  static const char kernelLaunchGroupNum[] = "FXRT_KERNEL_LAUNCH_GROUP_NUM";

  auto kernelLaunchThreadNumStr = GetEnv(kernelLaunchThreadNum);
  if (!IsPositiveInteger(kernelLaunchThreadNumStr)) {
    RT_GLOG(EXCEPTION) << "Invalid kernel launch thread number: " << kernelLaunchThreadNumStr;
  }

  parallelDispatchNum_ = std::stoull(kernelLaunchThreadNumStr);
  if (parallelDispatchNum_ < 1) {
    RT_GLOG(EXCEPTION) << "Invalid thread num: " << parallelDispatchNum_
                       << " for kernel launch group, please check the `thread_num` value of env: "
                          "FXRT_KERNEL_LAUNCH_THREAD_NUM";
  }
  RT_VLOG(VL_RUNTIME) << "The parallel dispatch thread number: " << parallelDispatchNum_;

  auto kernelLaunchGroupNumStr = GetEnv(kernelLaunchGroupNum);
  if (!IsPositiveInteger(kernelLaunchGroupNumStr)) {
    RT_GLOG(EXCEPTION) << "Invalid kernel launch group number: " << kernelLaunchGroupNumStr;
  }
  uint64_t totalKernelGroupNum = std::stoull(kernelLaunchGroupNumStr);
  parallelSliceNum_ = totalKernelGroupNum / parallelDispatchNum_;
  if (parallelSliceNum_ < 1) {
    RT_GLOG(EXCEPTION) << "Invalid kernel group num: " << totalKernelGroupNum
                       << " from env: FXRT_KERNEL_LAUNCH_GROUP_NUM"
                       << ", kernel group num must be greater than or equal to thread num: " << parallelDispatchNum_
                       << " from env: FXRT_KERNEL_LAUNCH_THREAD_NUM";
  }
  RT_VLOG(VL_RUNTIME) << "The kernel group per thread: " << parallelSliceNum_;

  CHECK_IF_FAIL(opRunnerGroups_ == nullptr);
  opRunnerGroups_ = std::make_shared<std::vector<std::pair<OpRunner*, size_t>>>();
  // Get parallel launch kernels slice/group.
  opRunnerGroups_->resize(parallelDispatchNum_ * parallelSliceNum_);
  size_t totalKernelNum = opRunners_->size();
  CHECK_IF_FAIL(totalKernelNum > 0);
  size_t kernelNumPerDispatcher = totalKernelNum / (parallelDispatchNum_ * parallelSliceNum_);
  RT_VLOG(VL_RUNTIME) << "Total kernel num: " << opRunners_->size();
  RT_VLOG(VL_RUNTIME) << "The kernel num per parallel slice: " << kernelNumPerDispatcher;
  OpRunner* begin = &(opRunners_->front());
  for (size_t i = 0; i < opRunnerGroups_->size(); i++) {
    if (i < opRunnerGroups_->size() - 1) {
      (*opRunnerGroups_)[i] = {begin + i * kernelNumPerDispatcher, kernelNumPerDispatcher};
    } else {
      size_t remainingOpNum = opRunners_->size() - kernelNumPerDispatcher * (opRunnerGroups_->size() - 1);
      (*opRunnerGroups_)[i] = {begin + i * kernelNumPerDispatcher, remainingOpNum};
    }
    RT_VLOG(VL_RUNTIME) << "The op runner group[" << i << "] op num: " << (*opRunnerGroups_)[i].second;
  }

  // TODO:  // NOLINT(readability/todo)
  // Ops can not parallel dispatch:
  // 1. The heterogeneous operator like MoveTo, device context type maybe 'Ascend' type, but output DeviceAddress type
  // is 'CPU'.
  // 2. force resize kernel
  // 3. Communication ops

  serialLaunchOps_ = std::make_shared<std::vector<OpRunner*>>();
}

void KernelLaunchGroupBuilder::RecordGraphInputs() {
  graphInputTensorsWithDynamicShape_ = std::make_shared<std::vector<std::pair<ir::TensorPtr, std::vector<int64_t>>>>();
  graphInputTensors_ = std::make_shared<std::vector<ir::TensorPtr>>();
  const auto& nodes = graph_->inputs;
  for (const auto& node : nodes) {
    CHECK_IF_FAIL(node->op == ops::Op_End);

    const auto& output = node->output;
    CHECK_IF_NULL(output);
    ir::VisitAllTensors(output, [&](const ir::TensorPtr& tensor) {
      CHECK_IF_NULL(tensor);
      (void)graphInputTensors_->emplace_back(tensor);
      if (tensor->HasDynamicShape()) {
        (void)graphInputTensorsWithDynamicShape_->emplace_back(tensor, tensor->Shape());
      }
    });
  }
}

void KernelLaunchGroupBuilder::RecordGraphOutputs() {
  const auto& nodes = graph_->nodes;
  CHECK_IF_FAIL(!nodes.empty());
  auto& returnNode = nodes.back();
  CHECK_IF_NULL(returnNode);
  CHECK_IF_FAIL(returnNode->op == ops::Op_return);
  auto& outputValue = returnNode->output;
  CHECK_IF_NULL(outputValue);
  CHECK_IF_FAIL(graphOutputs_ == nullptr);
  graphOutputs_ = std::make_shared<std::unordered_set<ir::Tensor*>>();
  if (outputValue->IsTensor()) {
    const auto& outputTensor = outputValue->ToTensor();
    CHECK_IF_NULL(outputTensor);
    (void)graphOutputs_->insert(outputTensor.get());
  } else if (outputValue->IsTuple()) {
    const auto& outputTuple = outputValue->ToTuple();
    CHECK_IF_NULL(outputTuple);
    size_t outputNum = outputTuple->Size();
    graphOutputs_->reserve(outputNum);

    for (size_t i = 0; i < outputNum; ++i) {
      const auto& output = (*outputTuple)[i];
      CHECK_IF_NULL(output);
      // Not support nested tuple currently.
      CHECK_IF_FAIL(!output->IsTuple());
      if (output->IsTensor()) {
        const auto& outputTensor = output->ToTensor();
        (void)graphOutputs_->insert(outputTensor.get());
      }
    }
  }
}
} // namespace runtime
} // namespace fxrt
