#ifndef __RUNTIME_EXECUTOR_H__
#define __RUNTIME_EXECUTOR_H__

#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/common.h"
#include "common/visible.h"
#include "ops/operator.h"
#include "runtime/executor/mempool.h"
#include "runtime/executor/op_runner.h"
#include "runtime/builder/builder.h"
#include "runtime/utils/utils.h"
#include "hardware/hardware_abstract/device_context.h"
#include "optimize/pass/pass.h"
#include "ir/graph.h"

#include "runtime/executor/ir_graph_export.h"

#define DUMP

namespace fxrt {
namespace runtime {
enum ExecutionMode : size_t {
  Base = 0,
  Pipeline = 1,
  GroupLaunch = 2,
  AclGraph = 3,
};

/**
 * @brief Base class for executing a computational graph.
 *
 * The Executor class provides the basic interface and implementation for
 * running computational graph. It holds the graph and operation runners
 * needed for execution.
 */
class DA_API Executor {
 public:
  Executor() = delete;
  Executor(
      const std::shared_ptr<std::vector<OpRunner>>& opRunners,
      const std::map<hardware::DeviceType, device::DeviceContext*>& deviceContexts,
      const ir::ValuePtr& output)
      : opRunners_(opRunners), deviceContexts_(deviceContexts), output_(output) {}

  virtual ~Executor() = default;

  /**
   * @brief Executes the computational graph.
   * This method runs all operations in the graph by execution order.
   * Subclasses can override this method to provide specialized execution behavior, such as Pipeline mode, AclGraph
   * mode.
   * @param isDynamic whether run graph by dynamic shape mode.
   */
  virtual void Run(bool isDynamic);

  virtual const ir::ValuePtr& GetOutput() const;

 protected:
  // Shared pointer to the vector of OpRunners for all operators by execution order.
  std::shared_ptr<std::vector<OpRunner>> opRunners_{nullptr};

  std::map<hardware::DeviceType, device::DeviceContext*> deviceContexts_{};

  ir::ValuePtr output_{nullptr};
};

class DA_API GraphExecutor {
 public:
  GraphExecutor();
  ~GraphExecutor();

  // 1. Graph construct and optimize, mlrl->infer ir->execution order(op runners)
  // Start building graph.
  void BeginGraph(const std::string& name);
  // Finish building graph.
  void EndGraph();

  // Optimize the graph.
  void OptGraph();
  // Add a parameter node for graph.
  ir::NodePtr AddParameterNode(const ir::ValuePtr& value = nullptr);
  // Add an input node for graph.
  ir::NodePtr AddInputNode(const ir::ValuePtr& value = nullptr);
  // Add a value node.
  ir::NodePtr AddValueNode(const ir::ValuePtr& value = nullptr);
  // Add an operation node.
  ir::NodePtr AddOpNode(ops::Op op, const std::vector<ir::NodePtr>& inputs, const ir::ValuePtr& output = nullptr);
  // Add return node.
  void AddReturnNode(const ir::NodePtr& node);

  // 2. Create Builder, analyse execution order and create Executor by execution mode.
  void BuildExecutor();

  // 3. Run graph via Executor.
  // Run the built graph.
  void RunGraph(bool isDynamic = false);
  // If the graph had been built.
  bool HasGraph() const {
    return graph_ != nullptr;
  }
  // Set memory free func for Tensor data
  void SetFreeFunc(std::function<void(void*)>&& func) {
    CHECK_IF_NULL(recycler_);
    recycler_->SetFreeFunc(std::move(func));
  }
  // Free the memory of graph outputs
  void FreeGraphOutputs();
  // Record tensor refCount
  void RecordTensorRefCount();
#ifdef DUMP
  // Dump the built graph.
  std::string DumpGraph(bool printStdout = true);
#endif

  ir::ValuePtr GetOutput() const;

  IrGraphExport ExportIrGraph() const;

 private:
  void RunNode(ir::NodePtr node);

  std::string name_;
  ir::GraphPtr graph_;
  bool isDynamic_{false};
  std::unordered_map<ir::NodePtr, ops::DAKernel*> kernels_;
  TensorDataRecycler* recycler_{nullptr};

  std::unique_ptr<Builder> builder_{nullptr};
  std::unique_ptr<Executor> executor_{nullptr};
#ifdef DUMP
  std::unordered_map<ir::NodePtr, size_t> paraNumMap_;
  std::unordered_map<ir::NodePtr, size_t> inputNumMap_;
  std::unordered_map<ir::NodePtr, size_t> nodeNumMap_;
#endif
};
} // namespace runtime
} // namespace fxrt

#endif // __RUNTIME_EXECUTOR_H__
