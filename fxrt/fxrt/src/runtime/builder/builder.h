#ifndef __RUNTIME_BUILDER_BUILDER_H__
#define __RUNTIME_BUILDER_BUILDER_H__

#include <vector>
#include <memory>
#include <unordered_map>
#include <map>

#include "ops/operator.h"
#include "hardware/device.h"
#include "hardware/hardware_abstract/device_context.h"
#include "runtime/executor/op_runner.h"

namespace fxrt {
namespace runtime {
class Executor;

/**
 * @brief Base class for building executor.
 *
 * The Builder class is responsible for constructing executor that can run
 * computational graph. It analyzes tensor dependencies, creates operator runners,
 * and builds the final executor with all necessary components.
 */
class DA_API Builder {
 public:
  Builder() = delete;
  explicit Builder(const ir::GraphPtr& graph) : graph_(graph) {}
  virtual ~Builder() = default;

  /**
   * @brief Build an executor for the computational graph.
   *
   * This method orchestrates the building process by:
   * 1. Analyzing tensor reference count
   * 2. Creating operation runners
   * 3. Constructing the final executor
   *
   * @return A unique pointer to the constructed executor.
   */
  virtual std::unique_ptr<Executor> BuildExecutor();

  /**
   * @brief Sets up operation runners for the computational graph.
   *
   * This method orchestrates the setup process by creating operation runners,
   * updating reference node output values, and recording storage free points.
   * It serves as the main entry point for preparing all operations in the graph
   * for execution.
   */
  void SetupOpRunners();

  const ir::ValuePtr& GetGraphOutput() const;

 protected:
  /**
   * @brief Creates operation runners for all nodes in the graph.
   *
   * This method iterates through all nodes in the computational graph,
   * creates corresponding operator runners, and configures them with
   * appropriate memory management settings.
   */
  void CreateOpRunners();

  /**
   * @brief Updates output values for reference nodes in the graph.
   *
   * This method processes nodes that reference input tensors and ensures their
   * output values are properly updated to reflect the current state of the
   * referenced data. This is essential for maintaining data consistency
   * across the computational graph.
   */
  void UpdateRefNodeOutputValue();

  /**
   * @brief Records tensor update points to update tensor lazily.
   *
   * This method analyzes the computational graph to mark the following tensors
   * to be updated lazily right before its first consumer op:
   * 1. Tensors in graph inputs: to reduce stall at the beginning of graph execution.
   * 2. Tensors of torch op output: to reduce stall at the end of op launch.
   * 3. Tensors of other situations has conversion cost to delay.
   */
  void RecordTensorUpdatePoint();

  /**
   * @brief Records storage free points to optimize memory management.
   *
   * This method analyzes the computational graph to determine when storages
   * are no longer needed and can be freed to optimize memory usage.
   */
  void RecordStorageFreePoint();

  // The graph that the executor will run.
  ir::GraphPtr graph_{nullptr};

  // Shared pointer to the vector of OpRunners for all operators by execution order in graph_.
  std::shared_ptr<std::vector<OpRunner>> opRunners_{nullptr};

  std::unordered_map<ir::Node*, OpRunner*> nodeToOpRunner_;
  std::map<hardware::DeviceType, device::DeviceContext*> deviceContexts_{};
};
} // namespace runtime
} // namespace fxrt

#endif // __RUNTIME_BUILDER_BUILDER_H__
