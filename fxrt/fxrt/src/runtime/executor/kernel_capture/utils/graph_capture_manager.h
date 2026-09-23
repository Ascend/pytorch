#ifndef __RUNTIME_EXECUTOR_KERNEL_CAPTURE_UTILS_GRAPH_CAPTURE_MANAGER_H__
#define __RUNTIME_EXECUTOR_KERNEL_CAPTURE_UTILS_GRAPH_CAPTURE_MANAGER_H__

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <utility>
#include <queue>
#include <functional>

#include "common/common.h"
#include "runtime/executor/op_runner.h"
#include "hardware/hardware_abstract/capture_graph.h"
#include "hardware/hardware_abstract/device_context.h"
#include "ir/tensor/tensor.h"

// Custom hash function for pairs
namespace std {
template <typename T, typename U>
struct hash<pair<T, U>> {
  size_t operator()(const pair<T, U>& p) const {
    auto h1 = hash<T>{}(p.first);
    auto h2 = hash<U>{}(p.second);
    return h1 ^ (h2 << 1);
  }
};
} // namespace std

namespace fxrt {
namespace runtime {

enum ExecutorType {
  KERNEL = 0,
  CAPTURE_GRAPH = 1,
};

struct CaptureKernelInfo {
  void* device_ptr;
  size_t size;
  std::vector<int64_t> shape;

  CaptureKernelInfo(void* ptr, size_t sz, const std::vector<int64_t>& shp) : device_ptr(ptr), size(sz), shape(shp) {}
};

using CaptureKernelInfoPtr = std::shared_ptr<CaptureKernelInfo>;
using CaptureKernelInfoList = std::vector<CaptureKernelInfoPtr>;

class DA_API GraphCaptureManager {
 public:
  GraphCaptureManager() = default;
  ~GraphCaptureManager() = default;

  void Initialize(const std::vector<OpRunner>& opRunners, const device::DeviceContext* expected_device_context);

  /**
   * @brief Check if a kernel runner supports capture.
   *
   * @param opRunner The OpRunner to check
   * @param expected_device_context The expected device context
   * @return true if the kernel supports capture, false otherwise
   */
  bool CheckKernelSupportCapture(const OpRunner& opRunner, const device::DeviceContext* expected_device_context);

  /**
   * @brief Find the positions of kernels that support capture.
   *
   * @param opRunners The vector of OpRunners
   * @param expected_device_context The expected device context
   * @return true if any kernels support capture, false otherwise
   */
  bool FindSupportCaptureKernelPositions(
      const std::vector<OpRunner>& opRunners,
      const device::DeviceContext* expected_device_context);

  /**
   * @brief Initialize capture graphs for a specific shape key.
   *
   * @param device_context The device context
   */
  void CreateCaptureGraph(const device::DeviceContext* device_context, bool fullGraphMode);

  /**
   * @brief Launch all kernels with capture graph.
   *
   * @param opRunners The vector of OpRunners
   * @return true if successful, false otherwise
   */
  bool LaunchAllKernelsWithCapture(std::vector<OpRunner>& opRunners);

  /**
   * @brief Launch all kernels with full-graph capture.
   *
   * @param opRunners The vector of OpRunners
   * @return true if successful, false otherwise
   */
  bool LaunchAllKernelsWithCaptureFullGraph(std::vector<OpRunner>& opRunners, void* captureStream);

  /**
   * @brief Launch all kernels with replay graph.
   *
   * @param opRunners The vector of OpRunners
   * @return true if successful, false otherwise
   */
  bool LaunchAllKernelsWithReplayFullGraph(std::vector<OpRunner>& opRunners, void* executeStream, void* updateStream);

  /**
   * @brief Set the shape key for current execution.
   *
   * @param shape_key The shape key string
   */
  void SetShapeKey(const std::string& shape_key) {
    shape_key_ = shape_key;
  }

  /**
   * @brief Check if a graph with the current shape key has been captured.
   *
   * @return true if captured, false otherwise
   */
  bool HasCapturedGraph() const;

  /**
   * @brief Get the shape key for current execution.
   *
   * @return The shape key string
   */
  const std::string& GetShapeKey() const {
    return shape_key_;
  }

  // Getters for captured kernel positions and executors
  const std::vector<std::pair<size_t, size_t>>& GetCaptureKernelRangePositions() const {
    return capture_kernel_range_positions_;
  }

  const std::vector<std::pair<ExecutorType, size_t>>& GetExecutors() const {
    return executors_;
  }

  // Setters to allow setting analysis results
  void SetCaptureKernelRangePositions(const std::vector<std::pair<size_t, size_t>>& positions) {
    capture_kernel_range_positions_ = positions;
  }

  void SetExecutors(const std::vector<std::pair<ExecutorType, size_t>>& execs) {
    executors_ = execs;
  }

  // Accessors for internal state
  std::unordered_map<std::string, std::vector<CaptureGraphPtr>>& GetCaptureGraphs() {
    return capture_graphs_;
  }

  // Cleanup resources
  void Finalize();

 private:
  // Helper functions
  void InitializeSingleOpUpdateResources(const device::DeviceContext* expected_device_context);
  CaptureGraphPtr GetCurrentFullGraph() const;
  void ExecuteCaptureOpsNeedNotUpdate(std::vector<OpRunner>& opRunners, size_t start, size_t end, void* captureStream)
      const;
  void ExecuteCaptureOpNeedUpdate(
      OpRunner& opRunner,
      void* captureStream,
      CaptureGraph* capture_graph,
      size_t single_op_index);
  void RecordGraphOutputKernelInfo(const OpRunner& opRunner, size_t index);
  void RecoverGraphOutputKernelInfo();
  void UpdateFixAddressBeforeReplayGraph();
  bool CheckParameterNotChange();
  bool IsSingleOp(const std::vector<OpRunner>& opRunners, size_t kernel_index);
  void InitFixedInputInfoForSingleOp(const std::vector<OpRunner>& opRunners);
  bool IsNonFixedInputInReplay(const OpRunner& opRunner, size_t kernel_input_index);

  // Members to track captured graph information
  std::vector<std::pair<size_t, size_t>> capture_kernel_range_positions_;
  std::vector<std::pair<ExecutorType, size_t>> executors_;
  std::vector<size_t> singleOpPos_;
  std::vector<DeviceEventPtr> singleOpUpdateEvents_;
  std::vector<void*> singleOpUpdateHandles_;
  size_t capture_graph_num_{0};
  bool init_{false};

  // Shape key for current execution
  std::string shape_key_;

  // Captured graphs indexed by shape key
  std::unordered_map<std::string, std::vector<CaptureGraphPtr>> capture_graphs_;

  // Fixed addresses for inputs that need to be consistent across captures
  std::unordered_map<std::string, std::unordered_map<std::pair<ir::Node*, size_t>, ir::ValuePtr>>
      fixed_addrs_for_set_inputs_;
  std::vector<std::tuple<std::pair<size_t, std::pair<ir::Node*, size_t>>, ir::ValuePtr, OpRunner*>>
      fixed_addrs_for_update_;

  // Information for non-fixed inputs (like weights and KV cache)
  std::unordered_map<
      std::string,
      std::unordered_map<std::pair<ir::Node*, size_t>, std::tuple<ir::ValuePtr, size_t, OpRunner*>>>
      weight_kv_addrs_;

  // Fixed info for single op execution
  // Using uintptr_t instead of pointers as keys to avoid hash issues
  std::unordered_map<std::string, std::unordered_map<std::pair<uintptr_t, size_t>, std::vector<ir::ValuePtr>>>
      fix_network_input_for_replay_single_op_;
  std::unordered_map<std::string, std::unordered_map<std::pair<uintptr_t, size_t>, CaptureKernelInfoList>>
      fix_single_op_input_info_;
  std::unordered_map<std::string, std::unordered_map<std::pair<uintptr_t, size_t>, CaptureKernelInfoList>>
      fix_single_op_output_info_;
  std::unordered_map<std::string, std::unordered_map<std::pair<uintptr_t, size_t>, CaptureKernelInfoList>>
      fix_single_op_workspace_info_;

  // Recorded kernel output information for graph outputs
  std::unordered_map<std::string, std::vector<std::pair<std::pair<uintptr_t, size_t>, std::vector<size_t>>>>
      recorded_kernel_output_for_graph_output_;
  std::unordered_map<std::string, std::unordered_map<std::pair<uintptr_t, size_t>, CaptureKernelInfoList>>
      fix_replay_graph_output_info_;

  // Task group handles for fullgraph update mode
  std::unordered_map<std::string, std::vector<void*>> task_groups_;
  std::unordered_map<std::string, std::vector<std::pair<OpRunner*, size_t>>> update_executors_;
};

} // namespace runtime
} // namespace fxrt

#endif // __RUNTIME_EXECUTOR_KERNEL_CAPTURE_UTILS_GRAPH_CAPTURE_MANAGER_H__
