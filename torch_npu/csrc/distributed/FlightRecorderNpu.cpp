#include <torch/csrc/distributed/c10d/FlightRecorderDetail.hpp>
#include <torch/csrc/distributed/c10d/control_plane/Handlers.hpp>

#include <unordered_map>

#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/distributed/ProcessGroupHCCL.hpp"

namespace c10d {

template <>
FlightRecorder<c10_npu::NPUEvent>::FlightRecorder() {
  max_entries_ = getCvarInt({"TORCH_FR_BUFFER_SIZE", "TORCH_HCCL_TRACE_BUFFER_SIZE"}, 0);
  capture_cpp_stack_ = getCvarBool({"TORCH_FR_CPP_STACK", "TORCH_HCCL_TRACE_CPP_STACK"}, false);
  enabled_ = max_entries_ > 0;
  reset_epoch_start_idx_[0] = 0;
}

control_plane::RegisterHandler dumpHcclHandler{
    "dump_hccl_trace_pickle",
    [](const control_plane::Request& req, control_plane::Response& res) {
      const auto& params = req.params();
      size_t validParamCount = 0;
      const std::string includeCollectivesStr = "includecollectives";
      const std::string includeStackTracesStr = "includestacktraces";
      const std::string onlyActiveStr = "onlyactive";
      std::unordered_map<std::string, bool> processedParams = {
          {includeCollectivesStr, true}, {includeStackTracesStr, true}, {onlyActiveStr, false}};

      for (const auto& [paramName, paramValue] : params) {
        auto it = processedParams.find(paramName);
        if (it != processedParams.end()) {
          validParamCount++;
          if (paramValue == "true") {
            it->second = true;
          } else if (paramValue == "false") {
            it->second = false;
          } else {
            res.setStatus(400);
            res.setContent("Invalid value for " + paramName + " valid values are true or false", "text/plain");
            return;
          }
        }
      }
      if (validParamCount < params.size()) {
        res.setStatus(400);
        res.setContent("Invalid parameters - unexpected param passed in", "text/plain");
        return;
      }
      res.setContent(
          c10d_npu::dump_hccl_trace(
              processedParams[includeCollectivesStr],
              processedParams[includeStackTracesStr],
              processedParams[onlyActiveStr]),
          "application/octet-stream");
    }};

control_plane::RegisterHandler jsonDumpHcclHandler{
    "dump_hccl_trace_json",
    [](const control_plane::Request& req, control_plane::Response& res) {
      const auto& params = req.params();
      size_t validParamCount = 0;
      const std::string includeCollectivesStr = "includecollectives";
      const std::string onlyActiveStr = "onlyactive";
      std::unordered_map<std::string, bool> processedParams = {{includeCollectivesStr, true}, {onlyActiveStr, false}};

      for (const auto& [paramName, paramValue] : params) {
        auto it = processedParams.find(paramName);
        if (it != processedParams.end()) {
          validParamCount++;
          if (paramValue == "true") {
            it->second = true;
          } else if (paramValue == "false") {
            it->second = false;
          } else {
            res.setStatus(400);
            res.setContent("Invalid value for " + paramName + " valid values are true or false", "text/plain");
            return;
          }
        }
      }
      if (validParamCount < params.size()) {
        res.setStatus(400);
        res.setContent("Invalid parameters - unexpected param passed in", "text/plain");
        return;
      }
      res.setStatus(200);
      res.setContent(
          c10d_npu::dump_hccl_trace_json(processedParams[includeCollectivesStr], processedParams[onlyActiveStr]),
          "application/json");
    }};

control_plane::RegisterHandler resetHcclHandler{
    "reset_hccl_trace",
    [](const control_plane::Request&, control_plane::Response& res) {
      c10d_npu::reset_hccl_trace();
      res.setStatus(200);
      res.setContent(std::string(), "text/plain");
    }};

/* Helper used by WorkHCCL and the HCCL flight recorder. */
template <>
float getDurationFromEvent<c10_npu::NPUEvent>(c10_npu::NPUEvent& hcclStartEvent, c10_npu::NPUEvent& hcclEndEvent) {
  TORCH_CHECK(hcclEndEvent.query(), "getDuration can only be called after work is succeeded.");
  return hcclStartEvent.elapsed_time(hcclEndEvent);
}

template struct FlightRecorder<c10_npu::NPUEvent>;

} // namespace c10d
