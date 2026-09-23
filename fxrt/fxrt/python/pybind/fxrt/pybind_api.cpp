#include <nanobind/stl/string.h>

#include "ops/custom_op_register.h"
#include "profiler/profiler.h"

namespace nb = nanobind;

// Interface with python
NB_MODULE(_fxrt_api, mod) {
  mod.def(
      "is_custom_op_registered",
      [](const std::string& op_name) {
        return fxrt::ops::CustomOpRegistry::GetInstance().IsCustomOpRegistered(op_name);
      },
      nb::arg("op_name"),
      "Check if a custom operator is registered.");

  // Profiler functions
  mod.def(
      "fxrt_profiler_start_step",
      []() { fxrt::profiler::ProfilerAnalyzer::GetInstance().StartStep(); },
      "Start a profiling step.");
  mod.def(
      "fxrt_profiler_end_step",
      []() { fxrt::profiler::ProfilerAnalyzer::GetInstance().EndStep(); },
      "End a profiling step.");
  mod.def(
      "fxrt_profiler_clear",
      []() { fxrt::profiler::ProfilerAnalyzer::GetInstance().Clear(); },
      "Clear all profiling data.");
  mod.def(
      "fxrt_profiler_enable",
      []() { fxrt::profiler::ProfilerAnalyzer::GetInstance().Enable(); },
      "Enable the profiler.");
  mod.def(
      "fxrt_profiler_disable",
      []() { fxrt::profiler::ProfilerAnalyzer::GetInstance().Disable(); },
      "Disable the profiler.");
  mod.def(
      "fxrt_profiler_is_enabled",
      []() { return fxrt::profiler::ProfilerAnalyzer::GetInstance().IsProfilerEnabled(); },
      "Check if the profiler is enabled.");
}
