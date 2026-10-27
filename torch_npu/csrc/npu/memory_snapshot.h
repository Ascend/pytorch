#pragma once

#include <c10/util/Optional.h>
#include <torch/csrc/Export.h>
#include <string>
#include <vector>

namespace torch_npu {

TORCH_NPU_API void _record_memory_history(
    c10::optional<std::string> enabled = "all",
    c10::optional<std::string> context = "all",
    const std::string& stacks = "all",
    size_t max_entries = UINT64_MAX,
    const std::vector<std::string>& skip_actions = {});

TORCH_NPU_API std::string _memory_snapshot_pickled();

} // namespace torch_npu
