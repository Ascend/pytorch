#pragma once

#include <string>
#include <unordered_map>
#include <ATen/Tensor.h>

namespace torch_npu {
// Serialize npu-related information, mainly related to private formats
void npu_info_serialization(const at::Tensor& t, std::unordered_map<std::string, bool>& mate_map);
// Deserialize npu-related information, mainly related to private formats
void npu_info_deserialization(const at::Tensor& t, std::unordered_map<std::string, bool>& mate_map);
}
