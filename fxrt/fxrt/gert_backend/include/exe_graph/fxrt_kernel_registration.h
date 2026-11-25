#pragma once

#include <string>

namespace fx_converter {

struct FxrtKernelRegistration {
  const char* kernel_name;
  const char* run_func_symbol;
  const char* outputs_creator_symbol;
};

inline constexpr const char* kCustomKernelNameAttr = "_custom_kernel_name";
inline constexpr const char* kCustomKernelSoPathAttr = "_custom_kernel_so_path";
inline constexpr const char* kCustomRunFuncSymbolAttr = "_custom_run_func_symbol";
inline constexpr const char* kCustomOutputsCreatorSymbolAttr = "_custom_outputs_creator_symbol";

const FxrtKernelRegistration* FindFxrtKernelRegistration(const std::string& kernel_name);
const std::string& GetFxrtKernelModulePath();

} // namespace fx_converter
