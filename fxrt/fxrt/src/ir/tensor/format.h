#ifndef __IR_TENSOR_FORMAT_H__
#define __IR_TENSOR_FORMAT_H__

#include <cstdint>
#include <vector>
#include <map>
#include <string>

namespace fxrt {
namespace ir {
enum MemoryFormat : int8_t {
  FORMAT_UNDEFINED = -1,
  FORMAT_NCHW = 0,
  FORMAT_NHWC = 1,
  FORMAT_ND = 2,
  FORMAT_NC1HWC0 = 3,
  FORMAT_FRACTAL_Z = 4,
  FORMAT_NC1HWC0_C04 = 12,
  FORMAT_HWCN = 16,
  FORMAT_NDHWC = 27,
  FORMAT_FRACTAL_NZ = 29,
  FORMAT_NCDHW = 30,
  FORMAT_NDC1HWC0 = 32,
  FORMAT_FRACTAL_Z_3D = 33,
  FORMAT_NC = 35,
  FORMAT_NCL = 47,
};

inline const std::map<MemoryFormat, std::string>& GetFormatEnumToStrMap() {
  static std::map<MemoryFormat, std::string> formatEnumToStrMap = {
      {FORMAT_UNDEFINED, "FORMAT_UNDEFINED"},
      {FORMAT_NCHW, "FORMAT_NCHW"},
      {FORMAT_NHWC, "FORMAT_NHWC"},
      {FORMAT_ND, "FORMAT_ND"},
      {FORMAT_NC1HWC0, "FORMAT_NC1HWC0"},
      {FORMAT_FRACTAL_Z, "FORMAT_FRACTAL_Z"},
      {FORMAT_NC1HWC0_C04, "FORMAT_NC1HWC0_C04"},
      {FORMAT_HWCN, "FORMAT_HWCN"},
      {FORMAT_NDHWC, "FORMAT_NDHWC"},
      {FORMAT_FRACTAL_NZ, "FORMAT_FRACTAL_NZ"},
      {FORMAT_NCDHW, "FORMAT_NCDHW"},
      {FORMAT_NDC1HWC0, "FORMAT_NDC1HWC0"},
      {FORMAT_FRACTAL_Z_3D, "FORMAT_FRACTAL_Z_3D"},
      {FORMAT_NC, "FORMAT_NC"},
      {FORMAT_NCL, "FORMAT_NCL"},
  };
  return formatEnumToStrMap;
}

inline const std::map<std::string, MemoryFormat>& GetFormatStrToEnumMap() {
  static std::map<std::string, MemoryFormat> formatStrToEnumMap = {
      {"FORMAT_UNDEFINED", FORMAT_UNDEFINED},
      {"FORMAT_NCHW", FORMAT_NCHW},
      {"FORMAT_NHWC", FORMAT_NHWC},
      {"FORMAT_ND", FORMAT_ND},
      {"FORMAT_NC1HWC0", FORMAT_NC1HWC0},
      {"FORMAT_FRACTAL_Z", FORMAT_FRACTAL_Z},
      {"FORMAT_NC1HWC0_C04", FORMAT_NC1HWC0_C04},
      {"FORMAT_HWCN", FORMAT_HWCN},
      {"FORMAT_NDHWC", FORMAT_NDHWC},
      {"FORMAT_FRACTAL_NZ", FORMAT_FRACTAL_NZ},
      {"FORMAT_NCDHW", FORMAT_NCDHW},
      {"FORMAT_NDC1HWC0", FORMAT_NDC1HWC0},
      {"FORMAT_FRACTAL_Z_3D", FORMAT_FRACTAL_Z_3D},
      {"FORMAT_NC", FORMAT_NC},
      {"FORMAT_NCL", FORMAT_NCL},
  };
  return formatStrToEnumMap;
}

inline std::string FormatEnumToStr(MemoryFormat format) {
  const auto& formatEnumToStrMap = GetFormatEnumToStrMap();
  auto it = formatEnumToStrMap.find(format);
  if (it == formatEnumToStrMap.end()) {
    return "FORMAT_UNDEFINED";
  }
  return it->second;
}

inline MemoryFormat FormatStrToEnum(const std::string& formatStr) {
  const auto& formatStrToEnumMap = GetFormatStrToEnumMap();
  auto it = formatStrToEnumMap.find(formatStr);
  if (it == formatStrToEnumMap.end()) {
    return FORMAT_UNDEFINED;
  }
  return it->second;
}

} // namespace ir
} // namespace fxrt
#endif // __IR_TENSOR_FORMAT_H__
