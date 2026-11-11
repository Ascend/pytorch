#include <gtest/gtest.h>

#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/distributed/HCCLUtils.hpp"

namespace c10d_npu {
namespace {

TEST(HcclDataTypeTest, E8M0ScalarTypeMapping) {
  if (c10_npu::GetSocVersion() != c10_npu::SocVersion::Ascend950) {
    GTEST_SKIP() << "E8M0 HCCL communication is only supported on Ascend950 (Atlas A5).";
  }
  EXPECT_EQ(getHcclDataType(at::ScalarType::Float8_e8m0fnu), HCCL_DATA_TYPE_FP8E8M0);
}

TEST(HcclDataTypeTest, E8M0TypeName) {
  EXPECT_EQ(getHcclDataTypeSerialString(HCCL_DATA_TYPE_FP8E8M0),
            "at::ScalarType::Float8_e8m0fnu");
}

struct HcclDataTypeCase {
  const char* test_name;
  at::ScalarType scalar_type;
  HcclDataType hccl_type;
  const char* type_name;
};

class HcclExistingDataTypeTest : public ::testing::TestWithParam<HcclDataTypeCase> {};

TEST_P(HcclExistingDataTypeTest, ScalarTypeMapping) {
  const auto& param = GetParam();
  EXPECT_EQ(getHcclDataType(param.scalar_type), param.hccl_type);
}

TEST_P(HcclExistingDataTypeTest, TypeName) {
  const auto& param = GetParam();
  EXPECT_EQ(getHcclDataTypeSerialString(param.hccl_type), param.type_name);
}

INSTANTIATE_TEST_SUITE_P(
    ExistingDataTypes,
    HcclExistingDataTypeTest,
    ::testing::Values(
        HcclDataTypeCase{"FP16", at::kHalf, HCCL_DATA_TYPE_FP16, "at::kHalf"},
        HcclDataTypeCase{"BF16", at::kBFloat16, HCCL_DATA_TYPE_BFP16, "at::kBFloat16"},
        HcclDataTypeCase{"FP32", at::kFloat, HCCL_DATA_TYPE_FP32, "at::kFloat"},
        HcclDataTypeCase{"FP8E4M3", at::ScalarType::Float8_e4m3fn,
                         HCCL_DATA_TYPE_FP8E4M3, "at::ScalarType::Float8_e4m3fn"},
        HcclDataTypeCase{"FP8E5M2", at::ScalarType::Float8_e5m2,
                         HCCL_DATA_TYPE_FP8E5M2, "at::ScalarType::Float8_e5m2"}),
    [](const ::testing::TestParamInfo<HcclDataTypeCase>& info) {
      return info.param.test_name;
    });

} // namespace
} // namespace c10d_npu
