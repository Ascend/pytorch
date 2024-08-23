// Copyright (c) 2022 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NPUVARIABLES_H
#define NPUVARIABLES_H

namespace c10_npu {
enum class SocVersion {
  UnsupportedSocVersion = -1,
  Ascend910PremiumA = 100,
  Ascend910ProA,
  Ascend910A,
  Ascend910ProB,
  Ascend910B,
  Ascend310P1 = 200,
  Ascend310P2,
  Ascend310P3,
  Ascend310P4,
  Ascend910B1 = 220,
  Ascend910B2,
  Ascend910B2C,
  Ascend910B3,
  Ascend910B4,
  Ascend910B4_1,
  Ascend310B1 = 240,
  Ascend310B2,
  Ascend310B3,
  Ascend310B4,
  Ascend910C1 = 250,
  Ascend910C2,
  Ascend910C3,
  Ascend910C4,
  Ascend910C4_1
};

bool SetSocVersion(const char* const socVersion);

const SocVersion& GetSocVersion();

bool IsSupportInfNan();

bool IsBF16Supported();
}  // namespace c10_npu
#endif

