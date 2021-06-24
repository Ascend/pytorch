// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION. 
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
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

#pragma once

#include <ATen/core/Generator.h>

namespace at {

struct TORCH_NPU_API NPUGenerator : public Generator {
  // Constructors
  NPUGenerator(DeviceIndex device_index = -1);
  ~NPUGenerator() = default;

  // NPUGenerator methods
  std::shared_ptr<NPUGenerator> clone() const;
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  void set_philox_offset_per_thread(uint64_t offset);
  uint64_t philox_offset_per_thread();
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);
  static DeviceType device_type();

private:
  NPUGenerator* clone_impl() const override;
  uint64_t seed_ = default_rng_seed_val;
  uint64_t philox_offset_per_thread_ = 0;
};

namespace npu {
namespace detail {

  TORCH_NPU_API NPUGenerator* getDefaultNPUGenerator(DeviceIndex device_index = -1);
  TORCH_NPU_API std::shared_ptr<NPUGenerator> createNPUGenerator(DeviceIndex device_index = -1);

} // namespace detail
} // namespace npu
} // namespace at

