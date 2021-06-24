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

#include <ATen/npu/NPUGenerator.h>
#include <c10/npu/NPUFunctions.h>

namespace at {

namespace npu { namespace detail {

// Ensures we only call npuGetDeviceCount only once.
static std::once_flag num_npu_init_flag;

// Total number of npus in the system.
static int64_t num_npus;

// Ensures default_gens_npu is initialized once.
static std::deque<std::once_flag> npu_gens_init_flag;

// Default, global NPU generators, one per NPU.
static std::vector<std::shared_ptr<NPUGenerator>> default_gens_npu;

/* 
* Populates the global variables related to NPU generators
* Warning: this function must only be called once!
*/
static void initNPUGenVector(){
  num_npus = c10::npu::device_count();
  npu_gens_init_flag.resize(num_npus);
  default_gens_npu.resize(num_npus);
}

/**
 * PyTorch maintains a collection of default generators that get
 * initialized once. The purpose of these default generators is to
 * maintain a global running state of the pseudo random number generation,
 * when a user does not explicitly mention any generator.
 * getDefaultNPUGenerator gets the default generator for a particular
 * npu device.
 */
NPUGenerator* getDefaultNPUGenerator(DeviceIndex device_index) {
  std::call_once(num_npu_init_flag, initNPUGenVector);
  DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = c10::npu::current_device();
  } else {
    TORCH_CHECK(idx >= 0 && idx < num_npus);
  }
  std::call_once(npu_gens_init_flag[idx], [&] {
    default_gens_npu[idx] = std::make_shared<NPUGenerator>(idx);
    default_gens_npu[idx]->seed();
  });
  return default_gens_npu[idx].get();
}

/**
 * Utility to create a NPUGenerator. Returns a shared_ptr
 */
std::shared_ptr<NPUGenerator> createNPUGenerator(DeviceIndex device_index) {
  std::call_once(num_npu_init_flag, initNPUGenVector);
  DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = c10::npu::current_device();
  }
  TORCH_CHECK(idx >= 0 && idx < num_npus, "The device_index is invalid.");
  auto gen = std::make_shared<NPUGenerator>(idx);
  gen->set_current_seed(default_rng_seed_val);
  gen->set_philox_offset_per_thread(0);
  return gen;
}

} // namespace detail
} // namespace npu


/**
 * NPUGenerator class implementation
 */
NPUGenerator::NPUGenerator(DeviceIndex device_index)
  : Generator{Device(DeviceType::NPU, device_index),
              DispatchKeySet(c10::DispatchKey::NPUTensorId)} { }

/**
 * Sets the seed to be used by curandStatePhilox4_32_10
 * Resets the philox_offset_per_thread_ to 0
 * 
 * See Note [Acquire lock when using random generators]
 */
void NPUGenerator::set_current_seed(uint64_t seed) {
  seed_ = seed;
  philox_offset_per_thread_ = 0;
}

/**
 * Gets the current seed of NPUGenerator.
 */
uint64_t NPUGenerator::current_seed() const {
  return seed_;
}

/**
 * Gets a nondeterministic random number from /dev/urandom or time,
 * seeds the CPUGenerator with it and then returns that number.
 * 
 * FIXME: You can move this function to Generator.cpp if the algorithm
 * in getNonDeterministicRandom is unified for both CPU and NPU
 */
uint64_t NPUGenerator::seed() {
  auto random = at::detail::getNonDeterministicRandom(true);
  this->set_current_seed(random);
  return random;
}

/**
 * Sets the philox_offset_per_thread_ to be used by curandStatePhilox4_32_10
 * 
 * See Note [Acquire lock when using random generators]
 */
void NPUGenerator::set_philox_offset_per_thread(uint64_t offset) {
  philox_offset_per_thread_ = offset;
}

/**
 * Gets the current philox_offset_per_thread_ of NPUGenerator.
 */
uint64_t NPUGenerator::philox_offset_per_thread() {
  return philox_offset_per_thread_;
}

/**
 * Gets the seed and philox offset value to be used in
 * curandStatePhilox4_32_10
 * 
 * Each kernel using philox has to sensibly increment offset
 * for future users of philox. So it gets the "old" value for
 * itself (before add), and tells subsequent users which offset
 * they should use, since only the kernel knows how many randoms
 * it intends to generate. 
 * 
 * Increment should be at least the number of curand() random numbers used in
 * each thread. It is the user's responsibility to make sure that the increment
 * for philox is never smaller than the number of curand() calls. Increment
 * value > the number of curand() calls won't harm but anything less would mean
 * that you would be reusing random values from previous calls.
 * 
 * See Note [Acquire lock when using random generators]
 */
std::pair<uint64_t, uint64_t> NPUGenerator::philox_engine_inputs(uint64_t increment) {
  uint64_t offset = this->philox_offset_per_thread_;
  this->philox_offset_per_thread_ += increment;
  return std::make_pair(this->seed_, offset);
}

/*
 * Gets the DeviceType of NPUGenerator.
 * Used for type checking during run time.
 */
DeviceType NPUGenerator::device_type() {
  return DeviceType::NPU;
}

/**
 * Public clone method implementation
 * 
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<NPUGenerator> NPUGenerator::clone() const {
  return std::shared_ptr<NPUGenerator>(this->clone_impl());
}

/**
 * Private clone method implementation
 * 
 * See Note [Acquire lock when using random generators]
 */
NPUGenerator* NPUGenerator::clone_impl() const {
  auto gen = new NPUGenerator(this->device().index());
  gen->set_current_seed(this->seed_);
  gen->set_philox_offset_per_thread(this->philox_offset_per_thread_);
  return gen;
}
} // namespace at
