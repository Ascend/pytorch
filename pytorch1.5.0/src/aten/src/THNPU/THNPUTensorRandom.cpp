#include <THNPU/THNPUTensorRandom.h>
#include <ATen/Config.h>

void THNPURandom_getRNGState(at::Generator *gen_, THByteTensor *rng_state)
{
  auto gen = at::check_generator<at::NPUGenerator>(gen_);
  std::lock_guard<std::mutex> lock(gen->mutex_);

  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = seed_size + offset_size;
  THByteTensor_resize1d(rng_state, total_size);
  THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
  THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");

  auto current_seed = gen->current_seed();
  auto offset = static_cast<int64_t>(gen->philox_offset_per_thread()); // Note that old THCGeneratorState had offset as std::atomic<int64_t>
  memcpy(THByteTensor_data(rng_state), &current_seed, seed_size);
  memcpy(THByteTensor_data(rng_state) + seed_size, &offset, offset_size);
}

void THNPURandom_setRNGState(at::Generator *gen_, THByteTensor *rng_state)
{
  auto gen = at::check_generator<at::NPUGenerator>(gen_);
  std::lock_guard<std::mutex> lock(gen->mutex_);
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = seed_size + offset_size;
  bool no_philox_seed = false;
  if (THByteTensor_nElement(rng_state) == total_size - offset_size) {
    no_philox_seed = true;
  }
  else {
    THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
  }
  THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");
  uint64_t input_seed;
  memcpy(&input_seed, THByteTensor_data(rng_state), seed_size);
  gen->set_current_seed(input_seed);
  int64_t philox_offset = 0;
  if (!no_philox_seed) {
    memcpy(&philox_offset, THByteTensor_data(rng_state) + seed_size, offset_size);
  }
  gen->set_philox_offset_per_thread(static_cast<uint64_t>(philox_offset));
}
