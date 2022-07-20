#ifndef TH_NPU_TENSOR_RANDOM_INC
#define TH_NPU_TENSOR_RANDOM_INC

#include <TH/THTensor.h>
#include <ATen/npu/NPUGenerator.h>

TH_API void THNPURandom_getRNGState(at::Generator *gen_, THByteTensor *rng_state);
TH_API void THNPURandom_setRNGState(at::Generator *gen_, THByteTensor *rng_state);

#endif
