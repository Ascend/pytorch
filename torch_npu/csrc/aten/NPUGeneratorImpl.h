#pragma once

#include <limits>
#include <memory>
#include <unordered_set>

#include <c10/core/GeneratorImpl.h>
#include <ATen/core/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/Context.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace c10_npu {
    struct NPUGraph;
}

namespace at_npu {
/**
 * Note [NPU Graph-safe RNG states]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Strategy:
 * ~~~~~~~~~
 * A NPU graph containing multiple RNG ops behaves like a
 * single giant kernel from the perspective of ops external
 * to the graph.  During graph capture, logic below records
 * the total of all offset increments that occur in the graphed
 * region, and records the final total as the offset for the
 * entire graph.
 *
 * When the graph reruns, the logic that reruns it
 * increments this device's NPU generator's offset
 * by that total.
 *
 * Meanwhile, within the graph, at capture time, instead of
 * populating PhiloxNpuStates with the uint64_t offset pulled
 * directly from the global state, PhiloNpuState instead
 * holds a pointer to one-element stream-local int64_t device tensor
 * holding an initial offset value, and a uint64_t holding an
 * intra-graph offset. (The intra-graph offset starts from zero
 * when capture begins.)  In each consumer kernel,
 * at::npu::philox::unpack computes the offset to use for this kernel
 * as intra-graph offset + *initial offset.
 *
 * When the graph reruns, the logic that reruns it first
 * fill_s the initial offset tensor with this device's
 * NPU generator's current offset.
 *
 * The control flow above ensures graphed execution is bitwise
 * identical to eager execution as long as RNG ops are enqueued
 * from a single thread, even if RNG ops and graphs containing
 * RNG ops are enqueued and run simultaneously on multiple streams.
 *
 * Usage:
 * ~~~~~~
 * PhiloxNPUState in this file, and unpack() in
 * npu/NPUGraphsUtils.cuh allow non-divergent use of
 * NPUGeneratorImpl whether graph capture is underway or not.
 *
 * Each PhiloxNpuState instance should be used for one and only one
 * consumer kernel.
 *
 * Example (see e.g. native/npu/Dropout.cu):
 *
 * #include <ATen/NPUGeneratorImpl.h>
 * #include <ATen/npu/NPUGraphsUtils.cuh>
 *
 * __global__ void kernel(..., PhiloxnpuState philox_args) {
 *   auto seeds = at::npu::philox::unpack(philox_args);
 *   IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
 *   curandStatePhilox4_32_10_t state;
 *   curand_init(std::get<0>(seeds), // seed
 *               idx,                // per-thread subsequence
 *               std::get<1>(seeds), // offset in subsequence
 *               &state);
 *   ...
 * }
 *
 * host_caller(...) {
 *   PhiloxnpuState rng_engine_inputs;
 *   {
 *     // See Note [Acquire lock when using random generators]
 *     std::lock_guard<std::mutex> lock(gen->mutex_);
 *
 *     // gen could be HostState or DevState here! No divergent code needed!
 *     rng_engine_inputs = gen->philox_npu_state(offset_increment);
 *   }
 *   kernel<<<...>>>(..., rng_engine_inputs);
 * }
 *
 */


// Stores state values. Passed as a kernel argument. See "Usage:" above.
struct PhiloxNpuState {
    PhiloxNpuState() = default;
    PhiloxNpuState(const PhiloxNpuState&) = default;
    // Called if graph capture is not underway
    PhiloxNpuState(uint64_t seed, uint64_t offset)
    {
        seed_.val = seed;
        offset_.val = offset;
    }
    // Called if graph capture is underway
    PhiloxNpuState(at::Tensor* seed, at::Tensor* offset_extragraph, uint32_t offset_intragraph)
    {
        seed_.ptr = seed;
        offset_.ptr = offset_extragraph;
        offset_intragraph_ = offset_intragraph;
        captured_ = true;
    }

    // Public members, directly accessible by at::Npu::philox::unpack.
    // If we made them private with getters/setters, the getters/setters
    // would have to be __device__, and we can't declare __device__ in ATen.
    union Payload {
        uint64_t val;
        at::Tensor* ptr;
    };

    Payload seed_{};
    Payload offset_{};
    uint32_t offset_intragraph_{0};
    bool captured_ = false;
};


struct NPUGeneratorState : public c10::intrusive_ptr_target {
    uint64_t seed_;
    uint64_t philox_offset_per_thread_;
    uint32_t offset_intragraph_;
    bool capturing_{};
    std::unordered_set<c10_npu::NPUGraph*> registered_graphs_;
    at::Tensor seed_extragraph_{};
    at::Tensor offset_extragraph_{};

    NPUGeneratorState(
        uint64_t seed = c10::default_rng_seed_val,
        uint64_t philox_offset_per_thread = 0,
        uint32_t offset_intragraph = 0)
        : seed_(seed),
        philox_offset_per_thread_(philox_offset_per_thread),
        offset_intragraph_(offset_intragraph) {}

    void increase(uint64_t increment);
    void register_graph(c10_npu::NPUGraph* graph);
    void unregister_graph(c10_npu::NPUGraph* graph);
    void capture_prologue();
    // capture_epilogue returns the wholegraph_increment
    uint64_t capture_epilogue();
    void replay_prologue(uint64_t wholegraph_increment);
    c10::intrusive_ptr<NPUGeneratorState> clone();
};


struct TORCH_NPU_API NPUGeneratorImpl : public c10::GeneratorImpl {
    // Constructors
    NPUGeneratorImpl(c10::DeviceIndex device_index = -1);
    NPUGeneratorImpl(c10::DeviceIndex device_index, c10::intrusive_ptr<NPUGeneratorState> state);
    ~NPUGeneratorImpl() override = default;

    // NPUGeneratorImpl methods
    std::shared_ptr<NPUGeneratorImpl> clone() const;
    void set_current_seed(uint64_t seed) override;
    void set_offset(uint64_t offset) override;
    uint64_t get_offset() const override;
    uint64_t current_seed() const override;
    uint64_t seed() override;
    void set_state(const c10::TensorImpl& new_state) override;
    c10::intrusive_ptr<c10::TensorImpl> get_state() const override;
    void set_philox_offset_per_thread(uint64_t offset);
    uint64_t philox_offset_per_thread() const;
    PhiloxNpuState philox_npu_state(uint64_t increment);
    // For aclgraph
    void graphsafe_set_state(const c10::intrusive_ptr<GeneratorImpl>& state) override;
    c10::intrusive_ptr<c10::GeneratorImpl> graphsafe_get_state() const override;
    void register_graph(c10_npu::NPUGraph* graph);
    void unregister_graph(c10_npu::NPUGraph* graph);
    // Temporarily accommodates call sites that use philox_engine_inputs.
    // Allows incremental refactor of call sites to use philox_npu_state.
    std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);
    static c10::DeviceType device_type();

private:
    NPUGeneratorImpl* clone_impl() const override;
    c10::intrusive_ptr<NPUGeneratorState> state_;
};

namespace detail {
TORCH_NPU_API const at::Generator& getDefaultNPUGenerator(c10::DeviceIndex device_index = -1);
TORCH_NPU_API at::Generator createNPUGenerator(c10::DeviceIndex device_index = -1);
} // namespace detail
} // namespace at_npu
