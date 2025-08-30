#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <c10/core/StreamGuard.h>
#include <ATen/core/GeneratorForPrivateuseone.h>

#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "torch_npu/csrc/core/npu/NPUGraph.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"

namespace at_npu {
namespace detail {

namespace {

// Ensures we only call npuGetDeviceCount only once.
static std::once_flag num_npu_init_flag;

// Total number of npus in the system.
static int64_t num_npus;

// Ensures default_gens_npu is initialized once.
static std::deque<std::once_flag> npu_gens_init_flag;

// Default, global NPU generators, one per NPU.
static std::vector<at::Generator> default_gens_npu;

/*
* Populates the global variables related to NPU generators
* Warning: this function must only be called once!
*/
static void initNPUGenVector()
{
    num_npus = c10_npu::device_count();
    npu_gens_init_flag.resize(num_npus);
    default_gens_npu.resize(num_npus);
}

} // anonymous namespace

/**
 * PyTorch maintains a collection of default generators that get
 * initialized once. The purpose of these default generators is to
 * maintain a global running state of the pseudo random number generation,
 * when a user does not explicitly mention any generator.
 * getDefaultNPUGenerator gets the default generator for a particular
 * NPU device.
 */
const at::Generator& getDefaultNPUGenerator(c10::DeviceIndex device_index)
{
    std::call_once(num_npu_init_flag, initNPUGenVector);
    c10::DeviceIndex idx = device_index;
    if (idx == -1) {
        idx = c10_npu::current_device();
    } else {
        TORCH_CHECK(idx >= 0 && idx < num_npus, PTA_ERROR(ErrCode::VALUE));
    }
    std::call_once(npu_gens_init_flag[idx], [&] {
        default_gens_npu[idx] = at::make_generator<NPUGeneratorImpl>(idx);
        default_gens_npu[idx].seed();
    });
    return default_gens_npu[idx];
}

/**
 * Utility to create a NPUGeneratorImpl. Returns a shared_ptr
 */
at::Generator createNPUGenerator(c10::DeviceIndex device_index)
{
    std::call_once(num_npu_init_flag, initNPUGenVector);
    c10::DeviceIndex idx = device_index;
    if (idx == -1) {
        idx = c10_npu::current_device();
    }
    TORCH_CHECK(idx >= 0 && idx < num_npus, "The device_index is invalid.", PTA_ERROR(ErrCode::VALUE));
    auto gen = at::make_generator<NPUGeneratorImpl>(idx);
    auto npu_gen = at::check_generator<NPUGeneratorImpl>(gen);
    npu_gen->set_current_seed(c10::default_rng_seed_val);
    npu_gen->set_philox_offset_per_thread(0);
    return gen;
}

} // namespace detail

/**
 * Creates a clone of this NPU Generator State.
 */
c10::intrusive_ptr<NPUGeneratorState> NPUGeneratorState::clone()
{
    return c10::make_intrusive<NPUGeneratorState>(seed_, philox_offset_per_thread_, offset_intragraph_);
}

/**
 * Function to increase the internal offset based on the specified increment.
 */
void NPUGeneratorState::increase(uint64_t increment)
{
    // Rounds increment up to the nearest multiple of 4 to meet alignment
    // requirements.
    // see Note [Why enforce RNG offset % 4 == 0?]
    increment = ((increment + 3) / 4) * 4;
    // Handling different behaviors based on whether capturing is active.
    if (c10_npu::currentStreamCaptureStatus() != c10_npu::CaptureStatus::None) {
        // Ensures that the state is actually capturing.
        TORCH_CHECK(
            capturing_,
            "Attempt to increase offset for a NPU generator not in capture mode.");
        // Ensures the offset is a multiple of 4
        // see Note [Why enforce RNG offset % 4 == 0?]
        TORCH_INTERNAL_ASSERT(
            offset_intragraph_ % 4 == 0, "RNG offset must be a multiple of 4.");
        // Ensures the increment does not cause overflow.
        TORCH_INTERNAL_ASSERT(
            offset_intragraph_ <= std::numeric_limits<uint32_t>::max() - increment,
            "Increment causes overflow in the offset value.");
        offset_intragraph_ += increment;
    } else {
        // Checks that the increment is expected outside graph capturing.
        TORCH_CHECK(
            !capturing_,
            "Offset increment outside graph capture encountered unexpectedly.");
        // Ensures the offset is a multiple of 4
        // see Note [Why enforce RNG offset % 4 == 0?]
        TORCH_INTERNAL_ASSERT(
            philox_offset_per_thread_ % 4 == 0,
            "RNG offset must be a multiple of 4.");
        philox_offset_per_thread_ += increment;
    }
}

/**
 * Registers this state to a NPU graph to manage within the graph.
 */
void NPUGeneratorState::register_graph(c10_npu::NPUGraph* graph)
{
    // Ensures that the RNG state is not currently being captured.
    c10_npu::assertNotCapturing("Cannot register the state during capturing stage.");

    // If this is the first graph to be registered, allocate memory for the seed
    // and offset on the GPU.
    if (registered_graphs_.empty()) {
        auto options = at::TensorOptions().device(at::kPrivateUse1).dtype(at::kLong);
        seed_extragraph_ = at::empty({1}, options);
        offset_extragraph_ = at::empty({1}, options);
    }

    // Insert the graph into the set of registered graphs if it's not already
    // registered.
    if (registered_graphs_.find(graph) == registered_graphs_.end()) {
        registered_graphs_.insert(graph);
    }
}

/**
 * Unregisters a NPU graph from the RNG state.
 */
void NPUGeneratorState::unregister_graph(c10_npu::NPUGraph* graph)
{
    // Ensures that the RNG state is not currently being captured.
    c10_npu::assertNotCapturing(
        "Cannot unregister the state during capturing stage.");
    // Verify the graph was previously registered.
    TORCH_CHECK(
        registered_graphs_.find(graph) != registered_graphs_.end(),
        "The graph should be registered to the state");

    // Remove the graph from the set of registered graphs.
    registered_graphs_.erase(graph);

    // If no more graphs are registered, deallocate the GPU memory for the seed
    // and offset.
    if (registered_graphs_.empty()) {
        seed_extragraph_.reset();
        offset_extragraph_.reset();
    }
}

/**
 * Performs the prologue steps for capturing a NPU graph state.
 * This method is intended to reset graph-related state variables before capturing begins.
 */
void NPUGeneratorState::capture_prologue()
{
    capturing_ = true;
    offset_intragraph_ = 0;
    seed_extragraph_.fill_(int64_t(seed_));
    offset_extragraph_.fill_(int64_t(0));
}

/**
 * Ends the capturing phase and resets related variables, returning the whole
 * graph increment.
 */
uint64_t NPUGeneratorState::capture_epilogue()
{
    capturing_ = false;
    return offset_intragraph_;
}

/**
 * Prepares the state for replay by setting initial state tensors and applying
 * total increment.
 */
void NPUGeneratorState::replay_prologue(uint64_t wholegraph_increment)
{
    // Ensures the generator is not in capturing mode.
    c10_npu::assertNotCapturing(
        "Cannot prepare for replay during capturing stage.");
    seed_extragraph_.fill_(int64_t(seed_));
    offset_extragraph_.fill_(int64_t(philox_offset_per_thread_));
    // Applies the total increment achieved during previous captures to update the
    // offset.
    increase(wholegraph_increment);
}

/**
 * Note [Why enforce RNG offset % 4 == 0?]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Curand philox does allow offsets that aren't a multiple of 4.
 * But jit kernels don't use curand, they use a custom "Philox" class (see
 * torch/csrc/jit/tensorexpr/npu_random.h or
 * torch/csrc/jit/codegen/npu/runtime/random_numbers.cu).
 * The "Philox" constructor computes offset/4 (a uint64_t division) to locate its
 * internal start in its virtual bitstream viewed as 128-bit chunks, then, when called
 * in a thread, returns one 32-bit chunk at a time from that start in the bitstream.
 * In other words, if the incoming offset is not a multiple of 4, each thread
 * might repeat some previously-generated 32-bit values in the bitstream.
 */

/**
 * NPUGeneratorImpl class implementation
 */
NPUGeneratorImpl::NPUGeneratorImpl(c10::DeviceIndex device_index)
  : c10::GeneratorImpl{c10::Device(c10::DeviceType::PrivateUse1, device_index),
              c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)}
{
    state_ = c10::make_intrusive<NPUGeneratorState>();
}

NPUGeneratorImpl::NPUGeneratorImpl(c10::DeviceIndex device_index, c10::intrusive_ptr<NPUGeneratorState> state)
    : c10::GeneratorImpl{c10::Device(c10::DeviceType::PrivateUse1, device_index), c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)},
      state_(std::move(state))
{}

/**
 * Sets the seed to be used by curandStatePhilox4_32_10
 * Resets the philox_offset_per_thread_ to 0
 *
 * See Note [Acquire lock when using random generators]
 */
void NPUGeneratorImpl::set_current_seed(uint64_t seed)
{
    c10_npu::assertNotCapturing("Cannot call NPUGeneratorImpl::set_current_seed while in capture mode");
    state_->seed_ = seed;
    state_->philox_offset_per_thread_ = 0;
}

/**
 * Sets the offset to be used by curandStatePhilox4_32_10
 *
 * See Note [Acquire lock when using random generators]
 */
void NPUGeneratorImpl::set_offset(uint64_t offset)
{
    c10_npu::assertNotCapturing("Cannot call NPUGeneratorImpl::set_offset while in capture mode");
    set_philox_offset_per_thread(offset);
}

/**
 * Gets the current offset of NPUGeneratorImpl.
 */
uint64_t NPUGeneratorImpl::get_offset() const
{
    c10_npu::assertNotCapturing("Cannot call NPUGeneratorImpl::get_offset while in capture mode");
    // Debatable if get_offset() should be allowed in captured regions.
    // Conservatively disallow it for now.
    return state_->philox_offset_per_thread_;
}

/**
 * Gets the current seed of NPUGeneratorImpl.
 */
uint64_t NPUGeneratorImpl::current_seed() const
{
    c10_npu::assertNotCapturing("Cannot call NPUGeneratorImpl::current_seed while in capture mode");
    // Debatable if current_seed() should be allowed in captured regions.
    // Conservatively disallow it for now.
    return state_->seed_;
}

/**
 * Gets a nondeterministic random number from /dev/urandom or time,
 * seeds the CPUGeneratorImpl with it and then returns that number.
 *
 * You can move this function to Generator.cpp if the algorithm
 * in getNonDeterministicRandom is unified for both CPU and NPU
 */
uint64_t NPUGeneratorImpl::seed()
{
    c10_npu::assertNotCapturing("Cannot call NPUGeneratorImpl::seed while in capture mode");
    auto random = c10::detail::getNonDeterministicRandom(true);
    this->set_current_seed(random);
    return random;
}

/**
 * Gets the current internal state of NpuGeneratorImpl. The internal
 * state is returned as a CPU byte tensor.
 */
c10::intrusive_ptr<c10::TensorImpl> NPUGeneratorImpl::get_state() const
{
    // The RNG state comprises the seed, and an offset used for Philox.
    // The following line is just here for BC reason. sizeof curandStateMtgp32 is 4120.
    // It used to be static const size_t states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
    // MAX_NUM_BLOCKS was 200 and sizeof(curandStateMtgp32) is 4120. Hardcoding these numbers here
    // because this is just host side code and we don't want to worry about linking with npu
    static const size_t seed_size = sizeof(uint64_t);
    static const size_t offset_size = sizeof(int64_t);
    static const size_t total_size = seed_size + offset_size;

    auto state_tensor = at::detail::empty_cpu({(int64_t)total_size}, at::ScalarType::Byte,
                                              c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
    auto rng_state = state_tensor.data_ptr<uint8_t>();
    // since curandStateMTGP is not used anymore, fill gen_states of THCGenerator with deterministic garbage value of -1
    // gen_states in THCGenerator struct was an array of curandStateMtgp32s.
    auto current_seed = this->current_seed();
    auto offset = static_cast<int64_t>(this->philox_offset_per_thread()); // Note that old THCGeneratorState had offset as std::atomic<int64_t>
    memcpy(rng_state, &current_seed, seed_size);
    memcpy(rng_state + seed_size, &offset, offset_size);

    return state_tensor.getIntrusivePtr();
}

/**
 * Sets the internal state of NPUGeneratorImpl. The new internal state
 * must be a strided CPU byte tensor and have appropriate size. See
 * comments of NPUGeneratorImpl::state for information about the layout
 * and size of the internal state.
 */
void NPUGeneratorImpl::set_state(const c10::TensorImpl& new_state)
{
    c10_npu::assertNotCapturing("Please ensure to utilize the NPUGeneratorImpl::set_state_index method during capturing.");
    static const size_t seed_size = sizeof(uint64_t);
    static const size_t offset_size = sizeof(int64_t);
    static const size_t total_size = seed_size + offset_size;

    at::detail::check_rng_state(new_state);

    bool no_philox_seed = false;
    auto new_state_size = new_state.numel();
    if (new_state_size == total_size - offset_size) {
        no_philox_seed = true;
    } else {
        TORCH_CHECK(new_state_size == total_size, "RNG state is wrong size", PTA_ERROR(ErrCode::PARAM));
    }

    uint64_t input_seed = 0;
    auto new_rng_state = new_state.data_dtype_initialized<uint8_t>();
    memcpy(&input_seed, new_rng_state, seed_size);
    this->set_current_seed(input_seed);
    int64_t philox_offset = 0;
    if (!no_philox_seed) {
        memcpy(&philox_offset, new_rng_state + seed_size, offset_size);
    }
    this->set_philox_offset_per_thread(static_cast<uint64_t>(philox_offset));
}

/**
 * Sets the generator's current state to
 * This function allows switching between different registered states of
 * the generator.
 */
void NPUGeneratorImpl::graphsafe_set_state(const c10::intrusive_ptr<GeneratorImpl>& gen)
{
    c10::intrusive_ptr<NPUGeneratorImpl> npu_gen = c10::dynamic_intrusive_pointer_cast<NPUGeneratorImpl>(gen);
    TORCH_CHECK(npu_gen, "Expected a NPU Generator");
    state_ = npu_gen->state_;
}

/**
 * Get the GeneratorImpl that point to current state_
 */
c10::intrusive_ptr<c10::GeneratorImpl> NPUGeneratorImpl::graphsafe_get_state() const
{
    auto gen = c10::make_intrusive<NPUGeneratorImpl>(this->device().index(), state_);
    return gen;
}


/**
 * Sets the philox_offset_per_thread_ to be used by curandStatePhilox4_32_10
 *
 * See Note [Acquire lock when using random generators]
 */
void NPUGeneratorImpl::set_philox_offset_per_thread(uint64_t offset)
{
    // see Note [Why enforce RNG offset % 4 == 0?]
    TORCH_CHECK(offset % 4 == 0, "offset must be a multiple of 4", PTA_ERROR(ErrCode::VALUE));
    state_->philox_offset_per_thread_ = offset;
}

/**
 * Gets the current philox_offset_per_thread_ of NpuGeneratorImpl.
 */
uint64_t NPUGeneratorImpl::philox_offset_per_thread() const
{
    return state_->philox_offset_per_thread_;
}

/**
 * Registers this state to a NPU graph to manage within the graph.
 */
void NPUGeneratorImpl::register_graph(c10_npu::NPUGraph* graph)
{
    graph->register_generator_state(state_);
    state_->register_graph(graph);
}

/**
 * Unregisters a NPU graph from the RNG state.
 */
void NPUGeneratorImpl::unregister_graph(c10_npu::NPUGraph* graph)
{
    state_->unregister_graph(graph);
}

/**
 * Gets the seed and philox offset value to be used in
 * curandStatePhilox4_32_10, in an opaque PhiloxNpuState that's safe
 * and can be used non-divergently in callers whether NPU graph
 * capture is underway or not.  See
 * Note [NPU Graph-safe RNG states]
 *
 * Each kernel using philox has to sensibly increment offset
 * for future users of philox. So it gets the "old" value for
 * itself (before add), and tells subsequent users which offset
 * they should use, since only the kernel knows how many randoms
 * it intends to generate.
 *
 * Increment should be at least the number of curand() random numbers used in
 * each thread. It is the user's responsibility to make sure the increment
 * for philox is never smaller than the number of curand() calls. Increment
 * value > the number of curand() calls won't harm but anything less would mean
 * that you would be reusing random values from previous calls.
 *
 * See Note [Acquire lock when using random generators]
 */
PhiloxNpuState NPUGeneratorImpl::philox_npu_state(uint64_t increment)
{
    if (c10_npu::currentStreamCaptureStatus() != c10_npu::CaptureStatus::None) {
        uint32_t offset = state_->offset_intragraph_;
        state_->increase(increment);
        return PhiloxNpuState(
            &state_->seed_extragraph_,
            &state_->offset_extragraph_,
            offset);
    } else {
        uint64_t offset = state_->philox_offset_per_thread_;
        state_->increase(increment);
        return PhiloxNpuState(state_->seed_, offset);
    }
}

/**
 * Temporarily accommodates call sites that use philox_engine_inputs.
 * Allows incremental refactor of call sites to use philox_npu_state.
 */
std::pair<uint64_t, uint64_t> NPUGeneratorImpl::philox_engine_inputs(uint64_t increment)
{
    c10_npu::assertNotCapturing(
        "Refactor this op to use NPUGeneratorImpl::philox_npu_state. Cannot call NPUGeneratorImpl::philox_engine_inputs");
    uint64_t offset = state_->philox_offset_per_thread_;
    state_->increase(increment);
    return std::make_pair(state_->seed_, offset);
}

/*
 * Gets the DeviceType of NPUGeneratorImpl.
 * Used for type checking during run time.
 */
c10::DeviceType NPUGeneratorImpl::device_type()
{
    return c10::DeviceType::PrivateUse1;
}

/**
 * Public clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<NPUGeneratorImpl> NPUGeneratorImpl::clone() const
{
    return std::shared_ptr<NPUGeneratorImpl>(this->clone_impl());
}

/**
 * Private clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
NPUGeneratorImpl* NPUGeneratorImpl::clone_impl() const
{
    c10_npu::assertNotCapturing("Cannot call NPUGeneratorImpl::clone_impl while in capture mode");
    auto gen = new NPUGeneratorImpl(this->device().index(), state_->clone());
    return gen;
}

} // namespace at_npu
