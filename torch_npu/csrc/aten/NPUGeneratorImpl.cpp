#include <ATen/Utils.h>
#include <c10/core/StreamGuard.h>
#include <ATen/core/GeneratorForPrivateuseone.h>

#include "torch_npu/csrc/core/npu/NPUFunctions.h"

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"

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
    c10_npu::assertNotCapturing("Not support Generator while in capture mode");
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
    c10_npu::assertNotCapturing("Not support Generator while in capture mode");
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
    c10_npu::assertNotCapturing("Not support Generator while in capture mode");
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
    c10_npu::assertNotCapturing("Not support Generator while in capture mode");
}

/**
 * Sets the seed to be used by curandStatePhilox4_32_10
 * Resets the philox_offset_per_thread_ to 0
 *
 * See Note [Acquire lock when using random generators]
 */
void NPUGeneratorImpl::set_current_seed(uint64_t seed)
{
    c10_npu::assertNotCapturing("Not support Generator while in capture mode");
    seed_ = seed;
    philox_offset_per_thread_ = 0;
}

/**
 * Sets the offset to be used by curandStatePhilox4_32_10
 *
 * See Note [Acquire lock when using random generators]
 */
void NPUGeneratorImpl::set_offset(uint64_t offset)
{
    c10_npu::assertNotCapturing("Not support Generator while in capture mode");
    philox_offset_per_thread_ = offset;
}

/**
 * Gets the current offset of NPUGeneratorImpl.
 */
uint64_t NPUGeneratorImpl::get_offset() const
{
    c10_npu::assertNotCapturing("Not support Generator while in capture mode");
    // Debatable if get_offset() should be allowed in captured regions.
    // Conservatively disallow it for now.
    return philox_offset_per_thread_;
}

#define CAPTURE_DEFAULT_GENS_MSG \
"In regions captured by NPU graphs, you may only use the default NPU RNG " \
"generator on the device that's current when capture begins. " \
"If you need a non-default (user-supplied) generator, or a generator on another " \
"device, please file an issue."

/**
 * Gets the current seed of NPUGeneratorImpl.
 */
uint64_t NPUGeneratorImpl::current_seed() const
{
    c10_npu::assertNotCapturing("Not support Generator while in capture mode");
    // Debatable if current_seed() should be allowed in captured regions.
    // Conservatively disallow it for now.
    return seed_;
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
    c10_npu::assertNotCapturing("Not support Generator while in capture mode");
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
    c10_npu::assertNotCapturing("Not support Generator while in capture mode");
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
    c10_npu::assertNotCapturing("Not support Generator while in capture mode");
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

    uint64_t input_seed;
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
 * Sets the philox_offset_per_thread_ to be used by curandStatePhilox4_32_10
 *
 * See Note [Acquire lock when using random generators]
 */
void NPUGeneratorImpl::set_philox_offset_per_thread(uint64_t offset)
{
    c10_npu::assertNotCapturing("Not support Generator while in capture mode");
    // see Note [Why enforce RNG offset % 4 == 0?]
    TORCH_CHECK(offset % 4 == 0, "offset must be a multiple of 4", PTA_ERROR(ErrCode::VALUE));
    philox_offset_per_thread_ = offset;
}

/**
 * Gets the current philox_offset_per_thread_ of NpuGeneratorImpl.
 */
uint64_t NPUGeneratorImpl::philox_offset_per_thread() const
{
    c10_npu::assertNotCapturing("Not support Generator while in capture mode");
    return philox_offset_per_thread_;
}

/**
 * Called by NpuGraph to prepare this instance for a graph capture region.
 * offset_extragraph is the initial offset at the start of the graphed region.
 * offset_intragraph tracks the offset in the graphed region.
 */
void NPUGeneratorImpl::capture_prologue(int64_t* offset_extragraph)
{
    c10_npu::assertNotCapturing("Not support Generator while in capture mode");
    offset_extragraph_ = offset_extragraph;
    offset_intragraph_ = 0;
    graph_expects_this_gen_ = true;
}

/**
 * Called by NpuGraph to finalize a graph capture region for this instance.
 */
uint64_t NPUGeneratorImpl::capture_epilogue()
{
    c10_npu::assertNotCapturing("Not support Generator while in capture mode");
    graph_expects_this_gen_ = false;
    return offset_intragraph_;
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
    c10_npu::assertNotCapturing("Not support Generator while in capture mode");
    // rounds increment up to the nearest multiple of 4
    increment = ((increment + 3) / 4) * 4;

    return PhiloxNpuState(this->seed_, 0);
}

/**
 * Temporarily accommodates call sites that use philox_engine_inputs.
 * Allows incremental refactor of call sites to use philox_npu_state.
 */
std::pair<uint64_t, uint64_t> NPUGeneratorImpl::philox_engine_inputs(uint64_t increment)
{
    c10_npu::assertNotCapturing("Not support Generator while in capture mode");
    // rounds increment up to the nearest multiple of 4
    increment = ((increment + 3) / 4) * 4;
    // see Note [Why enforce RNG offset % 4 == 0?]
    TORCH_INTERNAL_ASSERT(this->philox_offset_per_thread_ % 4 == 0, PTA_ERROR(ErrCode::INTERNAL));
    uint64_t offset = this->philox_offset_per_thread_;
    this->philox_offset_per_thread_ += increment;
    return std::make_pair(this->seed_, offset);
}

/*
 * Gets the DeviceType of NPUGeneratorImpl.
 * Used for type checking during run time.
 */
c10::DeviceType NPUGeneratorImpl::device_type()
{
    c10_npu::assertNotCapturing("Not support Generator while in capture mode");
    return c10::DeviceType::PrivateUse1;
}

/**
 * Public clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<NPUGeneratorImpl> NPUGeneratorImpl::clone() const
{
    c10_npu::assertNotCapturing("Not support Generator while in capture mode");
    return std::shared_ptr<NPUGeneratorImpl>(this->clone_impl());
}

/**
 * Private clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
NPUGeneratorImpl* NPUGeneratorImpl::clone_impl() const
{
    c10_npu::assertNotCapturing("Not support Generator while in capture mode");
    auto gen = new NPUGeneratorImpl(this->device().index());
    gen->set_current_seed(this->seed_);
    gen->set_philox_offset_per_thread(this->philox_offset_per_thread_);
    return gen;
}

// this is used to register generator
at::Generator make_npu_generator(c10::DeviceIndex device_index)
{
    c10_npu::assertNotCapturing("Not support Generator while in capture mode");
    return at::make_generator<NPUGeneratorImpl>(device_index);
}

REGISTER_GENERATOR_PRIVATEUSE1(make_npu_generator)

} // namespace at_npu
