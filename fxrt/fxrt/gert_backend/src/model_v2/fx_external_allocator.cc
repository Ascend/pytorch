#include "api/fx_allocator_bridge.h"
#include "model_v2/fx_external_allocator_internal.h"

#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>

#include "common/logger.h"
#include "ge/ge_allocator.h"  // public: ge::Allocator / ge::MemBlock
#include "ge/ge_api_v2.h"

// Turns the ABI=1 callbacks registered through FxSetExternalAllocator into the
// gert::Allocators bundle used by GE's external device allocator feed.
//
// Why this works without touching GE: SelectL1Allocator (runtime/v2/kernel/memory/
// memory_kernel.cc) prefers whatever Allocators the feed carries, and the only thing it
// asks of the allocator is that it be a ge::Allocator -- its dynamic_cast to
// CachingMemAllocator may fail, in which case it merely skips SetStream. So a plain
// gert::ExternalAllocator (which forwards every Malloc/Free to our callbacks) is a
// first-class L1 allocator, and every node-output / workspace / shape-buffer
// allocation for this executor lands in the torch_npu caching allocator.
//
// The bundle is built here, with ordinary calls into libgert.so. That is safe because
// this library links against libgert.so and no prebuilt GE object files: every gert symbol
// resolves from the shared library, so this image never carries its own copy of
// gert::ExternalAllocator / gert::Allocators typeinfo. Two failure modes depend on that:
//
//   * a second copy of that typeinfo in this image would make SelectL1Allocator (running
//     inside libgert.so) crash in __dynamic_cast on the foreign typeinfo;
//   * a second copy of GE's runtime objects would also carry their static initializers and
//     TLS globals, which corrupt the heap and abort at interpreter exit
//     ("corrupted size vs. prev_size").
//
// cmake/check_gert_typeinfo.cmake enforces this on the built artifact, so a regression is a
// build error instead of a runtime crash.

namespace fx_converter {
namespace {

// GE hands us the raw block pointer from alloc_fn and expects the same value back, so the
// identity mapping is the correct default -- alloc_fn returns the address itself.
void *IdentityGetAddr(void *block) {
  return block;
}

struct AllocatorHolder {
  std::mutex mutex;
  FxExternalAllocatorDesc desc{nullptr, nullptr, nullptr, nullptr};

  static AllocatorHolder &Instance() {
    static AllocatorHolder instance;
    return instance;
  }

  bool Ready() {
    std::lock_guard<std::mutex> lock(mutex);
    return desc.alloc_fn != nullptr && desc.free_fn != nullptr;
  }

  bool GetDesc(FxExternalAllocatorDesc *out) {
    std::lock_guard<std::mutex> lock(mutex);
    if (desc.alloc_fn == nullptr || desc.free_fn == nullptr) {
      return false;
    }
    *out = desc;
    return true;
  }

  // Stores the callbacks only. The gert::Allocator object is built by libgert.so in
  // FxLibgertAllocators(); building one here would drag the GE runtime objects into this
  // .so (see the file header comment).
  void Set(const FxExternalAllocatorDesc &new_desc) {
    std::lock_guard<std::mutex> lock(mutex);
    desc = new_desc;
    if (desc.get_addr_fn == nullptr) {
      desc.get_addr_fn = &IdentityGetAddr;
    }
    RT_GLOG(INFO) << "FxSetExternalAllocator: device HBM allocations now come from the external allocator";
  }
};

struct StreamAllocatorRegistry {
  std::mutex mutex;
  std::unordered_map<const void *, size_t> registrations;

  static StreamAllocatorRegistry &Instance() {
    static StreamAllocatorRegistry registry;
    return registry;
  }
};

}  // namespace

// Allocator built purely from CANN's public ge_allocator.h surface. The previous
// implementation called gert::CreateExternalAllocator with a ge::AllocatorDesc, but both
// live in GE-internal headers (framework/runtime/gert_api.h,
// framework/memory/allocator_desc.h) that are not packaged into CANN -- only the GE source
// tree has them. ge::Allocator and ge::MemBlock are public, so deriving from them keeps
// this library linkable against a stock CANN install.
//
// Behaviour matches GE's own gert::ExternalAllocator: every Malloc forwards to the
// registered callbacks, the address handed to GE comes from get_addr_from_block_func, and
// Free forwards the raw block that alloc_fn returned.
//
// A plain ge::MemBlock is used directly -- no subclass. The only registrar of this bridge is
// FxRegisterTorchPoolAllocator, and its get_addr_fn (FxTorchPoolGetAddr) is the identity, so
// MemBlock's addr_ *is* the raw block that alloc_fn returned. Keeping a second copy of it in
// a subclass, plus a dynamic_cast to recover it in Free, bought nothing:
//   * MemBlock is concrete and publicly constructible (ge/ge_allocator.h), so it needs no
//     derived type to be instantiated;
//   * Free gets the raw address straight from block->GetAddr(), with no cast at all.
// Free simply deletes the block: MemBlock's destructor is virtual, so the delete is correct
// for whatever this allocator handed out.
class FxTorchPoolAllocatorImpl : public ge::Allocator {
 public:
  explicit FxTorchPoolAllocatorImpl(const FxExternalAllocatorDesc &desc) : desc_(desc) {}

  ge::MemBlock *Malloc(size_t size) override {
    void *raw_block = desc_.alloc_fn(desc_.obj, size);
    if (raw_block == nullptr) {
      RT_GLOG(WARNING) << "FxTorchPoolAllocatorImpl: external malloc failed, size=" << size;
      return nullptr;
    }
    auto *block = new (std::nothrow) ge::MemBlock(*this, desc_.get_addr_fn(raw_block), size);
    if (block == nullptr) {
      desc_.free_fn(desc_.obj, raw_block);
      return nullptr;
    }
    return block;
  }

  // alloc_advise_func is never set by this bridge, so the default MemBlock path (plain
  // Malloc) is exactly what gert::ExternalAllocator did; no override needed.

  void Free(ge::MemBlock *block) override {
    if (block == nullptr) {
      return;
    }
    // get_addr_fn is the identity (see above), so GetAddr() returns the raw block.
    desc_.free_fn(desc_.obj, const_cast<void *>(block->GetAddr()));
    delete block;
  }

 private:
  FxExternalAllocatorDesc desc_;
};

std::shared_ptr<ge::Allocator> FxTorchPoolDeviceAllocator() {
  static std::mutex mutex;
  static std::shared_ptr<ge::Allocator> device_allocator;
  std::lock_guard<std::mutex> lock(mutex);
  if (device_allocator != nullptr) {
    return device_allocator;
  }

  FxExternalAllocatorDesc fx_desc{};
  if (!AllocatorHolder::Instance().GetDesc(&fx_desc)) {
    return nullptr;
  }

  device_allocator = std::make_shared<FxTorchPoolAllocatorImpl>(fx_desc);
  RT_GLOG(INFO) << "FxTorchPoolDeviceAllocator: built from public ge::Allocator";
  return device_allocator;
}

bool FxRegisterExternalAllocatorForStream(ge::GeSession *session, const void *stream) {
  if (session == nullptr || stream == nullptr) {
    return false;
  }
  const auto allocator = FxTorchPoolDeviceAllocator();
  if (allocator == nullptr) {
    return false;
  }

  auto &registry = StreamAllocatorRegistry::Instance();
  std::lock_guard<std::mutex> lock(registry.mutex);
  auto registration = registry.registrations.find(stream);
  if (registration != registry.registrations.end()) {
    ++registration->second;
    return true;
  }
  const ge::Status status = session->RegisterExternalAllocator(stream, allocator);
  if (status != ge::SUCCESS) {
    RT_GLOG(WARNING) << "FxRegisterExternalAllocatorForStream: GE registration failed, status=" << status;
    return false;
  }
  registry.registrations.emplace(stream, 1U);
  return true;
}

bool FxUnregisterExternalAllocatorForStream(ge::GeSession *session, const void *stream) {
  if (session == nullptr || stream == nullptr) {
    return false;
  }
  auto &registry = StreamAllocatorRegistry::Instance();
  std::lock_guard<std::mutex> lock(registry.mutex);
  auto registration = registry.registrations.find(stream);
  if (registration == registry.registrations.end()) {
    return true;
  }
  if (--registration->second != 0U) {
    return true;
  }
  const ge::Status status = session->UnregisterExternalAllocator(stream);
  registry.registrations.erase(registration);
  if (status != ge::SUCCESS) {
    RT_GLOG(WARNING) << "FxUnregisterExternalAllocatorForStream: GE unregistration failed, status=" << status;
    return false;
  }
  return true;
}

extern "C" __attribute__((visibility("default"))) void FxSetExternalAllocator(
    const FxExternalAllocatorDesc *desc) {
  if (desc == nullptr || desc->alloc_fn == nullptr || desc->free_fn == nullptr) {
    RT_GLOG(WARNING) << "FxSetExternalAllocator: incomplete descriptor ignored, GE keeps its own pool";
    return;
  }
  AllocatorHolder::Instance().Set(*desc);
}

extern "C" __attribute__((visibility("default"))) bool FxHasExternalAllocator() {
  return AllocatorHolder::Instance().Ready();
}

extern "C" __attribute__((visibility("default"))) bool FxGetExternalAllocatorDesc(
    FxExternalAllocatorDesc *out) {
  if (out == nullptr) {
    return false;
  }
  return AllocatorHolder::Instance().GetDesc(out);
}

// gert::Allocators is a GE-internal type (framework/runtime/mem_allocator.h), so this
// library no longer builds one. GE does that itself: GraphManager::GetExecuteGraphAllocators
// takes the AllocatorPtr registered on the stream and fills a gert::Allocators from it
// (one SetAllocator per usage slot for kOnDeviceHbm), which is what both the OM2 path in
// ExecuteGraphWithStream and GetUserAllocatorOrFixedBaseAllocator consume. Registering our
// public-header ge::Allocator subclass through GeSession::RegisterExternalAllocator is
// therefore the whole contract.


}  // namespace fx_converter
