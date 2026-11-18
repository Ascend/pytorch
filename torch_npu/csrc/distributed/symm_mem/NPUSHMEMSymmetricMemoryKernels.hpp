#pragma once

#include "cstddef"
#include "cstdint"

namespace c10d {
namespace symmetric_memory {

// Keep the device workspace layout behind the host launch interface.
size_t get_npushmem_barrier_workspace_size(int world_size);

// Launchers enqueue on the supplied stream and return the ACL submission error.
int32_t launch_npushmem_put_signal(void* peer_signal_pad, uint64_t signal_index, uint64_t timeout_ms, void* stream);
int32_t launch_npushmem_wait_signal(void* local_signal_pad, uint64_t signal_index, uint64_t timeout_ms, void* stream);
int32_t launch_npushmem_barrier(
    void** signal_pads,
    void* peer_states,
    int channel,
    int rank,
    int world_size,
    uint64_t timeout_ms,
    void* stream);

} // namespace symmetric_memory
} // namespace c10d
