# Ascend C header dependency

`asc-devkit` is pinned by the gitlink to CANN asc-devkit v9.2.0
(`a63d5261c7c8077a0081e867f43a0e884fec066b`). The existing setup.py
submodule preparation discovers it through `.gitmodules`.

Symmetric-memory ASC compilation uses this source tree's `include/` and
`impl/` directories, without configuring the devkit top-level project or
fetching its nested build dependencies. Its Python tiling generator writes
`kernel_tiling/kernel_tiling.h` under the main build's `symm_mem/generated/`;
source headers and generator inputs participate in incremental rebuilds.
The checkout lives outside `acl_src`, which generate_code.sh cleans.

ACL headers and link-only `libascendcl.so` are reused from the main build:
`generate_code.sh` stages runtime headers, then GE headers take precedence;
`third_party/acl/libs/build_stub.sh` builds the stubs before CMake. Runtime
execution requires the real CANN library, not these stubs. The kernel object
is linked directly into `libtorch_npu.so`, using the main build's ACL linkage.
The compiler-generated host code also requires `libascendc_runtime.a` and its
CANN runtime libraries. These are linked explicitly into the main target;
the ordinary C++ linker does not supply BiSheng's implicit link dependencies.

BiSheng selection remains CMake `BISHENG`, environment `BISHENG`, then
SDK hints / PATH discovery. `ASCEND_HOME_PATH` selects the matching CANN
toolkit for runtime libraries as well as compiler discovery. If it is unset,
the runtime is located relative to the resolved compiler installation
(`bin/`, `tools/bisheng_compiler/bin/` or `compiler/ccec_compiler/bin/`).
An enabled kernel build fails at configuration if these libraries are missing.
BiSheng
installation, supported version pairing and CI deployment are out of scope
for this change and still require validation with the real compiler/device.

When BiSheng is not found during auto-discovery, CMake skips the optional
kernel compilation without requiring an NPU architecture or
Ascend C headers. An explicitly configured invalid `BISHENG` remains an error.
Without BiSheng, the host launch interfaces report that symmetric memory
kernels were not enabled when `barrier()`, `put_signal()` or `wait_signal()`
is called. With BiSheng, the interfaces call the embedded kernels directly.
There is no separate symmetric-memory shared library or runtime symbol lookup.
The launch interfaces and workspace-size query are defined in the `.asc` file;
the host implementation supplies the error-reporting versions when kernels
are not enabled. Both use `NPUSHMEMSymmetricMemoryKernels.hpp` for declarations.
Device primitives, signal state and timeout helpers are private to the `.asc`
implementation; they do not introduce Ascend C dependencies into the host header.
