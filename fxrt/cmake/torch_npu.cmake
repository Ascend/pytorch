# ===========================================================================
# torch_npu dependencies
# ===========================================================================
# fxrt is built only as part of pytorch_npu (add_subdirectory(fxrt/fxrt)), so the
# real `torch_npu` SHARED library target already exists and find_package(Torch)
# has already run in the parent CMakeLists.txt. fxrt links directly to that
# target and CMake handles the build ordering.
#
# ASCEND/CANN libraries: pytorch_npu ships stub .so files (empty implementations
# compiled from third_party/acl/libs/*.cpp by build_stub.sh) and headers in
# third_party/acl/inc/. fxrt uses these stubs at build time; the real CANN
# libraries are loaded at runtime via `import torch_npu`. This means fxrt can
# build on a machine without CANN installed -- same as torch_npu itself.
if(ENABLE_TORCH_FRONT OR ENABLE_ASCEND)
  find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
endif()

if(ENABLE_ASCEND)
  set(TORCH_NPU_INCLUDE "${CMAKE_SOURCE_DIR}")
  set(TORCH_NPU_LIB_PATH "")

  # pytorch_npu stub paths (headers + empty .so implementations)
  set(FXRT_ASCEND_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/third_party/acl/inc")
  set(FXRT_ASCEND_LIB_DIRS "${CMAKE_SOURCE_DIR}/third_party/acl/libs")
  set(FXRT_ASCENDCL_LIB "${FXRT_ASCEND_LIB_DIRS}/libascendcl.so")
  set(FXRT_HCCL_LIB "${FXRT_ASCEND_LIB_DIRS}/libhccl.so")
  # libruntime.so is not shipped as a separate stub; libascendcl.so stub
  # already provides all aclrt* symbols (59 of them in acl.cpp).
  set(FXRT_RUNTIME_LIB "${FXRT_ASCENDCL_LIB}")
  message("TORCH_NPU_INCLUDE: ${TORCH_NPU_INCLUDE}")
  message("FXRT_ASCEND_INCLUDE_DIRS: ${FXRT_ASCEND_INCLUDE_DIRS}")
endif()
