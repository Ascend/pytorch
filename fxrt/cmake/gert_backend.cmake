# Build FXConverter in the main FXRT CMake graph. Its targets stay separate from
# FXRT's libraries, while Ninja schedules both projects in one build.
if(NOT TARGET torch_npu)
  message(FATAL_ERROR "FXConverter requires the in-tree torch_npu target")
endif()
if(NOT DEFINED PYTORCH_INSTALL_DIR OR NOT DEFINED FXRT_INSTALL_DST)
  message(FATAL_ERROR "FXConverter requires PYTORCH_INSTALL_DIR and FXRT_INSTALL_DST")
endif()

set(FXC_TORCH_ROOT "${PYTORCH_INSTALL_DIR}")
set(FXC_TORCH_NPU_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
set(FXC_TORCH_NPU_LIBRARY "$<TARGET_FILE:torch_npu>")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/gert_backend"
                 "${CMAKE_CURRENT_BINARY_DIR}/gert_backend")
