
file(GLOB_RECURSE COMMON_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
add_library(fxrt_common SHARED ${COMMON_SRC_FILES})
target_link_libraries(fxrt_common dl)

# Logs are written through torch_npu's npu_logging, so the "fxrt" log alias of
# TORCH_NPU_LOGS controls them. Builds without torch_npu fall back to stderr.
if(ENABLE_ASCEND)
  target_compile_definitions(fxrt_common PRIVATE ENABLE_TORCH_NPU)
  target_include_directories(fxrt_common PRIVATE ${TORCH_NPU_INCLUDE} ${FXRT_ASCEND_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
  target_link_directories(fxrt_common PRIVATE ${TORCH_NPU_LIB_PATH})
  target_link_libraries(fxrt_common torch_npu)
endif()

set_target_properties(fxrt_common PROPERTIES
  INSTALL_RPATH "${FXRT_LIB_RPATH}"
  BUILD_WITH_INSTALL_RPATH TRUE
)
