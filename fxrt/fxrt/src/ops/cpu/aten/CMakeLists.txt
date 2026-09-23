file(GLOB_RECURSE ATEN_KERNEL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

add_library(ops_cpu_aten SHARED ${ATEN_KERNEL_SRC_FILES})
target_link_libraries(ops_cpu_aten PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(ops_cpu_aten PRIVATE fxrt_common ops_obj ops_base_obj ops_utils kernel)

set_target_properties(ops_cpu_aten PROPERTIES
  INSTALL_RPATH "${FXRT_LIB_RPATH}"
  BUILD_WITH_INSTALL_RPATH TRUE
)
