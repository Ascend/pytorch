
file(GLOB_RECURSE OPS_HCCL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

add_library(ops_ascend_hccl SHARED ${OPS_HCCL_SRC_FILES})
target_link_libraries(ops_ascend_hccl PRIVATE fxrt_common ops_base_obj hardware_ascend kernel ${FXRT_HCCL_LIB})

set_target_properties(ops_ascend_hccl PROPERTIES
  INSTALL_RPATH "${FXRT_LIB_RPATH}"
  BUILD_WITH_INSTALL_RPATH TRUE
)
