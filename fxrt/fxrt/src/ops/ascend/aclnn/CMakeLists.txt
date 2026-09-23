
add_subdirectory(utils)
file(GLOB_RECURSE OPS_ACLNN_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
list(FILTER OPS_ACLNN_SRC_FILES EXCLUDE REGEX "utils/.*")

add_library(ops_ascend_aclnn SHARED ${OPS_ACLNN_SRC_FILES})
target_link_libraries(ops_ascend_aclnn PRIVATE fxrt_common ops_ascend_aclnn_common ops_base_obj
                      hardware_ascend kernel fxrt_profiler)

set_target_properties(ops_ascend_aclnn PROPERTIES
  INSTALL_RPATH "${FXRT_LIB_RPATH}"
  BUILD_WITH_INSTALL_RPATH TRUE
)
