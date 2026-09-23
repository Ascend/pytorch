
file(GLOB_RECURSE OPS_SYMBOLIC_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

add_library(ops_cpu_symbolic SHARED ${OPS_SYMBOLIC_SRC_FILES})
target_link_libraries(ops_cpu_symbolic PRIVATE fxrt_common ops_base_obj kernel)

set_target_properties(ops_cpu_symbolic PROPERTIES
  INSTALL_RPATH "${FXRT_LIB_RPATH}"
  BUILD_WITH_INSTALL_RPATH TRUE
)
