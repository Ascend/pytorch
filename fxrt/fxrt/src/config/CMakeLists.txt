
set(CONFIG_SRC_FILES
  device/ascend/op_precision_conf.cpp
)
add_library(fxrt_config SHARED ${CONFIG_SRC_FILES})
target_link_libraries(fxrt_config PRIVATE fxrt_common)

set_target_properties(fxrt_config PROPERTIES
  INSTALL_RPATH "${FXRT_LIB_RPATH}"
  BUILD_WITH_INSTALL_RPATH TRUE
)

add_library(fxrt_aclgraph_config SHARED
  device/ascend/aclgraph_conf.cpp
)
target_link_libraries(fxrt_aclgraph_config PRIVATE execution_engine)

set_target_properties(fxrt_aclgraph_config PROPERTIES
  INSTALL_RPATH "${FXRT_LIB_RPATH}"
  BUILD_WITH_INSTALL_RPATH TRUE
)
