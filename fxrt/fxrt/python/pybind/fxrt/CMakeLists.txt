
# Add fxrt_api nanobind sub module
nanobind_add_module(_fxrt_api pybind_api.cpp)
target_link_libraries(_fxrt_api PUBLIC fxrt)

set_target_properties(_fxrt_api PROPERTIES
  INSTALL_RPATH "${FXRT_MODULE_RPATH}"
  BUILD_WITH_INSTALL_RPATH TRUE
)

# Add fxrt_ir nanobind sub module
nanobind_add_module(_fxrt_ir pybind_ir.cpp)
target_link_libraries(_fxrt_ir PUBLIC fxrt)
set_target_properties(_fxrt_ir PROPERTIES
  INSTALL_RPATH "${FXRT_MODULE_RPATH}"
  BUILD_WITH_INSTALL_RPATH TRUE
)

# Add collective nanobind sub module
nanobind_add_module(_fxrt_collective pybind_collective.cpp)
target_link_libraries(_fxrt_collective PRIVATE fxrt)
set_target_properties(_fxrt_collective PROPERTIES
  INSTALL_RPATH "${FXRT_MODULE_RPATH}"
  BUILD_WITH_INSTALL_RPATH TRUE
)

# Add fxrt_config nanobind sub module
nanobind_add_module(_fxrt_config pybind_config.cpp)
target_link_libraries(_fxrt_config PRIVATE fxrt)
set_target_properties(_fxrt_config PROPERTIES
  INSTALL_RPATH "${FXRT_MODULE_RPATH}"
  BUILD_WITH_INSTALL_RPATH TRUE
)

# Add aclgraph_config nanobind sub module
nanobind_add_module(_fxrt_aclgraph_config pybind_aclgraph_config.cpp)
target_link_libraries(_fxrt_aclgraph_config PRIVATE fxrt_aclgraph_config)
set_target_properties(_fxrt_aclgraph_config PROPERTIES
  INSTALL_RPATH "${FXRT_MODULE_RPATH}"
  BUILD_WITH_INSTALL_RPATH TRUE
)
