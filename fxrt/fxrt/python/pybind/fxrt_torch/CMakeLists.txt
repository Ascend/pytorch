
# Add fxrt_torch nanobind sub module
nanobind_add_module(_fxrt_torch pybind_torch.cpp)
target_link_libraries(_fxrt_torch PUBLIC fxrt_common fxrt ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

if(ENABLE_ASCEND)
  include_directories(${FXRT_ASCEND_INCLUDE_DIRS})

  add_compile_definitions(ENABLE_TORCH_NPU)
  include_directories(${TORCH_NPU_INCLUDE})
  target_link_directories(_fxrt_torch PRIVATE ${TORCH_NPU_LIB_PATH})
  target_link_libraries(_fxrt_torch PUBLIC torch_npu hardware_abstract execution_engine)
endif()

set_target_properties(_fxrt_torch PROPERTIES
  INSTALL_RPATH "${FXRT_MODULE_RPATH}:${TORCH_INSTALL_PREFIX}/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)
