# ABI=1 torch bridge for FxConverter: torch/python fallback entry points the
# converter reaches through registered function pointers. The two *_abi.h headers
# live in the fxrt public include directory as the single-source POD contract
# shared with the converter's ABI=0 island (fx_converter registers only
# function-table addresses).

# Add fxrt_torch_bridge nanobind sub module. The bridge calls only torch and
# torch_npu C++ APIs -- no fxrt symbols -- so it does not link fxrt_common.
nanobind_add_module(_fxrt_torch_bridge PROTECT_STACK fx_torch_bridge.cc)
target_include_directories(_fxrt_torch_bridge PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_link_libraries(_fxrt_torch_bridge PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

if(ENABLE_ASCEND)
  include_directories(${FXRT_ASCEND_INCLUDE_DIRS})

  add_compile_definitions(ENABLE_TORCH_NPU)
  include_directories(${TORCH_NPU_INCLUDE})
  target_link_directories(_fxrt_torch_bridge PRIVATE ${TORCH_NPU_LIB_PATH})
  target_link_libraries(_fxrt_torch_bridge PUBLIC torch_npu ${FXRT_RUNTIME_LIB})

  # Older torch_npu headers only expose getStreamFromPool / getCurrentNPUStream.
  set(_FX_NPU_STREAM_H "${TORCH_NPU_INCLUDE}/torch_npu/csrc/core/npu/NPUStream.h")
  if(EXISTS "${_FX_NPU_STREAM_H}")
    file(READ "${_FX_NPU_STREAM_H}" _FX_NPU_STREAM_H_CONTENTS)
    if(_FX_NPU_STREAM_H_CONTENTS MATCHES "getStreamFromExternal")
      target_compile_definitions(_fxrt_torch_bridge PRIVATE FXRT_HAS_C10_NPU_GET_STREAM_FROM_EXTERNAL)
      message(STATUS "fxrt bridge: c10_npu::getStreamFromExternal is available")
    endif()
  endif()
endif()

# Must match torch, never the GE ABI=0 island.
target_compile_options(_fxrt_torch_bridge PRIVATE
  -U_GLIBCXX_USE_CXX11_ABI -D_GLIBCXX_USE_CXX11_ABI=1)

set_target_properties(_fxrt_torch_bridge PROPERTIES
  INSTALL_RPATH "${FXRT_MODULE_RPATH}:${TORCH_INSTALL_PREFIX}/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)
