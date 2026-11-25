# Build fxrt::ir (+ minimal deps) with CXX11 ABI=0 for fx_converter / GE Converter.
# Must NOT link FxRT's ABI=1 shared libs.

# MRT_SRC_ROOT / _FXRT_PKG_ROOT are set by torch_npu_fxrt_core.cmake and point into the
# torch_npu source tree, not the fxrt/src copy sitting next to this file. Recompiling
# torch_npu's own IR sources is what keeps this ABI=0 island in step with the dict that
# torch_npu's ABI=1 exporter produces.
if(NOT MRT_SRC_ROOT OR NOT _FXRT_PKG_ROOT)
  message(FATAL_ERROR "mrt_ir_abi0.cmake: include torch_npu_fxrt_core.cmake first")
endif()

# Meta-only IR: no hardware_abstract / CPU plugin (avoids DeviceContext teardown races).
#
# Listed without an extension: this repository names core sources .cc while torch_npu
# renamed them .cpp on the way in-tree, and a stale list silently yields an empty target
# rather than an error. Resolve each one and fail by name if it is gone.
set(_MRT_IR_ABI0_STEMS
  ir/graph
  ir/value/value
  ir/tensor/tensor
  ir/tensor/storage
  ir/symbolic/symbolic
  ops/op_def/ops_name
  common/logger
  common/dynamic_lib_loader
)
set(MRT_IR_ABI0_SOURCES "")
foreach(_stem ${_MRT_IR_ABI0_STEMS})
  set(_found "")
  foreach(_ext cpp cc)
    if(EXISTS "${MRT_SRC_ROOT}/${_stem}.${_ext}")
      set(_found "${MRT_SRC_ROOT}/${_stem}.${_ext}")
      break()
    endif()
  endforeach()
  if(NOT _found)
    message(FATAL_ERROR
      "mrt_ir_abi0: no ${_stem}.cpp or ${_stem}.cc under ${MRT_SRC_ROOT}. "
      "The fxrt core this converter rebuilds at ABI=0 has moved or been renamed; "
      "update _MRT_IR_ABI0_STEMS.")
  endif()
  list(APPEND MRT_IR_ABI0_SOURCES "${_found}")
endforeach()

add_library(mrt_ir_abi0 STATIC ${MRT_IR_ABI0_SOURCES})
target_include_directories(mrt_ir_abi0 PUBLIC
  ${MRT_SRC_ROOT}
  ${_FXRT_PKG_ROOT}/include
)
target_compile_options(mrt_ir_abi0 PRIVATE
  -U_GLIBCXX_USE_CXX11_ABI -D_GLIBCXX_USE_CXX11_ABI=0
  -UUSE_GLOG
)
target_compile_definitions(mrt_ir_abi0 PRIVATE MRT_IR_ABI0_META_ALLOC=1)
target_link_libraries(mrt_ir_abi0 PUBLIC dl)
set_target_properties(mrt_ir_abi0 PROPERTIES POSITION_INDEPENDENT_CODE ON)
