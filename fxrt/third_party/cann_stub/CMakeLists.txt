cmake_minimum_required(VERSION 3.12)
project(fxrt_cann_stub LANGUAGES CXX)

# Targets deliberately have no install() rule. Only the converter DSOs enter
# the wheel; their DT_NEEDED entries name the real CANN SONAMEs.
function(fxc_add_cann_stub name soname_version)
  set(_fxc_source "${CMAKE_CURRENT_SOURCE_DIR}/${name}.cc")
  if(NOT EXISTS "${_fxc_source}")
    set(_fxc_source "${CMAKE_CURRENT_SOURCE_DIR}/empty.cc")
  endif()
  add_library(fxc_stub_${name} SHARED "${_fxc_source}")
  target_compile_options(fxc_stub_${name} PRIVATE -U_GLIBCXX_USE_CXX11_ABI -D_GLIBCXX_USE_CXX11_ABI=0)
  target_compile_definitions(fxc_stub_${name} PRIVATE
    nlohmann=ascend_nlohmann)
  target_include_directories(fxc_stub_${name} BEFORE PRIVATE
    ${FXC_GE_INCLUDE_DIR}/inc/graph_metadef
    ${FXC_GE_INCLUDE_DIR}/graph_metadef
    ${FXC_GE_INCLUDE_DIR}/graph_metadef/graph
    ${FXC_CANN_HEADER_DIR}/include
    ${FXC_CANN_HEADER_DIR}/include/ge
    ${FXC_VENDORED_INCLUDE_DIRS}
    ${FXC_CANN_HEADER_DIR}/pkg_inc
    ${FXC_CANN_HEADER_DIR}/pkg_inc/runtime
    ${FXC_CANN_HEADER_DIR}/pkg_inc/base
    ${FXC_CANN_HEADER_DIR}/pkg_inc/common
    ${FXC_CANN_HEADER_DIR}/pkg_inc/common/ge_common
    ${FXC_CANN_HEADER_DIR}/pkg_inc/profiling
    ${FXC_GE_INCLUDE_DIR}/base
    ${FXC_GE_INCLUDE_DIR}
    ${FXC_GE_INCLUDE_DIR}/runtime/v2
    ${FXC_GE_INCLUDE_DIR}/inc
    ${FXC_GE_INCLUDE_DIR}/inc/external
    ${FXC_GE_INCLUDE_DIR}/inc/framework
    ${FXC_GE_INCLUDE_DIR}/inc/graph_metadef/external
    ${FXC_GE_INCLUDE_DIR}/inc/graph_metadef/exe_graph
    ${FXC_GE_INCLUDE_DIR}/inc/graph_metadef/common
    ${FXC_JSON_INCLUDE_DIR})
  set_target_properties(fxc_stub_${name} PROPERTIES
    OUTPUT_NAME "${name}"
    CXX_VISIBILITY_PRESET default
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
    SKIP_BUILD_RPATH TRUE)
  if(NOT "${soname_version}" STREQUAL "")
    set_target_properties(fxc_stub_${name} PROPERTIES
      VERSION "${soname_version}" SOVERSION "${soname_version}")
  endif()
  set(FXC_CANN_STUB_LIBRARIES ${FXC_CANN_STUB_LIBRARIES} fxc_stub_${name} PARENT_SCOPE)
endfunction()

fxc_add_cann_stub(eager_style_graph_builder_base "")
fxc_add_cann_stub(graph "")
fxc_add_cann_stub(graph_base "")
fxc_add_cann_stub(opp_registry "")
fxc_add_cann_stub(metadef "")
fxc_add_cann_stub(ge_common "")
fxc_add_cann_stub(ascendcl "")
fxc_add_cann_stub(platform "")
fxc_add_cann_stub(lowering "")
fxc_add_cann_stub(ge_runner_v2 "")
fxc_add_cann_stub(c_sec "")
set(FXC_CANN_STUB_LIBRARIES ${FXC_CANN_STUB_LIBRARIES} PARENT_SCOPE)
