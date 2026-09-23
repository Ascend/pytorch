find_package(Threads REQUIRED)

file(GLOB_RECURSE RUNTIME_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
add_library(execution_engine SHARED ${RUNTIME_SRC_FILES})

target_link_libraries(execution_engine ops_obj ops_utils fxrt_ir_obj kernel hardware_abstract pass_obj fxrt_common
                      Threads::Threads Python::Module nanobind-static)
target_include_directories(execution_engine PUBLIC ${NANOBIND_INCLUDE_DIR})

set_target_properties(execution_engine PROPERTIES
  INSTALL_RPATH "${FXRT_LIB_RPATH}"
  BUILD_WITH_INSTALL_RPATH TRUE
)
