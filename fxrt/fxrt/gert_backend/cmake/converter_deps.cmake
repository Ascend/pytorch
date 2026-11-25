# Prepare GE/runtime submodule headers and pinned metadef headers.
get_filename_component(FXC_THIRD_PARTY_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../third_party" ABSOLUTE)
if(NOT Python_EXECUTABLE)
  find_package(Python REQUIRED COMPONENTS Interpreter)
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${FXC_THIRD_PARTY_DIR}/converter_deps.json"
  "${FXC_THIRD_PARTY_DIR}/prepare_converter_deps.py")
execute_process(
  COMMAND "${Python_EXECUTABLE}" "${FXC_THIRD_PARTY_DIR}/prepare_converter_deps.py"
  RESULT_VARIABLE _FXC_DEPS_RESULT)
if(NOT _FXC_DEPS_RESULT EQUAL 0)
  message(FATAL_ERROR "Cannot prepare converter dependencies in ${FXC_THIRD_PARTY_DIR}")
endif()
set(FXC_GE_INCLUDE_DIR "${FXC_THIRD_PARTY_DIR}/ge/include")
set(FXC_CANN_HEADER_DIR "${FXC_THIRD_PARTY_DIR}/cann")
# Keep the upstream header layouts separate instead of merging them into cann/.
set(FXC_VENDORED_INCLUDE_DIRS
  "${FXC_THIRD_PARTY_DIR}/securec/include")
# Reuse torch-npu's nlohmann submodule, as the main FXRT build already does.
set(FXC_JSON_INCLUDE_DIR "${FXC_TORCH_NPU_SOURCE_DIR}/third_party/nlohmann/include")
if(NOT EXISTS "${FXC_JSON_INCLUDE_DIR}/nlohmann/json.hpp")
  message(FATAL_ERROR "nlohmann/json.hpp not found under ${FXC_JSON_INCLUDE_DIR}. "
                      "Run: git submodule update --init third_party/nlohmann")
endif()
message(STATUS "gert_backend GE headers: ${FXC_GE_INCLUDE_DIR}")
