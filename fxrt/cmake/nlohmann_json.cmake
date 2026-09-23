# JSON comes from torch_npu's third_party/nlohmann submodule. fxrt is built
# in-tree as part of torch_npu, which already puts that header on the include
# path, so there is nothing to download here.
set(NLOHMANN_JSON_INC "${TOP_DIR}/../third_party/nlohmann/include")

if(NOT EXISTS "${NLOHMANN_JSON_INC}/nlohmann/json.hpp")
    message(FATAL_ERROR "nlohmann/json.hpp not found under ${NLOHMANN_JSON_INC}. "
                        "Run: git submodule update --init third_party/nlohmann")
endif()

message(STATUS "Using nlohmann_json from ${NLOHMANN_JSON_INC}")

include_directories(${NLOHMANN_JSON_INC})
add_library(fxrt_json INTERFACE)
target_include_directories(fxrt_json INTERFACE ${NLOHMANN_JSON_INC})
add_library(fxrt::json ALIAS fxrt_json)
