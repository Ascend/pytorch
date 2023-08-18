set(PROFILER_NAME npu_profiler)

FILE(GLOB NPU_PROF_SRCS
    common/*.cpp
    src/*.cpp
)

set(NPU_PROF_INC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

add_library(${PROFILER_NAME} SHARED
    ${NPU_PROF_SRCS}
)

target_include_directories(${PROFILER_NAME} PRIVATE
    ${NPU_PROF_INC}
)

target_link_libraries(${PROFILER_NAME} PRIVATE torch_cpu pthread)

target_compile_options(${PROFILER_NAME} PRIVATE
    ${TORCH_CXX_FLAGS}
)
