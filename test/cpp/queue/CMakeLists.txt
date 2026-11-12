cmake_minimum_required(VERSION 3.16)
project(sequence_queue_correctness LANGUAGES CXX)

get_filename_component(QUEUE_REPOSITORY_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(QUEUE_SOURCE_ROOT "${QUEUE_REPOSITORY_ROOT}" CACHE PATH
    "Root containing torch_npu/csrc/core/npu queue headers; override for historical RED")
foreach(header ExperimentalSequenceMpscQueueCore.h OrderedMpscQueueCore.h)
    if(NOT EXISTS "${QUEUE_SOURCE_ROOT}/torch_npu/csrc/core/npu/${header}")
        message(FATAL_ERROR "Missing queue header: ${QUEUE_SOURCE_ROOT}/torch_npu/csrc/core/npu/${header}")
    endif()
endforeach()

find_package(Threads REQUIRED)
add_executable(sequence_mpsc_queue_test sequence_mpsc_queue_test.cpp)
target_compile_features(sequence_mpsc_queue_test PRIVATE cxx_std_17)
target_include_directories(sequence_mpsc_queue_test PRIVATE "${QUEUE_SOURCE_ROOT}")
target_link_libraries(sequence_mpsc_queue_test PRIVATE Threads::Threads)
if(MSVC)
    target_compile_options(sequence_mpsc_queue_test PRIVATE /W4 /WX)
else()
    target_compile_options(sequence_mpsc_queue_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

enable_testing()
set(QUEUE_CASES
    capacity-rejected claimed-last-drain inspect-gate-consumer failed-consumer-drain
    producer-before-wait-throws producer-after-wait-throws
    consumer-before-wait-throws consumer-after-wait-throws
    stop-full copy-throws discard discard-throws restart
    stress-2 stress-8 stress-64 stress-4096)
foreach(case_name IN LISTS QUEUE_CASES)
    add_test(NAME sequence_queue.${case_name} COMMAND sequence_mpsc_queue_test ${case_name})
    set_tests_properties(sequence_queue.${case_name} PROPERTIES TIMEOUT 20 LABELS "queue;correctness")
endforeach()
