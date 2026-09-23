
file(GLOB_RECURSE OPS_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
add_library(ops_obj STATIC ${OPS_SRC_FILES})
target_link_libraries(ops_obj PRIVATE fxrt_common)
