
file(GLOB_RECURSE IR_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
add_library(fxrt_ir_obj STATIC ${IR_SRC_FILES})
target_link_libraries(fxrt_ir_obj hardware_abstract fxrt_common)
