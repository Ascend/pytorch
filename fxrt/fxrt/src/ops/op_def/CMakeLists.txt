
file(GLOB_RECURSE OPS_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
add_library(ops_obj STATIC ${OPS_SRC_FILES})
target_link_libraries(ops_obj PRIVATE fxrt_common)

# ops_name.cpp #includes ops.list to generate the Op enum. CMake does not see
# through #include, so without this every ops.list edit reused a stale
# ops_name.cpp.o and the enum silently lagged the file -- a new OP(x) stayed
# unknown to Python as Op.x, and the failure surfaced far away as an
# AttributeError during `import fx_converter`.
set_source_files_properties(ops_name.cpp PROPERTIES OBJECT_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/ops.list")
