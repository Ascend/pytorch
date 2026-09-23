
file(GLOB_RECURSE OPS_COMPOSITE_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
add_library(ops_ascend_composite SHARED ${OPS_COMPOSITE_SRC_FILES})
target_link_libraries(ops_ascend_composite PRIVATE
        fxrt_common
        ops_ascend_aclnn
        ops_base_obj
        hardware_ascend
        kernel)

set_target_properties(ops_ascend_composite PROPERTIES
        INSTALL_RPATH "${FXRT_LIB_RPATH}"
        BUILD_WITH_INSTALL_RPATH TRUE
)
