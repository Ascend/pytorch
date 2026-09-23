
file(GLOB OPS_CUSTOM_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
add_library(ops_ascend_custom SHARED ${OPS_CUSTOM_SRC_FILES})

target_link_libraries(ops_ascend_custom PRIVATE
        fxrt_common
        ops_base_obj
        ops_utils
        hardware_ascend
        kernel
        ${CMAKE_DL_LIBS})

set_target_properties(ops_ascend_custom PROPERTIES
        INSTALL_RPATH "${FXRT_LIB_RPATH}"
        BUILD_WITH_INSTALL_RPATH TRUE
)
