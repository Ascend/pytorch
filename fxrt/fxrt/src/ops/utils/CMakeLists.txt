
file(GLOB_RECURSE OPS_UTILS_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
add_library(ops_utils SHARED ${OPS_UTILS_SRC_FILES})
target_link_libraries(ops_utils PRIVATE fxrt_common fxrt_ir_obj ${CMAKE_DL_LIBS})
target_link_libraries(ops_utils PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_link_libraries(ops_utils PUBLIC Python::Module)

if(ENABLE_ASCEND)
    add_compile_definitions(ENABLE_TORCH_NPU)
    include_directories(${TORCH_NPU_INCLUDE})
    target_link_directories(ops_utils PUBLIC ${TORCH_NPU_LIB_PATH})
    target_link_libraries(ops_utils PUBLIC torch_npu hardware_abstract)
endif()

target_link_options(ops_utils PRIVATE -Wl,--as-needed)

set_target_properties(ops_utils PROPERTIES
        INSTALL_RPATH "${FXRT_LIB_RPATH}"
        BUILD_WITH_INSTALL_RPATH TRUE
)
