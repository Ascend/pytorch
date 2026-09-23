
# ATB is an optional Ascend library that is not part of the CANN stub set
# shipped by pytorch_npu.  In standalone mode it is looked up under the
# CANN install; in integration mode it is simply skipped (fxrt builds
# without ATB support, matching torch_npu).
if(TARGET torch_npu)
    message(STATUS "ATB skipped in integration mode (no stub available)")
    return()
endif()

if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(ASCEND_PATH_BASE $ENV{ASCEND_CUSTOM_PATH})
    if(ASCEND_PATH_BASE MATCHES "(cann|ascend-toolkit)$")
        get_filename_component(ASCEND_PATH_BASE ${ASCEND_PATH_BASE} DIRECTORY)
    endif()
else()
    set(ASCEND_PATH_BASE /usr/local/Ascend)
endif()

set(ASCEND_ATB_PATH ${ASCEND_PATH_BASE}/nnal/atb/latest/atb/cxx_abi_0)

# TODO(jiaorui)
if(NOT EXISTS ${ASCEND_ATB_PATH})
    message(WARNING "ATB not found at ${ASCEND_ATB_PATH}, skipping ATB configuration")
    return()
endif()

file(GLOB_RECURSE OPS_ATB_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
include_directories(${ASCEND_ATB_PATH}/include)
find_library(ATB_KERNEL atb ${ASCEND_ATB_PATH}/lib)
add_library(ops_ascend_atb SHARED ${OPS_ATB_SRC_FILES})
target_link_libraries(ops_ascend_atb PRIVATE ${ATB_KERNEL}
        fxrt_common ops_ascend_aclnn ops_base_obj ops_utils hardware_ascend kernel)

set_target_properties(ops_ascend_atb PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS ops_ascend_atb
        LIBRARY DESTINATION ${FXRT_LIB_DIR}
        RUNTIME DESTINATION ${FXRT_BIN_DIR})
