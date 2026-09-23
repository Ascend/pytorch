# DVM operator module CMakeLists.txt
#
# Only dvm_call_v2 lives here; it uses the DVM package torch_npu builds from
# third_party/dvm (see cmake/dvm.cmake).

set(DVM_V2_SRCS op_dvm_call_v2.cpp op_dvm_call_v2_register.cpp)

string(TOUPPER "${ENABLE_DVM_V2}" _enable_dvm_v2)
if(NOT _enable_dvm_v2 STREQUAL "AUTO" AND NOT _enable_dvm_v2 STREQUAL "ON" AND NOT _enable_dvm_v2 STREQUAL "OFF")
    message(FATAL_ERROR "ENABLE_DVM_V2 must be AUTO, ON, or OFF, but got: ${ENABLE_DVM_V2}")
endif()

set(_build_dvm_v2 FALSE)
set(_dvm_v2_disable_reason "")
if(_enable_dvm_v2 STREQUAL "OFF")
    set(_dvm_v2_disable_reason "ENABLE_DVM_V2=OFF")
else()
    fxrt_check_dvm_v2_package(_dvm_v2_package_available _dvm_v2_package_reason)
    set(DVM_TORCH_NPU_PYBIND_API "${TORCH_NPU_INCLUDE}/torch_npu/csrc/inductor/dvm/pybind_api.h")
    if(NOT _dvm_v2_package_available)
        set(_dvm_v2_disable_reason "${_dvm_v2_package_reason}")
    elseif(NOT TORCH_NPU_INCLUDE)
        set(_dvm_v2_disable_reason "TORCH_NPU_INCLUDE is not configured")
    elseif(NOT EXISTS "${DVM_TORCH_NPU_PYBIND_API}")
        set(_dvm_v2_disable_reason "torch-npu DVM pybind header was not found at ${DVM_TORCH_NPU_PYBIND_API}")
    else()
        file(STRINGS "${DVM_TORCH_NPU_PYBIND_API}" _dvm_torch_npu_shape_refs REGEX "ShapeRefs[ \t]*\\(")
        if("${_dvm_torch_npu_shape_refs}" STREQUAL "")
            set(_dvm_v2_disable_reason
                "torch-npu DVM pybind header does not provide DynKernelPy::ShapeRefs required by dvm_call_v2: "
                "${DVM_TORCH_NPU_PYBIND_API}")
        else()
            set(_build_dvm_v2 TRUE)
        endif()
    endif()
endif()

if(NOT _build_dvm_v2)
    if(_enable_dvm_v2 STREQUAL "ON")
        message(FATAL_ERROR "DVM V2 is required by ENABLE_DVM_V2=ON, but it is unavailable: "
                            "${_dvm_v2_disable_reason}")
    endif()
    message(STATUS "DVM V2 operator module disabled: ${_dvm_v2_disable_reason}")
else()
    add_library(ops_ascend_dvm_v2 SHARED ${DVM_V2_SRCS})

    fxrt_require_dvm_v2()
    # libdvm.a is produced by torch_npu's dvm_build target.
    add_dependencies(ops_ascend_dvm_v2 dvm_build)

    target_include_directories(ops_ascend_dvm_v2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../..
        ${DVM_V2_INCLUDE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${TORCH_NPU_INCLUDE}
    )

    target_link_libraries(ops_ascend_dvm_v2 PRIVATE
        fxrt::dvm_v2
        ops_ascend_aclnn_common
        ops_utils
        torch_npu
        hardware_ascend
        kernel
        fxrt_ir_obj
        Python::Module
    )

    target_compile_options(ops_ascend_dvm_v2 PRIVATE
        -Wall
        -Werror
        -Wno-error=unused-function
        -Wno-error=unused-variable
    )

    set_target_properties(ops_ascend_dvm_v2 PROPERTIES
      INSTALL_RPATH "${FXRT_LIB_RPATH}"
      BUILD_WITH_INSTALL_RPATH TRUE
    )

endif()

message(STATUS "DVM operator module configured successfully")
