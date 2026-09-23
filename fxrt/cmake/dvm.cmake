# ===========================================================================
# DVM v2 (used by dvm_call_v2)
# ===========================================================================
# fxrt is built in-tree with torch_npu, which builds third_party/dvm/dvm into
# libdvm.a (target dvm_build) and compiles its own inductor/dvm pybind API
# against those headers. dvm_call_v2 hands DVM objects across that boundary, so
# it has to use that very build -- a second DVM copy would collide with it.
set(DVM_V2_ROOT_DIR "${TOP_DIR}/../third_party/dvm/dvm")
set(DVM_V2_INCLUDE_DIR "${DVM_V2_ROOT_DIR}/include"
    CACHE PATH "DVM v2 include directory used by dvm_call_v2" FORCE)
set(DVM_V2_LIBRARY "${DVM_V2_ROOT_DIR}/libdvm.a"
    CACHE FILEPATH "DVM v2 static library used by dvm_call_v2" FORCE)

set(ENABLE_DVM_V2 "AUTO" CACHE STRING "Enable dvm_call_v2 support: AUTO, ON, or OFF")
set_property(CACHE ENABLE_DVM_V2 PROPERTY STRINGS AUTO ON OFF)

function(fxrt_check_dvm_v2_package out_available out_reason)
    # libdvm.a itself is produced by dvm_build during the build, so only the
    # headers and that target can be checked while configuring.
    set(_available TRUE)
    set(_reason "")
    if(NOT EXISTS "${DVM_V2_INCLUDE_DIR}/dvm.h")
        set(_available FALSE)
        set(_reason "dvm.h was not found at ${DVM_V2_INCLUDE_DIR}/dvm.h")
    elseif(NOT EXISTS "${DVM_V2_INCLUDE_DIR}/dvm_py.h")
        set(_available FALSE)
        set(_reason "dvm_py.h was not found at ${DVM_V2_INCLUDE_DIR}/dvm_py.h")
    elseif(NOT TARGET dvm_build)
        set(_available FALSE)
        set(_reason "torch_npu's dvm_build target is not defined")
    endif()
    set(${out_available} ${_available} PARENT_SCOPE)
    set(${out_reason} "${_reason}" PARENT_SCOPE)
endfunction()

function(fxrt_require_dvm_v2)
    fxrt_check_dvm_v2_package(_dvm_v2_available _dvm_v2_reason)
    if(NOT _dvm_v2_available)
        message(FATAL_ERROR "dvm_call_v2 requires the DVM package torch_npu builds from "
                            "third_party/dvm, but it is unavailable: ${_dvm_v2_reason}. "
                            "Run: git submodule update --init third_party/dvm/dvm")
    endif()

    message(STATUS "DVM v2 library: ${DVM_V2_LIBRARY}")
    message(STATUS "DVM v2 include directory: ${DVM_V2_INCLUDE_DIR}")

    if(NOT TARGET fxrt::dvm_v2)
        add_library(fxrt::dvm_v2 STATIC IMPORTED GLOBAL)
        set_target_properties(fxrt::dvm_v2 PROPERTIES
            IMPORTED_LOCATION ${DVM_V2_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${DVM_V2_INCLUDE_DIR}
        )
    endif()
endfunction()
