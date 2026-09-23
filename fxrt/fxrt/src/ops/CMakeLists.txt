
if(ENABLE_ASCEND)
    include_directories(${FXRT_ASCEND_INCLUDE_DIRS})
    link_directories(${FXRT_ASCEND_LIB_DIRS})

    if(ENABLE_TORCH_FRONT)
        add_compile_definitions(ENABLE_TORCH_NPU)
        link_directories(${TORCH_NPU_LIB_PATH})
        set(TORCH_NPU_ACL_INCLUDE "${TORCH_NPU_INCLUDE}/third_party/acl/inc")
        if(EXISTS "${TORCH_NPU_ACL_INCLUDE}")
            include_directories(BEFORE ${TORCH_NPU_ACL_INCLUDE})
        endif()
        include_directories(${TORCH_NPU_INCLUDE})
    endif()
endif()

add_subdirectory(op_def)

add_library(kernel SHARED kernel_lib.cpp op_register.cpp custom_op_register.cpp operator.cpp)
target_link_libraries(kernel PRIVATE fxrt_common fxrt_ir_obj ${CMAKE_DL_LIBS})

set_target_properties(kernel PROPERTIES
  INSTALL_RPATH "${FXRT_LIB_RPATH}"
  BUILD_WITH_INSTALL_RPATH TRUE
)

add_subdirectory(utils)

add_subdirectory(op_base)
add_subdirectory(cpu)

if(ENABLE_ASCEND)
    add_subdirectory(ascend)
endif()
