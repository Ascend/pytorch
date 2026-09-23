
add_subdirectory(symbolic)

if(ENABLE_TORCH_FRONT)
    add_subdirectory(aten)
endif()
