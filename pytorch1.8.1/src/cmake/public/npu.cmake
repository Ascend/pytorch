if(NOT TARGET npu_interface)
  add_library(npu_interface INTERFACE)
endif()

set(NPU_BASE_DIRS "${CMAKE_BINARY_DIR}/../third_party/acl")
# Npu headers
set(NPU_INCLUDE_DIRS "${NPU_BASE_DIRS}/inc")
list(APPEND NPU_INCLUDE_DIRS "${NPU_BASE_DIRS}/inc/acl")

link_directories("${NPU_BASE_DIRS}/libs")
link_directories("$ENV{ACL_HOME}/lib64")
link_directories("$ENV{ASCEND_DRIVER_HOME}")

if(${CMAKE_VERSION} VERSION_LESS "3.12.0")
message(FATAL_ERROR "Please consider switch to CMake 3.12.0 or above")
endif()

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0")
  find_package (Python3 COMPONENTS Interpreter Development REQUIRED)
  message("Python3 RUNTIME LIBRAY DIRS: " ${Python3_RUNTIME_LIBRARY_DIRS})
  link_directories(${Python3_RUNTIME_LIBRARY_DIRS})
endif()

target_include_directories(npu_interface INTERFACE ${NPU_INCLUDE_DIRS})

if(USE_HCCL)
  target_link_libraries(npu_interface INTERFACE acl_op_compiler ascendcl hccl python3.7m graph ge_runner)
else()
  target_link_libraries(npu_interface INTERFACE acl_op_compiler ascendcl python3.7m graph ge_runner)
endif()