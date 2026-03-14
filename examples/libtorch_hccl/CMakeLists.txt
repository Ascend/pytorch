cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(HCCLAllreduceExample CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

# 定义构建类型
IF(CMAKE_BUILD_TYPE MATCHES Debug)
  message("Debug构建")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_DEBUG")
ELSEIF(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  message("RelWithDebInfo构建")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
ELSE()
  message("Release构建")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
ENDIF()

SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
SET(CMAKE_CXX_FLAGS_RELEASE "-O2")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# 自动查找libtorch路径
# 优先使用环境变量，如果未设置则自动从Python获取
if(DEFINED ENV{TORCH_INSTALL_DIR})
  set(torch_path "$ENV{TORCH_INSTALL_DIR}")
else()
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE torch_path
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()

if(NOT EXISTS ${torch_path})
  message(FATAL_ERROR "未找到libtorch安装目录，请设置环境变量TORCH_INSTALL_DIR或确保Python环境中已安装torch")
endif()
message(STATUS "libtorch路径: ${torch_path}")
include_directories(${torch_path}/include)
# c10d/ProcessGroup.hpp
include_directories(${torch_path}/include/torch/csrc/distributed)
# torch/torch.h
include_directories(${torch_path}/include/torch/csrc/api/include)
link_directories(${torch_path}/lib)


# 使用环境变量查找libtorch_npu路径
if(DEFINED ENV{TORCH_NPU_INSTALL_DIR})
  set(torch_npu_path "$ENV{TORCH_NPU_INSTALL_DIR}")
endif()


if(NOT EXISTS ${torch_npu_path})
  message(WARNING "未找到libtorch_npu安装目录，请设置环境变量TORCH_NPU_INSTALL_DIR")
else()
  message(STATUS "libtorch_npu路径: ${torch_npu_path}")
  include_directories(${torch_npu_path}/include)
  link_directories(${torch_npu_path}/lib)
endif()

# arm环境构建依赖torch.libs/*.so库，需要手动加载
# link_directories(<python path>/site-packages/torch.libs)

# 添加可执行文件
add_executable(example_allreduce_hccl allreduce_hccl.cpp)

# 链接库
target_link_libraries(example_allreduce_hccl 
  -ltorch 
  -ltorch_cpu 
  -lc10 
  -ltorch_npu
)
