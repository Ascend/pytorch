FILE(GLOB _INDUCTOR_SRCS
         *.cpp
         aoti_runner/*.cpp
         aoti_torch/*.cpp
         aoti_torch/generated/*.cpp
         aoti_package/*.cpp
         ${PROJECT_SOURCE_DIR}/torch_npu/_inductor/experimental/python_wrapper_fast_launch/csrc/*.cpp
         dvm/*.cpp
         mlir/*.cpp)

LIST(APPEND INDUCTOR_SRCS ${_INDUCTOR_SRCS})

# Pass to parent
set(INDUCTOR_SRCS ${INDUCTOR_SRCS} PARENT_SCOPE)
