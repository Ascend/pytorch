FILE(GLOB _CORE_SRCS *.cpp npu/*.cpp npu/register/*.cpp npu/sys_ctrl/*.cpp npu/impl/*.cpp npu/interface/*.cpp)

LIST(APPEND CORE_SRCS ${_CORE_SRCS})

# Pass to parent
set(CORE_SRCS ${CORE_SRCS} PARENT_SCOPE)