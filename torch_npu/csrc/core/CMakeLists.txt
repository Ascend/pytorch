FILE(GLOB _CORE_SRCS *.cpp npu/*.cpp npu/*/*.cpp)

LIST(APPEND CORE_SRCS ${_CORE_SRCS})

# Pass to parent
set(CORE_SRCS ${CORE_SRCS} PARENT_SCOPE)