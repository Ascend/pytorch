FILE(GLOB _CUS_DTYPE_SRCS *.cpp)

LIST(APPEND CUS_DTYPE_SRCS ${_CUS_DTYPE_SRCS})

# Pass to parent
set(CUS_DTYPE_SRCS ${CUS_DTYPE_SRCS} PARENT_SCOPE)
