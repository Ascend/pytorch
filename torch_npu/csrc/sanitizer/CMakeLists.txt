FILE(GLOB _SAN_SRCS *.cpp)

LIST(APPEND SAN_SRCS ${_SAN_SRCS})

# Pass to parent
set(SAN_SRCS ${SAN_SRCS} PARENT_SCOPE)