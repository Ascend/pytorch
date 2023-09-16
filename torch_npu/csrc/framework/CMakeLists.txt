FILE(GLOB _FRAMEWORK_SRCS
    *.cpp
    aoe/*.cpp
    autograd/*.cpp
    contiguous/*.cpp
    interface/*.cpp
    utils/*.cpp)

LIST(APPEND FRAMEWORK_SRCS ${_FRAMEWORK_SRCS})

# Pass to parent
set(FRAMEWORK_SRCS ${FRAMEWORK_SRCS} PARENT_SCOPE)