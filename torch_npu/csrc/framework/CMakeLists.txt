FILE(GLOB _FRAMEWORK_SRCS
    *.cpp
    aoe/*.cpp
    contiguous/*.cpp
    graph/*.cpp
    graph/cache/*.cpp
    graph/construct/*.cpp
    graph/execute/*.cpp
    graph/scalar/*.cpp
    graph/util/*.cpp
    graph/tools/*.cpp
    interface/*.cpp
    utils/*.cpp)

LIST(APPEND FRAMEWORK_SRCS ${_FRAMEWORK_SRCS})

# Pass to parent
set(FRAMEWORK_SRCS ${FRAMEWORK_SRCS} PARENT_SCOPE)