FILE(GLOB _PROF_SRCS *.cpp python/*.cpp dyno/*.h dyno/*.cpp)

LIST(APPEND PROF_SRCS ${_PROF_SRCS})

# Pass to parent
set(PROF_SRCS ${PROF_SRCS} PARENT_SCOPE)