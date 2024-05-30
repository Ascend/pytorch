FILE(GLOB _PROF_SRCS *.cpp unwind/*.cpp python/*.cpp)

LIST(APPEND PROF_SRCS ${_PROF_SRCS})

# Pass to parent
set(PROF_SRCS ${PROF_SRCS} PARENT_SCOPE)