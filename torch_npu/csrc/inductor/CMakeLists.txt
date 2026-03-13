FILE(GLOB _INDUCTOR_SRCS
         *.cpp
         aoti_runner/*.cpp
         aoti_torch/*.cpp
         mlir/*.cpp)

LIST(APPEND INDUCTOR_SRCS ${_INDUCTOR_SRCS})

# Pass to parent
set(INDUCTOR_SRCS ${INDUCTOR_SRCS} PARENT_SCOPE)