FILE(GLOB _ATEN_SRCS
    *.cpp
    common/*.cpp
    mirror/*.cpp
    ops/*.cpp
    ops/convolution/*.cpp
    ops/loss/*.cpp
    ops/normalization/*.cpp
    ops/pooling/*.cpp
    ops/hcom/*.cpp)

LIST(APPEND ATEN_SRCS ${_ATEN_SRCS})

# Pass to parent
set(ATEN_SRCS ${ATEN_SRCS} PARENT_SCOPE)

if(DEFINED ENV{BSCPP_OPS_ENABLE})
  add_subdirectory(bscpp_ops)
endif()