if (DEFINED BUILD_LIBTORCH)
  FILE(GLOB _DIST_SRCS *.cpp)
  # Exclude Python binding files when building libtorch
  list(REMOVE_ITEM _DIST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/Init.cpp")
else()
  FILE(GLOB _DIST_SRCS *.cpp rpc/*.cpp symm_mem/*.cpp)
endif()

LIST(APPEND DIST_SRCS ${_DIST_SRCS})

# Pass to parent
set(DIST_SRCS ${DIST_SRCS} PARENT_SCOPE)