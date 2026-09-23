# Set the source files for profiler
set(PROFILER_SOURCES
  profiler.cpp
)

# Create object library
add_library(fxrt_profiler SHARED ${PROFILER_SOURCES})

# Include directories
target_include_directories(fxrt_profiler PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${CMAKE_CURRENT_SOURCE_DIR}/../ir
  ${CMAKE_CURRENT_SOURCE_DIR}/../runtime
)

# Link with dependencies
target_link_libraries(fxrt_profiler PRIVATE fxrt_common fxrt::json)
set_target_properties(fxrt_profiler PROPERTIES
  INSTALL_RPATH "${FXRT_LIB_RPATH}"
  BUILD_WITH_INSTALL_RPATH TRUE
)
