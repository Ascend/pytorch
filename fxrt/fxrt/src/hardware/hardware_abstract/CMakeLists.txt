
file(GLOB_RECURSE HARDWARE_ABSTRACT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
add_library(hardware_abstract SHARED ${HARDWARE_ABSTRACT_SRC_FILES})
target_link_libraries(hardware_abstract fxrt_common dl)

set_target_properties(hardware_abstract PROPERTIES
  INSTALL_RPATH "${FXRT_LIB_RPATH}"
  BUILD_WITH_INSTALL_RPATH TRUE
)
