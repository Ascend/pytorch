
if(ENABLE_CPU)
  file(GLOB_RECURSE HARDWARE_CPU_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
  add_library(hardware_cpu SHARED ${HARDWARE_CPU_SRC_FILES})
  target_link_libraries(hardware_cpu hardware_abstract fxrt_common)

  set_target_properties(hardware_cpu PROPERTIES
    INSTALL_RPATH "${FXRT_LIB_RPATH}"
    BUILD_WITH_INSTALL_RPATH TRUE
  )

endif()
