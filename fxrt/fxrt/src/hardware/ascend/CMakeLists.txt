
if(ENABLE_ASCEND)
  message("Note compile ascend path: ${FXRT_ASCEND_INCLUDE_DIRS}")
  include_directories(${FXRT_ASCEND_INCLUDE_DIRS})
  link_directories(${FXRT_ASCEND_LIB_DIRS})

  file(GLOB_RECURSE HARDWARE_ASCEND_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
  add_library(hardware_ascend SHARED ${HARDWARE_ASCEND_SRC_FILES})
  target_link_libraries(hardware_ascend
      hardware_abstract fxrt_common fxrt_config
      ${FXRT_ASCENDCL_LIB}
      ${FXRT_RUNTIME_LIB}
      ${FXRT_HCCL_LIB}
      dl
    )

  set_target_properties(hardware_ascend PROPERTIES
    INSTALL_RPATH "${FXRT_LIB_RPATH}"
    BUILD_WITH_INSTALL_RPATH TRUE
  )

endif()
