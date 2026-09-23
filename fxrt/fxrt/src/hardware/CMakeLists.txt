

add_subdirectory(hardware_abstract)
if(ENABLE_ASCEND)
  add_subdirectory(ascend)
  add_dependencies(hardware_ascend hardware_abstract fxrt_config)
endif()

if(ENABLE_CPU)
  add_subdirectory(cpu)
  add_dependencies(hardware_cpu hardware_abstract)
endif()