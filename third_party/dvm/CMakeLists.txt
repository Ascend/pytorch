set(DVM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/dvm)
set(DVM_LIB ${DVM_ROOT}/libdvm.a)

set(DVM_MAKE_ARGS_STR "PRE_ASCEND=1")

file(GLOB DVM_DEPENDS
  CONFIGURE_DEPENDS
  "${DVM_ROOT}/Makefile"
  "${DVM_ROOT}/env.sh"
  "${DVM_ROOT}/src/*.cc"
  "${DVM_ROOT}/src/*.h"
  "${DVM_ROOT}/include/*.h"
)

set(ACL_INC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../acl/inc)
if (EXISTS "${ACL_INC_ROOT}")
  file(GLOB ACL_INC_DIRS LIST_DIRECTORIES true "${ACL_INC_ROOT}/*")
  foreach(acl_inc_dir IN LISTS ACL_INC_DIRS)
    if (IS_DIRECTORY "${acl_inc_dir}")
      file(COPY "${acl_inc_dir}" DESTINATION "${DVM_ROOT}/include")
    endif()
  endforeach()
endif()

add_custom_command(
  OUTPUT ${DVM_LIB}
  COMMAND make -C ${DVM_ROOT} ${DVM_MAKE_ARGS_STR} libdvm.a
  WORKING_DIRECTORY ${DVM_ROOT}
  DEPENDS ${DVM_DEPENDS}
  COMMENT "Building DVM static library"
  VERBATIM
)

add_custom_target(dvm_build DEPENDS ${DVM_LIB})
add_library(dvm STATIC IMPORTED GLOBAL)
set_target_properties(dvm PROPERTIES IMPORTED_LOCATION ${DVM_LIB})
add_dependencies(dvm dvm_build)
