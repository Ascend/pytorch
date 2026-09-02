set(DVM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/dvm)
set(DVM_LIB ${DVM_ROOT}/libdvm.a)

set(ACL_INC_ROOT "${TORCHNPU_THIRD_PARTY_ROOT}/acl/inc")
set(DVM_MAKE_ARGS
  PRE_ASCEND=1
  DVM_CUSTOM_FLAGS=-I${ACL_INC_ROOT}
)

file(GLOB DVM_DEPENDS
  CONFIGURE_DEPENDS
  "${DVM_ROOT}/Makefile"
  "${DVM_ROOT}/env.sh"
  "${DVM_ROOT}/src/*.cc"
  "${DVM_ROOT}/src/*.h"
  "${DVM_ROOT}/include/*.h"
)

add_custom_command(
  OUTPUT ${DVM_LIB}
  COMMAND make -C ${DVM_ROOT} ${DVM_MAKE_ARGS} libdvm.a
  WORKING_DIRECTORY ${DVM_ROOT}
  DEPENDS ${DVM_DEPENDS}
  COMMENT "Building DVM static library"
  VERBATIM
)

add_custom_target(dvm_build DEPENDS ${DVM_LIB})
add_library(dvm STATIC IMPORTED GLOBAL)
set_target_properties(dvm PROPERTIES IMPORTED_LOCATION ${DVM_LIB})
add_dependencies(dvm dvm_build)
