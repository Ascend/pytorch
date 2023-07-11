cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

if(NOT DEFINED TORCHAIR_TARGET_PYTHON)
  message(FATAL_ERROR, "torchair target python path is not defined")
endif()

if(NOT DEFINED TORCHAIR_INSTALL_PREFIX)
  message(FATAL_ERROR, "torchair install prefix is not defined")
endif()

set(TORCHAIR_BASE ${CMAKE_CURRENT_LIST_DIR}/torchair)

add_custom_target(copy_torchair_pyfiles ALL
        COMMAND export NO_ASCEND_SDK=1 && export TARGET_PYTHON_PATH=${TORCHAIR_TARGET_PYTHON} && cd ${TORCHAIR_BASE} && bash configure
        COMMAND chmod +x ${CMAKE_CURRENT_LIST_DIR}/install.sh
        COMMAND bash ${CMAKE_CURRENT_LIST_DIR}/install.sh ${TORCHAIR_INSTALL_PREFIX}
        COMMENT "Copy torchair python files to ${TORCHAIR_INSTALL_PREFIX}"
        VERBATIM)
