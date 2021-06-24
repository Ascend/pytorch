#Build file for C10 NPU
#
#C10 NPU is a minimal library, but it does depend on NPU.

file(GLOB C10_NPU_SYS_CTRL_SRCS sys_ctrl/*.cpp *.cpp impl/*.cpp register/*.cpp interface/*.cpp)

add_library(c10_npu ${C10_NPU_SYS_CTRL_SRCS})

target_link_libraries(c10_npu PUBLIC c10)

if(USE_NPU)
  target_link_libraries(
      c10_npu PRIVATE ${Caffe2_NPU_DEPENDENCY_LIBS})

endif()

install(TARGETS c10_npu EXPORT Caffe2Targets DESTINATION lib)

