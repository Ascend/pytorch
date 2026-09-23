#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

#include "config/device/ascend/aclgraph_conf.h"

namespace nb = nanobind;
using AclGraphConf = fxrt::config::ascend::AclGraphConf;

NB_MODULE(_fxrt_aclgraph_config, m) {
  m.doc() = "Python binding for FXRT AclGraphConf";
  nb::set_leak_warnings(false);
  (void)nb::class_<AclGraphConf>(m, "AclGraphConf")
      .def_static("Instance", &AclGraphConf::Instance, nb::rv_policy::reference)
      .def("set_pool_id", &AclGraphConf::SetPoolId)
      .def("pool_id", &AclGraphConf::GetPoolId)
      .def("set_op_capture_skip", &AclGraphConf::SetOpCaptureSkip)
      .def("begin_capture", &AclGraphConf::BeginCapture)
      .def("end_capture", &AclGraphConf::EndCapture);
}
