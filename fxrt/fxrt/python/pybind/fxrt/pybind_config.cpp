#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "config/device/ascend/op_precision_conf.h"

namespace nb = nanobind;
using OpPrecisionConf = fxrt::config::ascend::OpPrecisionConf;

NB_MODULE(_fxrt_config, m) {
  m.doc() = "Python binding for FXRT OpPrecisionConf";
  (void)nb::class_<OpPrecisionConf>(m, "OpPrecisionConf")
      .def_static("Instance", &OpPrecisionConf::Instance, nb::rv_policy::reference)
      .def("set_is_allow_matmul_hf32", &OpPrecisionConf::SetIsAllowMatmulHF32)
      .def("set_acl_precision_mode", &OpPrecisionConf::SetAclPrecisionMode);
}
