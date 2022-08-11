#include <mutex>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/utils/python_arg_parsing.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/tensor_types.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/python_variable.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include "torch_npu/csrc/utils/LazyInit.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/npu/Stream.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/aten/XLANativeFunctions.h"

namespace torch_npu {
namespace utils {

PyMethodDef* tensor_functions();

}
}