#include <mutex>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/Exceptions.h>
#include "torch_npu/csrc/utils/LazyInit.h"

namespace torch_npu {
namespace utils {


static bool npu_run_yet = false;

void npu_lazy_init()
{
    pybind11::gil_scoped_acquire g;
    // Protected by the GIL.  We don't use call_once because under ASAN it
    // has a buggy implementation that deadlocks if an instance throws an
    // exception.  In any case, call_once isn't necessary, because we
    // have taken a lock.
    if (!npu_run_yet) {
        auto module = THPObjectPtr(PyImport_ImportModule("torch_npu.npu"));
        if (!module) {
            throw python_error();
        }
        auto res = THPObjectPtr(PyObject_CallMethod(module.get(), "_lazy_init", ""));
        if (!res) {
            throw python_error();
        }
        npu_run_yet = true;
    }
}

void npu_set_run_yet_variable_to_false()
{
    npu_run_yet = false;
}

}
}