#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

#include "third_party/profiler/combined_traceback.h"

namespace torch {

// symbolize combined traceback objects, converting them into lists of
// dictionaries that are easily consumed in python.

// returns std::vector because one use is to call it with a batch of
// tracebacks that come from a larger datastructure (e.g. a memory snapshot)
// and then have more c++ code to put those objects in the right place.
std::vector<pybind11::object> py_symbolize(std::vector<CapturedTraceback*>& to_symbolize);

// requires GIL to be held, frees any pending free frames
void freeDeadCapturedTracebackFrames();

void installCapturedTracebackPython();

} // namespace torch
