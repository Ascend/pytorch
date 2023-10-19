#pragma once
#include <torch_npu/csrc/framework/graph/ReplayGraph.h>
#include <torch/csrc/python_headers.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

struct THNPReplayGraph {
    PyObject_HEAD
    at_npu::native::ReplayGraph replay_graph;
};

extern PyObject* THNPReplayGraphClass;

TORCH_NPU_API void THNPReplayGraph_init(PyObject *module);

inline bool THNPReplayGraph_Check(PyObject* obj) {
    return THNPReplayGraphClass && PyObject_IsInstance(obj, THNPReplayGraphClass);
}