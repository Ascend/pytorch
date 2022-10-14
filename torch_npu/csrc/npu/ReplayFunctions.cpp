// Copyright (c) 2020 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ReplayFunctions.h"
#include <pybind11/pybind11.h>
#include <structmember.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include "torch/csrc/autograd/utils/wrap_outputs.h"

PyObject *THNPReplayGraphClass = nullptr;
using at::TensorList;
static PyObject* THNPReplayGraph_pynew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    HANDLE_TH_ERRORS
    THPObjectPtr ptr(type->tp_alloc(type, 0));
    if (!ptr) {
        return nullptr;
    }

    THNPReplayGraph* self = (THNPReplayGraph*)ptr.get();
    new (&self->replay_graph) at_npu::native::ReplayGraph();

    return (PyObject*)ptr.release();
    END_HANDLE_TH_ERRORS
}

static void THNPReplayGraph_dealloc(THNPReplayGraph* self) {
    self->replay_graph.~ReplayGraph();
    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject* THNPReplayGraph_generate_replay_graph(THNPReplayGraph* self, PyObject* args) {
    HANDLE_TH_ERRORS
    PyObject* inputs = nullptr;
    PyObject* assigned_outputs = nullptr;
    PyObject* returnable_outputs = nullptr;
    PyObject* retain_inner_outputs = nullptr;
    if (!PyArg_ParseTuple(args, "OOOO", &inputs, &assigned_outputs, &returnable_outputs, &retain_inner_outputs)) {
        THPUtils_invalidArguments(
            args,
            nullptr,
            "generate_replay_graph",
            1,
            "(TensorList inputs, TensorList assigned_outputs, TensorList returnable_outputs, bool retain_inner_outputs);");
        return nullptr;
    }

    static torch::PythonArgParser parser({
        "generate_replay_graph(TensorList inputs, TensorList assigned_outputs, TensorList returnable_outputs, bool retain_inner_outputs)",
        }, true);
    torch::ParsedArgs<4> parsed_args;
    auto _r = parser.parse(args, nullptr, parsed_args);
    pybind11::gil_scoped_release no_gil;
    self->replay_graph.GenerateGraph(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2), _r.toBool(3));
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPReplayGraph_replay(THNPReplayGraph* self, PyObject* args) {
    HANDLE_TH_ERRORS
    PyObject* inputs = nullptr;
    PyObject* assigned_outputs = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &inputs, &assigned_outputs)) {
        THPUtils_invalidArguments(
            args,
            nullptr,
            "replay",
            1,
            "(TensorList inputs, TensorList assigned_outputs);");
        return nullptr;
    }

    static torch::PythonArgParser parser({
        "replay(TensorList inputs, TensorList assigned_outputs)",
        }, true);
    torch::ParsedArgs<2> parsed_args;
    auto _r = parser.parse(args, nullptr, parsed_args);
    auto call_replay = [&](const at::TensorList& inputs, at::TensorList assigned_outputs) -> std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self->replay_graph.Replay(inputs, assigned_outputs);
    };
    return torch::autograd::utils::wrap(call_replay(_r.tensorlist(0), _r.tensorlist(1)));
    END_HANDLE_TH_ERRORS
}

PyObject* THNPReplayGraph_get_inner_outputs(THNPReplayGraph* self, PyObject* args) {
    HANDLE_TH_ERRORS
    PyObject* inputs = nullptr;
    if (!PyArg_ParseTuple(args, "O", &inputs)) {
        THPUtils_invalidArguments(
            args,
            nullptr,
            "get_inner_outputs",
            1,
            "(TensorList inputs);");
        return nullptr;
    }

    static torch::PythonArgParser parser({
        "get_inner_outputs(TensorList inputs)",
        }, true);
    torch::ParsedArgs<1> parsed_args;
    auto _r = parser.parse(args, nullptr, parsed_args);
    auto call_get_inner_outputs = [&](const at::TensorList& inputs) -> std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self->replay_graph.GetInnerOutputs(inputs);
    };
    return torch::autograd::utils::wrap(call_get_inner_outputs(_r.tensorlist(0)));
    END_HANDLE_TH_ERRORS
}

PyObject* THNPReplayGraph_is_replay_cache_hit(THNPReplayGraph* self, PyObject* args) {
    HANDLE_TH_ERRORS
    PyObject* inputs = nullptr;
    if (!PyArg_ParseTuple(args, "O", &inputs)) {
        THPUtils_invalidArguments(
            args,
            nullptr,
            "replay_cache_hit",
            1,
            "(TensorList inputs);");
        return nullptr;
    }

    static torch::PythonArgParser parser({
        "is_replay_cache_hit(TensorList inputs)",
        }, true);
    torch::ParsedArgs<1> parsed_args;
    auto _r = parser.parse(args, nullptr, parsed_args);
    auto call_is_replay_cache_hit = [&](const at::TensorList& inputs) -> bool {
        pybind11::gil_scoped_release no_gil;
        return self->replay_graph.ReplayCacheHit(inputs);
    };
    return torch::autograd::utils::wrap(call_is_replay_cache_hit(_r.tensorlist(0)));
    END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef THNPReplayGraph_properties[] ={
    {nullptr}
};

static PyMethodDef THNPReplayGraph_methods[] = {
    {"generate_replay_graph", (PyCFunction)THNPReplayGraph_generate_replay_graph, METH_VARARGS, nullptr},
    {"replay", (PyCFunction)THNPReplayGraph_replay, METH_VARARGS, nullptr},
    {"get_inner_outputs", (PyCFunction)THNPReplayGraph_get_inner_outputs, METH_VARARGS, nullptr},
    {"is_replay_cache_hit", (PyCFunction)THNPReplayGraph_is_replay_cache_hit, METH_VARARGS, nullptr},
    {nullptr}
};

PyTypeObject THNPReplayGraphType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch_npu._C._NPUReplayGraphBase",    /* tp_name */
    sizeof(THNPReplayGraph),               /* tp_basicsize */
    0,                                     /* tp_itemsize */
    (destructor)THNPReplayGraph_dealloc,   /* tp_dealloc */
    0,                                     /* tp_vectorcall_offset */
    0,                                     /* tp_getattr */
    0,                                     /* tp_setattr */
    0,                                     /* tp_reserved */
    0,                                     /* tp_repr */
    0,                                     /* tp_as_number */
    0,                                     /* tp_as_sequence */
    0,                                     /* tp_as_mapping */
    0,                                     /* tp_hash  */
    0,                                     /* tp_call */
    0,                                     /* tp_str */
    0,                                     /* tp_getattro */
    0,                                     /* tp_setattro */
    0,                                     /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr,                                  /* tp_doc */
    0,                                     /* tp_traverse */
    0,                                     /* tp_clear */
    0,                                     /* tp_richcompare */
    0,                                     /* tp_weaklistoffset */
    0,                                     /* tp_iter */
    0,                                     /* tp_iternext */
    THNPReplayGraph_methods,               /* tp_methods */
    0,                                     /* tp_members */
    THNPReplayGraph_properties,            /* tp_getset */
    0,                                     /* tp_base */
    0,                                     /* tp_dict */
    0,                                     /* tp_descr_get */
    0,                                     /* tp_descr_set */
    0,                                     /* tp_dictoffset */
    0,                                     /* tp_init */
    0,                                     /* tp_alloc */
    THNPReplayGraph_pynew,                 /* tp_new */
};

void THNPReplayGraph_init(PyObject* module) {
    THNPReplayGraphClass = (PyObject*)&THNPReplayGraphType;
    if (PyType_Ready(&THNPReplayGraphType) < 0) {
        throw python_error();
    }
    Py_INCREF(&THNPReplayGraphType);
    if (PyModule_AddObject(module, "_NPUReplayGraphBase", (PyObject*)&THNPReplayGraphType) < 0) {
        throw python_error();
    }
}