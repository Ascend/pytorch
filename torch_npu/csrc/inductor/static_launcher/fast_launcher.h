#pragma once

#ifndef BUILD_LIBTORCH
#include <Python.h>

void RegisterNPUFastLauncherBindings(PyObject* module);
#endif
