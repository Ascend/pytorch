// Header file providing new C API functions to old Python versions.
//
// File distributed under the Zero Clause BSD (0BSD) license.
// Copyright Contributors to the pythoncapi_compat project.
//
// SPDX-License-Identifier: 0BSD

#ifndef PYTHONCAPI_COMPAT
#define PYTHONCAPI_COMPAT

#ifdef __cplusplus
extern "C" {
#endif

#include "frameobject.h" // PyFrameObject, PyFrame_GetBack()
#include <Python.h>

// Compatibility with Visual Studio 2013 and older which don't support
// the inline keyword in C (only in C++): use __inline instead.
#if (defined(_MSC_VER) && _MSC_VER < 1900 && !defined(__cplusplus) && !defined(inline))
#define PYCAPI_COMPAT_STATIC_INLINE(TYPE) static __inline TYPE
#else
#define PYCAPI_COMPAT_STATIC_INLINE(TYPE) static inline TYPE
#endif

#ifndef _Py_CAST
#define _Py_CAST(type, expr) ((type)(expr))
#endif

// On C++11 and newer, _Py_NULL is defined as nullptr on C++11,
// otherwise it is defined as NULL.
#ifndef _Py_NULL
#if defined(__cplusplus) && __cplusplus >= 201103
#define _Py_NULL nullptr
#else
#define _Py_NULL NULL
#endif
#endif

// Cast argument to PyObject* type.
#ifndef _PyObject_CAST
#define _PyObject_CAST(op) _Py_CAST(PyObject*, op)
#endif

// bpo-42262 added Py_NewRef() to Python 3.10.0a3
#if PY_VERSION_HEX < 0x030A00A3 && !defined(Py_NewRef)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
_Py_NewRef(PyObject* obj)
{
    Py_INCREF(obj);
    return obj;
}
#define Py_NewRef(obj) _Py_NewRef(_PyObject_CAST(obj))
#endif

// bpo-42262 added Py_XNewRef() to Python 3.10.0a3
#if PY_VERSION_HEX < 0x030A00A3 && !defined(Py_XNewRef)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
_Py_XNewRef(PyObject* obj)
{
    Py_XINCREF(obj);
    return obj;
}
#define Py_XNewRef(obj) _Py_XNewRef(_PyObject_CAST(obj))
#endif

// bpo-40421 added PyFrame_GetCode() to Python 3.9.0b1
#if PY_VERSION_HEX < 0x030900B1 || defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyCodeObject*)
PyFrame_GetCode(PyFrameObject* frame)
{
    assert(frame != _Py_NULL);
    assert(frame->f_code != _Py_NULL);
    return _Py_CAST(PyCodeObject*, Py_NewRef(frame->f_code));
}
#endif

// bpo-40421 added PyFrame_GetLasti() to Python 3.11.0b1
#if PY_VERSION_HEX < 0x030B00B1 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(int)
PyFrame_GetLasti(PyFrameObject* frame)
{
#if PY_VERSION_HEX >= 0x030A00A7
    // bpo-27129: Since Python 3.10.0a7, f_lasti is an instruction offset,
    // not a bytes offset anymore. Python uses 16-bit "wordcode" (2 bytes)
    // instructions.
    if (frame->f_lasti < 0) {
        return -1;
    }
    return frame->f_lasti * 2;
#else
    return frame->f_lasti;
#endif
}
#endif

// bpo-40421 added PyFrame_GetBack() to Python 3.9.0b1
#if PY_VERSION_HEX < 0x030900B1 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyFrameObject*)
PyFrame_GetBack(PyFrameObject* frame)
{
    assert(frame != _Py_NULL);
    return _Py_CAST(PyFrameObject*, Py_XNewRef(frame->f_back));
}
#endif
#ifdef __cplusplus
}
#endif
#endif // PYTHONCAPI_COMPAT
