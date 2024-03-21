#pragma once
// See c10/macros/Export.h for a detailed explanation of what the function
// of these macros are.  We need one set of macros for every separate library
// we build.

#ifdef _WIN32
#if defined(C10_NPU_BUILD_SHARED_LIBS)
#define C10_NPU_EXPORT __declspec(dllexport)
#define C10_NPU_IMPORT __declspec(dllimport)
#else
#define C10_NPU_EXPORT
#define C10_NPU_IMPORT
#endif
#else // _WIN32
#if defined(__GNUC__)
#define C10_NPU_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define C10_NPU_EXPORT
#endif // defined(__GNUC__)
#define C10_NPU_IMPORT C10_NPU_EXPORT
#endif // _WIN32

// This one is being used by libc10_cuda.so
#ifdef C10_NPU_BUILD_MAIN_LIB
#define C10_NPU_API C10_NPU_EXPORT
#else
#define C10_NPU_API C10_NPU_IMPORT
#endif

#define TORCH_NPU_API C10_NPU_API

#define C10_COMPILE_TIME_MAX_NPUS 16
// A maximum of 8 P2P links can be created on a NPU device
#define C10_P2P_ACCESS_MAX_NPUS 8
