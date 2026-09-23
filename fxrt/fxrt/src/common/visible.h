#ifndef __COMMON_VISIBLE_H__
#define __COMMON_VISIBLE_H__

#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__))
#define DA_API __declspec(dllexport)
#else
#define DA_API __attribute__((visibility("default")))
#endif

#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__))
#ifdef HARDWARE_DLL
#define FXRT_EXPORT __declspec(dllexport)
#else
#define FXRT_EXPORT __declspec(dllimport)
#endif
#define FXRT_LOCAL
#else
#define FXRT_EXPORT __attribute__((visibility("default")))
#define FXRT_LOCAL __attribute__((visibility("hidden")))
#endif
#endif // __COMMON_VISIBLE_H__
