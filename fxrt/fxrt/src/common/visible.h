#ifndef __COMMON_VISIBLE_H__
#define __COMMON_VISIBLE_H__

// FXRT_EXPORT is the one visibility macro; the DA_API that used to sit beside it expanded to
// exactly the same attribute. HARDWARE_DLL marks the translation units that build the shared
// library, so declarations export there and import for whoever consumes the headers. fxrt is
// built only as part of torch_npu, which ships Linux only, so nothing defines HARDWARE_DLL
// today and the Windows arm goes unused; it stays so the headers remain portable.
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
