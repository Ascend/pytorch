#ifndef HI_DVPP_TYPE_H_
#define HI_DVPP_TYPE_H_

#include <stdint.h>

typedef unsigned char hi_u8;
typedef signed char hi_s8;
typedef unsigned short hi_u16;
typedef short hi_s16;
typedef unsigned int hi_u32;
typedef int hi_s32;
typedef unsigned long long hi_u64;
typedef long long hi_s64;
typedef char hi_char;
typedef double hi_double;
typedef hi_u32 hi_fr32;
typedef float hi_float;

#define hi_void void
#define HI_NULL 0L
#define HI_SUCCESS 0
#define HI_FAILURE (-1)

typedef enum {
    HI_FALSE = 0,
    HI_TRUE = 1,
} hi_bool;

#endif // #ifndef HI_DVPP_TYPE_H_