#ifndef HI_DVPP_VB_H_
#define HI_DVPP_VB_H_

#include "hi_dvpp_common.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

typedef hi_u32 hi_vb_pool;

typedef enum  {
    HI_VB_SRC_COMMON  = 0,
    HI_VB_SRC_MOD  = 1,
    HI_VB_SRC_PRIVATE = 2,
    HI_VB_SRC_USER    = 3,
    HI_VB_SRC_BUTT
} hi_vb_src;

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif // #ifndef HI_DVPP_VB_H_