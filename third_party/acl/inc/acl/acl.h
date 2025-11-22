/**
* @file acl.h
*
* Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef INC_EXTERNAL_ACL_ACL_H_
#define INC_EXTERNAL_ACL_ACL_H_

#include "acl_rt.h"
#include "acl_op.h"
#include "acl_mdl.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACL_PKG_VERSION_MAX_SIZE       128
#define ACL_PKG_VERSION_PARTS_MAX_SIZE 64
#define ACL_IPC_HANDLE_SIZE            65

/**
 * @ingroup AscendCL
 * @brief acl initialize
 *
 * @par Restriction
 * This interface can be called multiple times in a process, the reference count will
 * increase by 1 each time when aclInit is called multiple times.
 * The aclFinalizeReference interface should be called the same number of times as aclInit,
 * or you can call aclFinalize to release all resources regardless of the reference count and reset it to 0.
 *
 * @param configPath [IN]    the config path,it can be NULL
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclInit(const char *configPath);

/**
 * @ingroup AscendCL
 * @brief acl finalize
 *
 * @par Restriction
 * This interface should be called before the process exits.
 * After calling aclFinalize, all allocated resources are released,
 * the internal reference count is reset to 0, and ACL services can no longer be used.
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclFinalize();

/**
 * @ingroup AscendCL
 * @brief query ACL interface version
 *
 * @param majorVersion[OUT] ACL interface major version
 * @param minorVersion[OUT] ACL interface minor version
 * @param patchVersion[OUT] ACL interface patch version
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetVersion(int32_t *majorVersion, int32_t *minorVersion, int32_t *patchVersion);

/**
 * @ingroup AscendCL
 * @brief acl finalize reference
 *
 * @par Restriction
 * This interface decrements the internal reference count each time it is called.
 * Resources are only released when the reference count reaches 0.
 * To get the current reference count, pass a valid pointer to refCount.
 * To ignore the reference count, pass nullptr instead.
 *
 * @param refCount [IN/OUT] Pointer to receive current reference count after calling aclFinalizeReference; can be nullptr.
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclFinalizeReference(uint64_t *refCount);

/**
 * @ingroup AscendCL
 * @brief get recent error message
 *
 * @retval null for failed
 * @retval OtherValues success
*/
ACL_FUNC_VISIBILITY const char *aclGetRecentErrMsg();

/**
 * @ingroup AscendCL
 * @brief enum for CANN package name
 */
typedef enum aclCANNPackageName {
    ACL_PKG_NAME_CANN,
    ACL_PKG_NAME_RUNTIME,
    ACL_PKG_NAME_COMPILER,
    ACL_PKG_NAME_HCCL,
    ACL_PKG_NAME_TOOLKIT,
    ACL_PKG_NAME_OPP,
    ACL_PKG_NAME_OPP_KERNEL,
    ACL_PKG_NAME_DRIVER
} aclCANNPackageName;

/**
 * @ingroup AscendCL
 * @brief struct for storaging CANN package version
 */
typedef struct aclCANNPackageVersion {
    char version[ACL_PKG_VERSION_MAX_SIZE];
    char majorVersion[ACL_PKG_VERSION_PARTS_MAX_SIZE];
    char minorVersion[ACL_PKG_VERSION_PARTS_MAX_SIZE];
    char releaseVersion[ACL_PKG_VERSION_PARTS_MAX_SIZE];
    char patchVersion[ACL_PKG_VERSION_PARTS_MAX_SIZE];
    char reserved[ACL_PKG_VERSION_MAX_SIZE];
} aclCANNPackageVersion;

/**
 * @ingroup AscendCL
 * @brief query CANN package version
 *
 * @param name[IN] CANN package name
 * @param version[OUT] CANN package version information
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval ACL_ERROR_INVALID_FILE Failure
 */
ACL_FUNC_VISIBILITY aclError aclsysGetCANNVersion(aclCANNPackageName name, aclCANNPackageVersion *version);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_H_
