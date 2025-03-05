/**
* @file acl_rt_allocator.h
*
* Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef INC_EXTERNAL_ACL_ACL_RT_ALLOCATOR_H_
#define INC_EXTERNAL_ACL_ACL_RT_ALLOCATOR_H_

#include "acl_base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *aclrtAllocatorDesc;
typedef void *aclrtAllocator;
typedef void *aclrtAllocatorBlock;
typedef void *aclrtAllocatorAddr;

typedef void *(*aclrtAllocatorAllocFunc)(aclrtAllocator allocator, size_t size);
typedef void (*aclrtAllocatorFreeFunc)(aclrtAllocator allocator, aclrtAllocatorBlock block);
typedef void *(*aclrtAllocatorAllocAdviseFunc)(aclrtAllocator allocator, size_t size, aclrtAllocatorAddr addr);
typedef void *(*aclrtAllocatorGetAddrFromBlockFunc)(aclrtAllocatorBlock block);

/**
 * @ingroup AscendCL
 * @brief Create allocator description
 *
 * @retval null for failed
 * @retval OtherValues success
 *
 * @see aclrtAllocatorDestroyDesc
 */
ACL_FUNC_VISIBILITY aclrtAllocatorDesc aclrtAllocatorCreateDesc();

/**
 * @ingroup AscendCL
 * @brief Relese allocator description
 *
 * @param allocatorDesc [IN]     allocator description
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtAllocatorCreateDesc
 */
ACL_FUNC_VISIBILITY aclError aclrtAllocatorDestroyDesc(aclrtAllocatorDesc allocatorDesc);

/**
 * @ingroup AscendCL
 * @brief Register allocator object to allocator description
 *
 * @param allocatorDesc [IN] allocator description
 * @param allocator [IN]    allocator object handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtAllocatorSetObjToDesc(aclrtAllocatorDesc allocatorDesc, aclrtAllocator allocator);

/**
 * @ingroup AscendCL
 * @brief Register the function pointer of alloc memory to the allocator description
 *
 * @param allocatorDesc [IN] allocator description
 * @param func [IN]    the function pointer of alloc memory
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtAllocatorSetAllocFuncToDesc(aclrtAllocatorDesc allocatorDesc,
                                                              aclrtAllocatorAllocFunc func);

/**
 * @ingroup AscendCL
 * @brief Register the function pointer of free memory to the allocator description
 *
 * @param allocatorDesc [IN] allocator description
 * @param func [IN]    free memory function pointer
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtAllocatorSetFreeFuncToDesc(aclrtAllocatorDesc allocatorDesc,
                                                             aclrtAllocatorFreeFunc func);

/**
 * @ingroup AscendCL
 * @brief Register the function pointer of alloc suggested memory to the allocator description
 *
 * @param allocatorDesc [IN] allocator description
 * @param func [IN]    the function pointer of alloc suggested memory
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtAllocatorSetAllocAdviseFuncToDesc(aclrtAllocatorDesc allocatorDesc,
                                                                    aclrtAllocatorAllocAdviseFunc func);

/**
 * @ingroup AscendCL
 * @brief Register the function pointer of get address from block to the allocator description
 *
 * @param allocatorDesc [IN] allocator description
 * @param func [IN]    the function pointer of get address from block
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtAllocatorSetGetAddrFromBlockFuncToDesc(aclrtAllocatorDesc allocatorDesc,
                                                                         aclrtAllocatorGetAddrFromBlockFunc func);

/**
 * @ingroup AscendCL
 * @brief Register allocator description to acl by stream
 *
 * @param stream [IN]    stream handle
 * @param allocatorDesc [IN] allocator description
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtAllocatorUnregister
 */
ACL_FUNC_VISIBILITY aclError aclrtAllocatorRegister(aclrtStream stream, aclrtAllocatorDesc allocatorDesc);

/**
 * @ingroup AscendCL
 * @brief Unregister allocator description from acl by stream
 *
 * @param stream [IN]    stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtAllocatorRegister
 */
ACL_FUNC_VISIBILITY aclError aclrtAllocatorUnregister(aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_RT_ALLOCATOR_H_
