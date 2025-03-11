/**
* @file acl_tdt.h
*
* Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef INC_EXTERNAL_ACL_ACL_TDT_H_
#define INC_EXTERNAL_ACL_ACL_TDT_H_

#include "acl/acl_base.h"

#ifdef __cplusplus
extern "C" {
#endif

enum acltdtTensorType {
    ACL_TENSOR_DATA_UNDEFINED = -1,
    ACL_TENSOR_DATA_TENSOR,
    ACL_TENSOR_DATA_END_OF_SEQUENCE,
    ACL_TENSOR_DATA_ABNORMAL,
    ACL_TENSOR_DATA_SLICE_TENSOR,
    ACL_TENSOR_DATA_END_TENSOR
};

typedef struct acltdtDataItem acltdtDataItem;
typedef struct acltdtDataset acltdtDataset;
typedef struct acltdtChannelHandle acltdtChannelHandle;

/**
 * @ingroup AscendCL
 * @brief Get tensor type from item
 *
 * @param dataItem [IN] pointer to the data item
 *
 * @retval Tensor type.
 * @retval ACL_DT_UNDEFINED if dataItem is null
 */
ACL_FUNC_VISIBILITY acltdtTensorType acltdtGetTensorTypeFromItem(const acltdtDataItem *dataItem);

/**
 * @ingroup AscendCL
 * @brief Get data type from item
 *
 * @param dataItem [IN] pointer to the data item
 *
 * @retval Data type.
 * @retval ACL_DT_UNDEFINED if dataItem is null
 */
ACL_FUNC_VISIBILITY aclDataType acltdtGetDataTypeFromItem(const acltdtDataItem *dataItem);

/**
 * @ingroup AscendCL
 * @brief Get data address from item
 *
 * @param dataItem [IN] pointer to data item
 *
 * @retval null for failed
 * @retval OtherValues success
*/
ACL_FUNC_VISIBILITY void *acltdtGetDataAddrFromItem(const acltdtDataItem *dataItem);

/**
 * @ingroup AscendCL
 * @brief Get data size from item
 *
 * @param dataItem [IN] pointer to data item
 *
 * @retval 0 for failed
 * @retval OtherValues success
*/
ACL_FUNC_VISIBILITY size_t acltdtGetDataSizeFromItem(const acltdtDataItem *dataItem);

/**
 * @ingroup AscendCL
 * @brief Get dim's number from item
 *
 * @param dataItem [IN] pointer to data item
 *
 * @retval 0 for failed
 * @retval OtherValues success
*/
ACL_FUNC_VISIBILITY size_t acltdtGetDimNumFromItem(const acltdtDataItem *dataItem);

/**
 * @ingroup AscendCL
 * @brief Get slice info from item
 *
 * @param dataItem [IN] pointer to data item
 * @param sliceNum [OUT] pointer to the sliceNum of dataItem
 * @param sliceId [OUT] pointer to the sliceId of dataItem
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError acltdtGetSliceInfoFromItem(const acltdtDataItem *dataItem, size_t *sliceNum,
    size_t* sliceId);

/**
 * @ingroup AscendCL
 * @brief Get dims from item
 *
 * @param  dataItem [IN]      the struct of data item
 * @param  dims [IN|OUT]      pointer to the dims of dataItem
 * @param  dimNum [IN]        the size of the dims
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError acltdtGetDimsFromItem(const acltdtDataItem *dataItem, int64_t *dims, size_t dimNum);

/**
 * @ingroup AscendCL
 * @brief Create the struct of data item
 *
 * @param tdtType [IN]  Tdt tensor type
 * @param dims [IN]     pointer of tdtDataItem's dims
 * @param dimNum [IN]   Dim number
 * @param dataType [IN] Data type
 * @param data [IN]     Data pointer
 * @param size [IN]     Data size
 *
 * @retval null for failed
 * @retval OtherValues success
 *
 * @see acltdtDestroyDataItem
 */
ACL_FUNC_VISIBILITY acltdtDataItem *acltdtCreateDataItem(acltdtTensorType tdtType,
                                                         const int64_t *dims,
                                                         size_t dimNum,
                                                         aclDataType dataType,
                                                         void *data,
                                                         size_t size);

/**
 * @ingroup AscendCL
 * @brief Destroy the struct of data item
 *
 * @param dataItem [IN]  pointer to the data item
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtCreateDataItem
 */
ACL_FUNC_VISIBILITY aclError acltdtDestroyDataItem(acltdtDataItem *dataItem);

/**
 * @ingroup AscendCL
 * @brief Create the tdt dataset
 *
 * @retval null for failed
 * @retval OtherValues success
 *
 * @see acltdtDestroyDataset
 */
ACL_FUNC_VISIBILITY acltdtDataset *acltdtCreateDataset();

/**
 * @ingroup AscendCL
 * @brief Destroy the tdt dataset
 *
 * @param dataset [IN]  pointer to the dataset
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtCreateDataset
 */
ACL_FUNC_VISIBILITY aclError acltdtDestroyDataset(acltdtDataset *dataset);

/**
 * @ingroup AscendCL
 * @brief Get the data item
 *
 * @param dataset [IN] pointer to the dataset
 * @param index [IN]   index of the dataset
 *
 * @retval null for failed
 * @retval OtherValues success
 *
 * @see acltdtAddDataItem
 */
ACL_FUNC_VISIBILITY acltdtDataItem *acltdtGetDataItem(const acltdtDataset *dataset, size_t index);

/**
 * @ingroup AscendCL
 * @brief Get the data item
 *
 * @param dataset [OUT] pointer to the dataset
 * @param dataItem [IN] pointer to the data item
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtGetDataItem
 */
ACL_FUNC_VISIBILITY aclError acltdtAddDataItem(acltdtDataset *dataset, acltdtDataItem *dataItem);

/**
 * @ingroup AscendCL
 * @brief Get the size of dataset
 *
 * @param dataset [IN]  pointer to the dataset
 *
 * @retval 0 for failed
 * @retval OtherValues success
 */
ACL_FUNC_VISIBILITY size_t acltdtGetDatasetSize(const acltdtDataset *dataset);

/**
 * @ingroup AscendCL
 * @brief Get the name of dataset
 *
 * @param  dataset [IN]      pointer to the dataset
 *
 * @retval null for failed
 * @retval OtherValues success
 */
ACL_FUNC_VISIBILITY const char *acltdtGetDatasetName(const acltdtDataset *dataset);

/**
 * @ingroup AscendCL
 * @brief Stop the channel
 *
 * @param handle [IN]  pointer to the channel handle
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtCreateChannel | acltdtDestroyChannel
 */
ACL_FUNC_VISIBILITY aclError acltdtStopChannel(acltdtChannelHandle *handle);

/**
 * @ingroup AscendCL
 * @brief Create the channel
 *
 * @param deviceId [IN]  the device id
 * @param name [IN]      the name of channel
 *
 * @retval null for failed
 * @retval OtherValues success
 *
 * @see acltdtStopChannel | acltdtDestroyChannel
 */
ACL_FUNC_VISIBILITY acltdtChannelHandle *acltdtCreateChannel(uint32_t deviceId, const char *name);

/**
 * @ingroup AscendCL
 * @brief Create the channel with max size
 *
 * @param deviceId [IN]  the device id
 * @param name [IN]      the name of channel
 * @param capacity [IN]   the capacity of channel
 *
 * @retval null for failed
 * @retval OtherValues success
 *
 * @see acltdtDestroyChannel
 */
ACL_FUNC_VISIBILITY acltdtChannelHandle *acltdtCreateChannelWithCapacity(uint32_t deviceId,
                                                                         const char *name,
                                                                         size_t capacity);

/**
 * @ingroup AscendCL
 * @brief Destroy the channel
 *
 * @param handle [IN]  pointer to the channel handle
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtCreateChannel | acltdtStopChannel
 */
ACL_FUNC_VISIBILITY aclError acltdtDestroyChannel(acltdtChannelHandle *handle);

/**
 * @ingroup AscendCL
 * @brief clean the channel
 *
 * @param handle [IN]      pointer to the channel handle
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 */
ACL_FUNC_VISIBILITY aclError acltdtCleanChannel(acltdtChannelHandle *handle);

/**
 * @ingroup AscendCL
 * @brief Send tensor to device
 *
 * @param handle [IN]  pointer to the channel handle
 * @param dataset [IN] pointer to the dataset
 * @param timeout [IN] timeout/ms
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtReceiveTensor
 */
ACL_FUNC_VISIBILITY aclError acltdtSendTensor(const acltdtChannelHandle *handle,
                                              const acltdtDataset *dataset,
                                              int32_t timeout);

/**
 * @ingroup AscendCL
 * @brief Receive tensor from device
 *
 * @param handle [IN]      pointer to the channel handle
 * @param dataset [OUT]    pointer to the dataset
 * @param timeout [IN]     timeout/ms
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see acltdtSendTensor
 */
ACL_FUNC_VISIBILITY aclError acltdtReceiveTensor(const acltdtChannelHandle *handle,
                                                 acltdtDataset *dataset,
                                                 int32_t timeout);

/**
 * @ingroup AscendCL
 * @brief query the size of the channel
 *
 * @param handle [IN]      pointer to the channel handle
 * @param size [OUT]       current size of this channel
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 */
ACL_FUNC_VISIBILITY aclError acltdtQueryChannelSize(const acltdtChannelHandle *handle, size_t *size);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_TDT_H_
