/**
 * @file prof_api.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 */

#ifndef PROF_API_H
#define PROF_API_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "prof_common.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
#define MSVP_PROF_API __declspec(dllexport)
#else
#define MSVP_PROF_API __attribute__((visibility("default")))
#endif

/*
 * @ingroup libprofapi
 * @name  profRegReporterCallback
 * @brief register report callback interface for atlas
 * @param [in] reporter: reporter callback handle
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t profRegReporterCallback(MsprofReportHandle reporter);

/*
 * @ingroup libprofapi
 * @name  profRegCtrlCallback
 * @brief register control callback, interface for atlas
 * @param [in] handle: control callback handle
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t profRegCtrlCallback(MsprofCtrlHandle handle);

/*
 * @ingroup libprofapi
 * @name  profRegDeviceStateCallback
 * @brief register device state notify callback, interface for atlas
 * @param [in] handle: handle of ProfNotifySetDevice
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t profRegDeviceStateCallback(MsprofSetDeviceHandle handle);

/*
 * @ingroup libprofapi
 * @name  profGetDeviceIdByGeModelIdx
 * @brief get device id by model id, interface for atlas
 * @param [in] modelIdx: ge model id
 * @param [out] deviceId: device id
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t profGetDeviceIdByGeModelIdx(const uint32_t modelIdx, uint32_t *deviceId);

/*
 * @ingroup libprofapi
 * @name  profSetProfCommand
 * @brief register set profiling command, interface for atlas
 * @param [in] command: 0 isn't aging, !0 is aging
 * @param [in] len: api of timestamp data
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t profSetProfCommand(VOID_PTR command, uint32_t len);

/*
 * @ingroup libprofapi
 * @name  profSetStepInfo
 * @brief set step info for torch, interface for atlas
 * @param [in] indexId: id of iteration index
 * @param [in] tagId: id of tag
 * @param [in] stream: api of timestamp data
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t profSetStepInfo(const uint64_t indexId, const uint16_t tagId, void* const stream);

/*
 * @ingroup libprofapi
 * @name  MsprofRegisterProfileCallback
 * @brief register profile callback by callback type, interface for atlas
 * @param [in] callbackType: type of callback(reporter/ctrl/device state/command)
 * @param [in] callback: callback of profile
 * @param [in] len: callback length
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofRegisterProfileCallback(int32_t callbackType, VOID_PTR callback, uint32_t len);

/**
 * @ingroup libprofapi
 * @name  MsprofInit
 * @brief Profiling module init
 * @param [in] dataType: profiling type: ACL Env/ACL Json/GE Option
 * @param [in] data: profiling switch data
 * @param [in] dataLen: Length of data
 * @return 0:SUCCESS, >0:FAILED
 */
MSVP_PROF_API int32_t MsprofInit(uint32_t dataType, VOID_PTR data, uint32_t dataLen);

/**
 * @ingroup libprofapi
 * @name  MsprofSetConfig
 * @brief Set profiling config
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofSetConfig(uint32_t configType, const char *config, size_t configLength);

/**
 * @ingroup libprofapi
 * @name  MsprofRegisterCallback
 * @brief register profiling switch callback for module
 * @param [in] agingFlag: 0 isn't aging, !0 is aging
 * @param [in] api: api of timestamp data
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofRegisterCallback(uint32_t moduleId, ProfCommandHandle handle);

/**
 * @ingroup libprofapi
 * @name  MsprofReportData
 * @brief report profiling data of module
 * @param [in] moduleId: module id
 * @param [in] type: report type(init/uninit/max length/hash)
 * @param [in] data: profiling data
 * @param [in] len: length of profiling data
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofReportData(uint32_t moduleId, uint32_t type, VOID_PTR data, uint32_t len);

/*
 * @ingroup libprofapi
 * @name  MsprofReportApi
 * @brief report api timestamp
 * @param [in] agingFlag: 0 isn't aging, !0 is aging
 * @param [in] api: api of timestamp data
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofReportApi(uint32_t agingFlag, const struct MsprofApi *api);

/*
 * @ingroup libprofapi
 * @name  MsprofReportEvent
 * @brief report event timestamp
 * @param [in] agingFlag: 0 isn't aging, !0 is aging
 * @param [in] event: event of timestamp data
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofReportEvent(uint32_t agingFlag, const struct MsprofEvent *event);

/*
 * @ingroup libprofapi
 * @name  MsprofReportCompactInfo
 * @brief report profiling compact infomation
 * @param [in] agingFlag: 0 isn't aging, !0 is aging
 * @param [in] data: profiling data of compact infomation
 * @param [in] length: length of profiling data
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofReportCompactInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length);

/*
 * @ingroup libprofapi
 * @name  MsprofReportAdditionalInfo
 * @brief report profiling additional infomation
 * @param [in] agingFlag: 0 isn't aging, !0 is aging
 * @param [in] data: profiling data of additional infomation
 * @param [in] length: length of profiling data
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofReportAdditionalInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length);

/*
 * @ingroup libprofapi
 * @name  MsprofRegTypeInfo
 * @brief reg mapping info of type id and type name
 * @param [in] level: level is the report struct's level
 * @param [in] typeId: type id is the report struct's type
 * @param [in] typeName: label of type id for presenting user
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofRegTypeInfo(uint16_t level, uint32_t typeId, const char *typeName);

/*
 * @ingroup libprofapi
 * @name  MsprofGetHashId
 * @brief return hash id of hash info
 * @param [in] hashInfo: infomation to be hashed
 * @param [in] length: the length of infomation to be hashed
 * @return hash id
 */
MSVP_PROF_API uint64_t MsprofGetHashId(const char *hashInfo, size_t length);

/**
 * @ingroup libprofapi
 * @name  MsprofSetDeviceIdByGeModelIdx
 * @brief insert device id by model id
 * @param [in] geModelIdx: ge model id
 * @param [in] deviceId: device id
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofSetDeviceIdByGeModelIdx(const uint32_t geModelIdx, const uint32_t deviceId);

/**
 * @ingroup libprofapi
 * @name  MsprofUnsetDeviceIdByGeModelIdx
 * @brief delete device id by model id
 * @param [in] geModelIdx: ge model id
 * @param [in] deviceId: device id
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofUnsetDeviceIdByGeModelIdx(const uint32_t geModelIdx, const uint32_t deviceId);

/**
 * @ingroup libprofapi
 * @name  register report interface for atlas
 * @brief report api timestamp
 * @param [in] chipId: multi die's chip
 * @param [in] deviceId: device id
 * @param [in] isOpen: device is open
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofNotifySetDevice(uint32_t chipId, uint32_t deviceId, bool isOpen);

/**
 * @ingroup libprofapi
 * @name  MsprofFinalize
 * @brief profiling finalize
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofFinalize();

/*
 * @ingroup libprofapi
 * @name  MsprofSysCycleTime
 * @brief get systime cycle time of CPU
 * @return system cycle time of CPU
 */
MSVP_PROF_API uint64_t MsprofSysCycleTime();

#ifdef __cplusplus
}
#endif

#endif
