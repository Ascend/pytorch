/**
* @file rt_error_codes.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2019-2020. All Rights Reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef __INC_EXTERNEL_RT_ERROR_CODES_H__
#define __INC_EXTERNEL_RT_ERROR_CODES_H__

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define  ACL_RT_SUCCESS    0                               // success
#define  ACL_ERROR_RT_PARAM_INVALID              107000 // param invalid
#define  ACL_ERROR_RT_INVALID_DEVICEID           107001 // invalid device id
#define  ACL_ERROR_RT_CONTEXT_NULL               107002 // current context null
#define  ACL_ERROR_RT_STREAM_CONTEXT             107003 // stream not in current context
#define  ACL_ERROR_RT_MODEL_CONTEXT              107004 // model not in current context
#define  ACL_ERROR_RT_STREAM_MODEL               107005 // stream not in model
#define  ACL_ERROR_RT_EVENT_TIMESTAMP_INVALID    107006 // event timestamp invalid
#define  ACL_ERROR_RT_EVENT_TIMESTAMP_REVERSAL   107007 // event timestamp reversal
#define  ACL_ERROR_RT_ADDR_UNALIGNED             107008 // memory address unaligned
#define  ACL_ERROR_RT_FILE_OPEN                  107009 // open file failed
#define  ACL_ERROR_RT_FILE_WRITE                 107010 // write file failed
#define  ACL_ERROR_RT_STREAM_SUBSCRIBE           107011 // error subscribe stream
#define  ACL_ERROR_RT_THREAD_SUBSCRIBE           107012 // error subscribe thread
#define  ACL_ERROR_RT_GROUP_NOT_SET              107013 // group not set
#define  ACL_ERROR_RT_GROUP_NOT_CREATE           107014 // group not create
#define  ACL_ERROR_RT_STREAM_NO_CB_REG           107015 // callback not register to stream
#define  ACL_ERROR_RT_INVALID_MEMORY_TYPE        107016 // invalid memory type
#define  ACL_ERROR_RT_INVALID_HANDLE             107017 // invalid handle
#define  ACL_ERROR_RT_INVALID_MALLOC_TYPE        107018 // invalid malloc type
#define  ACL_ERROR_RT_WAIT_TIMEOUT               107019 // wait timeout
#define  ACL_ERROR_RT_TASK_TIMEOUT               107020 // task timeout
#define  ACL_ERROR_RT_SYSPARAMOPT_NOT_SET        107021 // not set sysparamopt
#define  ACL_ERROR_RT_DEVICE_TASK_ABORT          107022 // device task aborting
#define  ACL_ERROR_RT_STREAM_ABORT               107023 // stream aborting
#define  ACL_ERROR_RT_CAPTURE_DEPENDENCY         107024 // capture dependency failure
#define  ACL_ERROR_RT_STREAM_UNJOINED            107025 // invalid capture model
#define  ACL_ERROR_RT_MODEL_CAPTURED             107026 // model is captured
#define  ACL_ERROR_RT_STREAM_CAPTURED            107027 // stream is captured
#define  ACL_ERROR_RT_EVENT_CAPTURED             107028 // event is captured
#define  ACL_ERROR_RT_STREAM_NOT_CAPTURED        107029 // stream is not in capture status
#define  ACL_ERROR_RT_CAPTURE_MODE_NOT_SUPPORT   107030 // stream is captured, not support current oper
#define  ACL_ERROR_RT_STREAM_CAPTURE_IMPLICIT    107031 // a disallowed implicit dependency from defalut stream
#define  ACL_ERROR_STREAM_CAPTURE_CONFLICT       107032 // interdependent stream cannot begin capture together
#define  ACL_ERROR_STREAM_TASK_GROUP_STATUS      107033 // task group status error
#define  ACL_ERROR_STREAM_TASK_GROUP_INTR        107034 // task group interrupted
#define  ACL_ERROR_RT_TASK_ABORT_STOP            107035 // device task aborting stop before post process
#define  ACL_ERROR_RT_STREAM_CAPTURE_UNMATCHED   107036 // the capture was not initiated in this stream

#define  ACL_ERROR_RT_FEATURE_NOT_SUPPORT        207000 // feature not support
#define  ACL_ERROR_RT_MEMORY_ALLOCATION          207001 // memory allocation error
#define  ACL_ERROR_RT_MEMORY_FREE                207002 // memory free error
#define  ACL_ERROR_RT_AICORE_OVER_FLOW           207003 // aicore over flow
#define  ACL_ERROR_RT_NO_DEVICE                  207004 // no device
#define  ACL_ERROR_RT_RESOURCE_ALLOC_FAIL        207005 // resource alloc fail
#define  ACL_ERROR_RT_NO_PERMISSION              207006 // no permission
#define  ACL_ERROR_RT_NO_EVENT_RESOURCE          207007 // no event resource
#define  ACL_ERROR_RT_NO_STREAM_RESOURCE         207008 // no stream resource
#define  ACL_ERROR_RT_NO_NOTIFY_RESOURCE         207009 // no notify resource
#define  ACL_ERROR_RT_NO_MODEL_RESOURCE          207010 // no model resource
#define  ACL_ERROR_RT_NO_CDQ_RESOURCE            207011 // no cdq resource
#define  ACL_ERROR_RT_OVER_LIMIT                 207012 // over limit
#define  ACL_ERROR_RT_QUEUE_EMPTY                207013 // queue is empty
#define  ACL_ERROR_RT_QUEUE_FULL                 207014 // queue is full
#define  ACL_ERROR_RT_REPEATED_INIT              207015 // repeated init
#define  ACL_ERROR_RT_AIVEC_OVER_FLOW            207016 // aivec over flow
#define  ACL_ERROR_RT_OVER_FLOW                  207017 // common over flow
#define  ACL_ERROR_RT_DEVICE_OOM                 207018 // device oom
#define  ACL_ERROR_RT_FEATURE_NOT_SUPPORT_UPDATE_OP 207019 // not support to update this op

#define  ACL_ERROR_RT_INTERNAL_ERROR             507000 // runtime internal error
#define  ACL_ERROR_RT_TS_ERROR                   507001 // ts internel error
#define  ACL_ERROR_RT_STREAM_TASK_FULL           507002 // task full in stream
#define  ACL_ERROR_RT_STREAM_TASK_EMPTY          507003 // task empty in stream
#define  ACL_ERROR_RT_STREAM_NOT_COMPLETE        507004 // stream not complete
#define  ACL_ERROR_RT_END_OF_SEQUENCE            507005 // end of sequence
#define  ACL_ERROR_RT_EVENT_NOT_COMPLETE         507006 // event not complete
#define  ACL_ERROR_RT_CONTEXT_RELEASE_ERROR      507007 // context release error
#define  ACL_ERROR_RT_SOC_VERSION                507008 // soc version error
#define  ACL_ERROR_RT_TASK_TYPE_NOT_SUPPORT      507009 // task type not support
#define  ACL_ERROR_RT_LOST_HEARTBEAT             507010 // ts lost heartbeat
#define  ACL_ERROR_RT_MODEL_EXECUTE              507011 // model execute failed
#define  ACL_ERROR_RT_REPORT_TIMEOUT             507012 // report timeout
#define  ACL_ERROR_RT_SYS_DMA                    507013 // sys dma error
#define  ACL_ERROR_RT_AICORE_TIMEOUT             507014 // aicore timeout
#define  ACL_ERROR_RT_AICORE_EXCEPTION           507015 // aicore exception
#define  ACL_ERROR_RT_AICORE_TRAP_EXCEPTION      507016 // aicore trap exception
#define  ACL_ERROR_RT_AICPU_TIMEOUT              507017 // aicpu timeout
#define  ACL_ERROR_RT_AICPU_EXCEPTION            507018 // aicpu exception
#define  ACL_ERROR_RT_AICPU_DATADUMP_RSP_ERR     507019 // aicpu datadump response error
#define  ACL_ERROR_RT_AICPU_MODEL_RSP_ERR        507020 // aicpu model operate response error
#define  ACL_ERROR_RT_PROFILING_ERROR            507021 // profiling error
#define  ACL_ERROR_RT_IPC_ERROR                  507022 // ipc error
#define  ACL_ERROR_RT_MODEL_ABORT_NORMAL         507023 // model abort normal
#define  ACL_ERROR_RT_KERNEL_UNREGISTERING       507024 // kernel unregistering
#define  ACL_ERROR_RT_RINGBUFFER_NOT_INIT        507025 // ringbuffer not init
#define  ACL_ERROR_RT_RINGBUFFER_NO_DATA         507026 // ringbuffer no data
#define  ACL_ERROR_RT_KERNEL_LOOKUP              507027 // kernel lookup error
#define  ACL_ERROR_RT_KERNEL_DUPLICATE           507028 // kernel register duplicate
#define  ACL_ERROR_RT_DEBUG_REGISTER_FAIL        507029 // debug register failed
#define  ACL_ERROR_RT_DEBUG_UNREGISTER_FAIL      507030 // debug unregister failed
#define  ACL_ERROR_RT_LABEL_CONTEXT              507031 // label not in current context
#define  ACL_ERROR_RT_PROGRAM_USE_OUT            507032 // program register num use out
#define  ACL_ERROR_RT_DEV_SETUP_ERROR            507033 // device setup error
#define  ACL_ERROR_RT_VECTOR_CORE_TIMEOUT        507034 // vector core timeout
#define  ACL_ERROR_RT_VECTOR_CORE_EXCEPTION      507035 // vector core exception
#define  ACL_ERROR_RT_VECTOR_CORE_TRAP_EXCEPTION 507036 // vector core trap exception
#define  ACL_ERROR_RT_CDQ_BATCH_ABNORMAL         507037 // cdq alloc batch abnormal
#define  ACL_ERROR_RT_DIE_MODE_CHANGE_ERROR      507038 // can not change die mode
#define  ACL_ERROR_RT_DIE_SET_ERROR              507039 // single die mode can not set die
#define  ACL_ERROR_RT_INVALID_DIEID              507040 // invalid die id
#define  ACL_ERROR_RT_DIE_MODE_NOT_SET           507041 // die mode not set
#define  ACL_ERROR_RT_AICORE_TRAP_READ_OVERFLOW       507042 // aic trap read overflow
#define  ACL_ERROR_RT_AICORE_TRAP_WRITE_OVERFLOW      507043 // aic trap write overflow
#define  ACL_ERROR_RT_VECTOR_CORE_TRAP_READ_OVERFLOW  507044 // aiv trap read overflow
#define  ACL_ERROR_RT_VECTOR_CORE_TRAP_WRITE_OVERFLOW 507045 // aiv trap write overflow
#define  ACL_ERROR_RT_STREAM_SYNC_TIMEOUT        507046 // stream sync time out
#define  ACL_ERROR_RT_EVENT_SYNC_TIMEOUT         507047 // event sync time out
#define  ACL_ERROR_RT_FFTS_PLUS_TIMEOUT          507048 // ffts+ timeout
#define  ACL_ERROR_RT_FFTS_PLUS_EXCEPTION        507049 // ffts+ exception
#define  ACL_ERROR_RT_FFTS_PLUS_TRAP_EXCEPTION   507050 // ffts+ trap exception
#define  ACL_ERROR_RT_SEND_MSG                   507051 // hdc send msg fail
#define  ACL_ERROR_RT_COPY_DATA                  507052 // copy data fail
#define  ACL_ERROR_RT_DEVICE_MEM_ERROR           507053 // device MEM ERROR
#define  ACL_ERROR_RT_HBM_MULTI_BIT_ECC_ERROR    507054 // hbm Multi-bit ECC error
#define  ACL_ERROR_RT_SUSPECT_DEVICE_MEM_ERROR   507055 // suspect device MEM ERROR
#define  ACL_ERROR_RT_LINK_ERROR                 507056 // link ERROR
#define  ACL_ERROR_RT_SUSPECT_REMOTE_ERROR       507057 // suspect remote ERROR
#define  ACL_ERROR_RT_DRV_INTERNAL_ERROR         507899 // drv internal error
#define  ACL_ERROR_RT_AICPU_INTERNAL_ERROR       507900 // aicpu internal error
#define  ACL_ERROR_RT_SOCKET_CLOSE               507901 // hdc disconnect
#define  ACL_ERROR_RT_AICPU_INFO_LOAD_RSP_ERR    507902 // aicpu info load response error
#define  ACL_ERROR_RT_STREAM_CAPTURE_INVALIDATED 507903 // capture status is invalidated
#define  ACL_ERROR_RT_COMM_OP_RETRY_FAIL         507904 // hccl operation retry failed

#define ACLNN_CLEAR_DEVICE_STATE_FAIL            574007  // voltage recovery fail
#define ACLNN_STRESS_BIT_FAIL                    574006
#define ACLNN_STRESS_LOW_BIT_FAIL                574008
#define ACLNN_STRESS_HIGH_BIT_FAIL               574009

#ifdef __cplusplus
}
#endif
#endif // __INC_EXTERNEL_RT_ERROR_CODES_H__
