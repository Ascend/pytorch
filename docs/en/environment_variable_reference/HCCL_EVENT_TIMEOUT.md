# HCCL\_EVENT\_TIMEOUT

## Function Description

When HCCL is used as the communication backend, this environment variable sets the timeout period for waiting for an event to complete.

Within a process, after initializing pyACL by calling the `acl.init` API, call the `acl.rt.set_op_wait_timeout` API to set the timeout period. Subsequent tasks issued by calling the `acl.rt.stream_wait_event` API in this process support waiting within the set timeout period. If the waiting time exceeds the set timeout period, pyACL returns an error.

The unit is seconds (s), the value range is \[0, 2147483647\], and the default value is 1868. When configured to 0, the wait never times out.

> [!NOTE]
>
> - For details about the `acl.init` API, see the "Function: init" section in *CANN Runtime API*.
<!-- [Function: init](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/runtimeapi/aclpythondevg_01_0851.html) -->
> - For details about the `acl.rt.set_op_wait_timeout` API, see the "Function: set\_op\_wait\_timeout" section in *CANN Runtime API*.
<!-- [Function: set\_op\_wait\_timeout](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/runtimeapi/aclpythondevg_01_0102.html) -->
> - For details about the `acl.rt.stream_wait_event` API, see the "Function: stream\_wait\_event" section in *CANN Runtime API*.
<!-- [Function: stream\_wait\_event](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/runtimeapi/aclpythondevg_01_0101.html) -->

## Configuration Example

```bash
export HCCL_EVENT_TIMEOUT=1800
```

## Usage Constraints

When this environment variable is configured, its value must be greater than the value of `HCCL_EXEC_TIMEOUT`. For details about `HCCL_EXEC_TIMEOUT`, see the "HCCL\_EXEC\_TIMEOUT" section in the *CANN HCCL Collective Communication Library*.
<!-- [HCCL\_EXEC\_TIMEOUT](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/hccl_env/HCCL_EXEC_TIMEOUT.md) -->

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950DT</term>
