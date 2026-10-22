# HCCL\_EVENT\_TIMEOUT

## Function Description

When HCCL is used as the communication backend, this environment variable sets the timeout period for waiting for an event to complete.

Within a process, after initializing pyACL by calling the `acl.init` API, call the `acl.rt.set_op_wait_timeout` API to set the timeout period. Subsequent tasks issued by calling the `acl.rt.stream_wait_event` API in this process support waiting within the set timeout period. If the waiting time exceeds the set timeout period, pyACL returns an error.

The unit is seconds (s), the value range is \[0, 2147483647\], and the default value is 1868. When configured to 0, the wait never times out.

> [!NOTE]
>
> - For details about the `acl.init` API, see the "[Function: init](https://gitcode.com/cann/runtime/blob/9.2.0/python/docs/en/api_ref/runtime/init/function-init.md)" section in *CANN Runtime API*.
> - For details about the `acl.rt.set_op_wait_timeout` API, see the "[Function: set\_op\_wait\_timeout](https://gitcode.com/cann/runtime/blob/9.2.0/python/docs/en/api_ref/runtime/event/function-set_op_wait_timeout.md)" section in *CANN Runtime API*.
> - For details about the `acl.rt.stream_wait_event` API, see the "[Function: stream\_wait\_event](https://gitcode.com/cann/runtime/blob/9.2.0/python/docs/en/api_ref/runtime/event/function-stream_wait_event.md)" section in *CANN Runtime API*.

## Configuration Example

```bash
export HCCL_EVENT_TIMEOUT=1800
```

## Usage Constraints

When this environment variable is configured, its value must be greater than the value of `HCCL_EXEC_TIMEOUT`. For details about `HCCL_EXEC_TIMEOUT`, see the "[HCCL\_EXEC\_TIMEOUT](https://gitcode.com/cann/hccl/blob/9.2.0/docs/en/user_guide/hccl_env/HCCL_EXEC_TIMEOUT.md)" section in the *CANN HCCL Collective Communication Library*.

## Supported Products

<!-- npu="910" id1 -->
- <term>Atlas training products</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 training products</term>
<!-- end id3 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT products</term>
<!-- end id4 -->
