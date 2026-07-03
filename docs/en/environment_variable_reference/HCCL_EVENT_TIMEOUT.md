# HCCL_EVENT_TIMEOUT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:03:46.470Z pushedAt=2026-06-16T03:14:22.196Z -->

## Feature Description

When HCCL is used as the communication backend, this environment variable sets the timeout period for waiting for an event to complete.

Within a process, after initializing pyACL by calling the `acl.init` API, call the `acl.rt.set_op_wait_timeout` API to set the timeout period. Subsequent tasks issued by calling the `acl.rt.stream_wait_event` API in this process support waiting within the set timeout period. If the waiting time exceeds the set timeout period, pyACL returns an error.

The unit is second (s), the value range is [0, 2147483647], and the default value is 1868. When set to 0, it means never timeout.

> [!NOTE]
>
> - For details about the `acl.init` API, see the "[Function: init](https://www.hiascend.com/document/detail/en/canncommercial/900/others/fvsearch/aclpythondevg_01_0823.html)" section in *CANN Feature Vector Retrieval*.
> - For details about the `acl.rt.set_op_wait_timeout` API, see the "[Function: set\_op\_wait\_timeout](https://www.hiascend.com/document/detail/en/canncommercial/900/API/runtimeapi/aclpythondevg_01_0102.html)" section in *CANN Runtime API*.
> - For details about the `acl.rt.stream_wait_event` API, see the "[Function: stream\_wait\_event](https://www.hiascend.com/document/detail/en/canncommercial/900/API/runtimeapi/aclpythondevg_01_0101.html)" section in *CANN Runtime API*.

## Configuration Example

```bash
export HCCL_EVENT_TIMEOUT=1800
```

## Usage Constraints

When this environment variable is configured, its value must be greater than the value of `HCCL_EXEC_TIMEOUT`. For details about `HCCL_EXEC_TIMEOUT`, see the "[HCCL\_EXEC\_TIMEOUT](https://www.hiascend.com/document/detail/en/canncommercial/900/maintenref/envvar/envref_07_0078.html)" section in *CANN Environment Variable Reference*.

## Supported Products

- <term>Atlas training series</term>
- <term>Atlas A2 training series</term>
- <term>Atlas A3 training series</term>
