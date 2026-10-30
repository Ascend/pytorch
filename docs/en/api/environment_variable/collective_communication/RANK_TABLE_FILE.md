# RANK\_TABLE\_FILE

## Function Description

This environment variable configures the path to the `RANK_TABLE_FILE`, which is used for collective communicator establishment.

- When not configured, collective communicator links are established through the default negotiation process.
- When configured and the full file path is valid, collective communicator links are established through the `RANK_TABLE_FILE`.

This environment variable is not configured by default.

> [!NOTE]
>
> In the scenario where RANK\_TABLE\_FILE is configured, if the error "RuntimeError: The Inner Error ..." occurs during model distributed training, you are advised to appropriately increase the timeout duration of HCCL\_CONNECT\_TIMEOUT to avoid the link establishment timeout caused by the absence of negotiation in the ranktable scenario. For details, see [Encountering the Error "RuntimeError: The Inner Error ..." During Model Distributed Training](../appendix/runtimeerror_Inner_Error.md).

## Configuration Example

Example of enabling link establishment using the ranktable file:

```bash
export RANK_TABLE_FILE=/home/ranktable.json
```

> [!CAUTION]
>
> - If the configured file path does not exist, collective communicator links are established through the default negotiation process.
> - If the configured file path exists but the configuration information is incorrect, collective communicator links are not established through the default negotiation process. Instead, a corresponding error is reported during actual communication.

Example of disabling link establishment using the ranktable file:

```bash
unset RANK_TABLE_FILE
```

## Usage Constraints

- The configured file path cannot be a symbolic link and must have read permission.
- The configured file must be in JSON format. For details, see the "[Cluster Information Configuration](https://gitcode.com/cann/hccl/blob/9.2.0/docs/en/user_guide/cluster_info_config/intro.md)" chapter in CANN HCCL Library.

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
