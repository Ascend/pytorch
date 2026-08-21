# RANK\_TABLE\_FILE

## Feature Description

This environment variable configures the path to the `RANK_TABLE_FILE`, which is used for collective communication domain establishment.

- When not configured, collective communication domain links are established through the default negotiation process.
- When configured and the full file path is valid, collective communication domain links are established through the `RANK_TABLE_FILE`.

This environment variable is not configured by default.

> [!NOTE]
>
> When the `RANK_TABLE_FILE` is configured, if the error "RuntimeError: The Inner Error ..." occurs during distributed model training, you are advised to appropriately increase the timeout value of `HCCL_CONNECT_TIMEOUT` to avoid link establishment timeout caused by the absence of negotiation in the rank table scenario. For details, see [Encountering the Error "RuntimeError: The Inner Error ..." During Distributed Model Training](runtimeerror_Inner_Error.md).

## Configuration Example

Example of enabling link establishment using the rank table file:

```bash
export RANK_TABLE_FILE=/home/ranktable.json
```

> [!CAUTION]
>
> - If the configured file path does not exist, collective communication domain links are established through the default negotiation process.
> - If the configured file path exists but the configuration information is incorrect, collective communication domain links are not established through the default negotiation process. Instead, a corresponding error is reported during actual communication.

Example of disabling link establishment using the rank table file:

```bash
unset RANK_TABLE_FILE
```

## Usage Constraints

- The configured file path cannot be a symbolic link and must have read permission.
- The configured file must be in JSON format. For details, see the "[Cluster Information Configuration](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/commlib/hcclug/docs/zh/user_guide/cluster_info_config/intro.md)" chapter in CANN HCCL Library.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950DT</term>
