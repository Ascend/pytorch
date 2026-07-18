# RANK\_TABLE\_FILE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:05:44.015Z pushedAt=2026-06-16T03:14:22.347Z -->

## Function Description

This environment variable configures the path to the `RANK_TABLE_FILE`, which is used for collective communication domain establishment.

- When not configured, collective communication domain links are established through the default negotiation process.
- When configured and the full file path is valid, collective communication domain establishment is performed through the `RANK_TABLE_FILE`.

This environment variable is not configured by default.

> [!NOTE]
> In the scenario where `RANK_TABLE_FILE` is configured, if the error "RuntimeError: The Inner Error ..." occurs during distributed model training, it is recommended to appropriately increase the `HCCL_CONNECT_TIMEOUT` value to avoid link establishment timeout caused by the absence of negotiation in the rank table scenario. For details, see [Encountering the Error "RuntimeError: The Inner Error ..." During Distributed Model Training](runtimeerror_Inner_Error.md).

## Configuration Example

Example of enabling link setup via ranktable file:

```bash
export RANK_TABLE_FILE=/home/ranktable.json
```

> [!CAUTION]
>
> - If the configured file path does not exist, collective communication domain establishment will proceed through the default negotiation process.
> - If the configured file path exists but the configuration information is incorrect, collective communication domain establishment will not proceed through the default negotiation process, and a corresponding error will be reported during actual communication.

Example of disabling link setup via ranktable file:

```bash
unset RANK_TABLE_FILE
```

## Usage Constraints

- The configured file path cannot be a symbolic link and must have read permission.
- The configured file must be in JSON format. For details, see the "Rank Table Configuration" section in [CANN Huawei Collective Communication Library (HCCL)](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/hcclug/hcclug_000001.html).

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
