# WatchDog

## Introduction

Without compromising the performance and accuracy of LLM training, WatchDog can quickly and reliably detect errors. It monitors processes, significantly improving the reliability of PyTorch distributed training using HCCL. Essentially, WatchDog prevents distributed training from being blocked by capturing errors that occur in collective communication, throwing exceptions from child processes, and stopping the main process training.

**Figure 1**  WatchDog working diagram  
![figure1](../figures/watchdog_working_principle.png)

The preceding figure shows how the WatchDog process works in distributed training. First, the WatchDog thread is started when each process initializes `process_group`, with one WatchDog thread monitoring the `process_group` in a single process. Then, the WatchDog thread asynchronously monitors collective communication anomalies in the `workCleanupLoop` subfunction. After capturing an anomaly, it re-throws the exception in the WatchDog main function, allowing the main training process to perceive the anomaly and quickly terminate the training task.

WatchDog supports not only operator execution anomaly monitoring, communication timeout monitoring, and timeout analysis, but also ERROR CQE detection. CQE detection checks the communication link status of the device network port. When the communication link is abnormal, it is usually reflected as an ERROR CQE.

## Use Scenario

During LLM training, if the model gets stuck after a collective communication anomaly, the training cannot be terminated in time, resulting in resource waste. Using this feature can prevent this situation.

## Usage Guide

You can use the `HCCL_ASYNC_ERROR_HANDLING` environment variable to enable or disable WatchDog.

Values of `HCCL_ASYNC_ERROR_HANDLING`:

- 0: Disables asynchronous error handling.
- 1: Enables asynchronous error handling.

The default value is 0 when the PyTorch version is 1.11.0, and 1 when the PyTorch version is 2.1.0 or later.

For details about this environment variable, see the "[HCCL_ASYNC_ERROR_HANDLING](../environment_variable_reference/HCCL_ASYNC_ERROR_HANDLING.md)" section in *Environment Variable Reference*.

## Usage Example

Enable asynchronous error handling:

```shell
export HCCL_ASYNC_ERROR_HANDLING=1
```

## Constraints

- This environment variable is only applicable to neural network scenarios built on the PyTorch framework, with HCCL used as the communication backend.
- When enabling asynchronous error handling through this environment variable, you are advised to set the timeout of `new_group` and `init_process_group` to a value greater than the time configured by the `HCCL_CONNECT_TIMEOUT` and `HCCL_EXEC_TIMEOUT` environment variables to better identify the cause of HCCL timeouts. For details about `HCCL_CONNECT_TIMEOUT`, see the "HCCL_CONNECT_TIMEOUT" section in the *CANN HCCL Collection Communication Library*. For details about `HCCL_EXEC_TIMEOUT`, see the "HCCL_EXEC_TIMEOUT" section in the *CANN HCCL Collection Communication Library*.
<!-- see the  "[HCCL_CONNECT_TIMEOUT](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/hccl_env/HCCL_CONNECT_TIMEOUT.md)" section in the *CANN HCCL Collection Communication Library*. For details about `HCCL_EXEC_TIMEOUT`, see the "[HCCL_EXEC_TIMEOUT](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/hccl_env/HCCL_EXEC_TIMEOUT.md)"  section in the *CANN HCCL Collection Communication Library*. -->
