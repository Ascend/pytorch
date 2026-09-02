# 故障处理

- [故障处理流程](troubleshooting_process.md)
- [Error Code介绍](error_code_description/_menu_error_code_description.md)
  - [使用说明](error_code_description/error_code_mechanism.md)
  - [错误码信息](error_code_description/error_code_information.md)

- [报错信息分析指导](error_information_analysis_guide/_menu_error_information_analysis_guide.md)
  - [报错信息分析说明](error_information_analysis_guide/error_information_introduction.md)
  - [报错信息分类](error_information_analysis_guide/error_information_classification/_menu_error_information_classification.md)
    - [回显信息](error_information_analysis_guide/error_information_classification/command_output.md)
    - [plog日志信息](error_information_analysis_guide/error_information_classification/plog_log.md)

  - [报错信息分析](error_information_analysis_guide/error_information_analysis/error_information_analysis.md)
    - [报错信息分析流程](error_information_analysis_guide/error_information_analysis/error_information_analysis_process.md)
    - [主要进程与线程说明](error_information_analysis_guide/error_information_analysis/process_thread_reference.md)
  - [Core Dump问题定位](error_information_analysis_guide/locating_core_dump_faults.md)

- [故障案例集](troubleshooting_cases/_menu_troubleshooting_cases.md)
  - [通信算子传入非连续tensor](troubleshooting_cases/communication_operator_transfers_Non-contiguous_tensors.md)
  - [调用算子参数校验失败](troubleshooting_cases/failed_verify_op_parameters.md)
  - [分布式任务端口号被占用](troubleshooting_cases/port_number_distributed_task_in_use.md)
  - [用于梯度计算的变量被inplace操作](troubleshooting_cases/variables_used_gradient_computation_modified_by_in-place_op.md)
  - [调用不支持的算子](troubleshooting_cases/unsupported_op_called.md)
  - [HCCL超时](troubleshooting_cases/HCCL_timeout.md)
  - [算子调用报错](troubleshooting_cases/operator_called_error.md)
  - [初始化报错](troubleshooting_cases/initialization_error.md)
  - [通信域建链超时](troubleshooting_cases/communication_domain_link_establishment_timeout.md)
  - [使用NZ格式后精度异常](troubleshooting_cases/nz_format_precision_issue.md)
