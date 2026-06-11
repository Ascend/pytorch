# 故障处理

- [故障处理流程](troubleshooting_process.md)
- [Error Code介绍](error_codes_introduction.md)
  - [使用说明](usage_instruction.md)
  - [错误码信息](error_code_information.md)

- [报错信息分析指导](error_information_analysis_guide.md)
  - [报错信息分析说明](error_information_introduction.md)
  - [报错信息分类](error_information_classification.md)
    - [回显信息](command_output.md)
    - [plog日志信息](plog_log.md)

  - [报错信息分析](error_information_analysis.md)
    - [报错信息分析流程](error_information_analysis_process.md)
    - [主要进程与线程说明](process_thread_reference.md)
  - [coredump问题定位](locating_coredump_faults.md)

- [故障案例集](troubleshooting_cases.md)
  - [通信算子传入非连续tensor](communication_operator_transfers_Non-contiguous_tensors.md)
  - [调用算子参数校验失败](failed_verify_op_parameters.md)
  - [分布式任务端口号被占用](port_number_distributed_task_in_use.md)
  - [用于梯度计算的变量被inplace操作](variables_used_gradient_computation_modified_by_in-place_op.md)
  - [调用不支持的算子](unsupported_op_called.md)
  - [HCCL超时](HCCL_timeout.md)
  - [算子调用报错](operator_called_error.md)
  - [初始化报错](initialization_error.md)
  - [通信域建链超时](communication_domain_link_establishment_timeout.md)
  - [使用NZ格式后精度异常](nz_format_precision_issue.md)
