class CsvHeaders(object):
    # op_summary
    TASK_START_TIME = "Task Start Time(us)"
    OP_SUMMARY_SHOW_HEADERS = ["Op Name", "OP Type", "Task Type", TASK_START_TIME, "Task Duration(us)",
                               "Task Wait Time(us)", "Block Dim"]
    OP_SUMMARY_KERNEL_BASE_HEADERS = ["Name", "Type", "Accelerator Core", "Start Time(us)", "Duration(us)",
                                      "Wait Time(us)", "Block Dim"]
