import os
import stat
import logging
from logging.handlers import RotatingFileHandler
from functools import wraps
import uuid
import time
import glob
import warnings
import torch
from torch.nn import Module

import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error
from torch_npu.asd.asd import _silent_check_decorator, silent_check, _matmul_silent_check_decorator, matmul_check


original_call = Module.__call__
DEFAULT_FALGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
DEFAULT_PERMISSION = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
loggerSilent = logging.getLogger("torch_npu.silent_check")


class PerfDumpState:
    def __init__(self):
        self.module_dict = {}
        self.is_outer_call = True
        self.log_file_name = ""
        self.last_time = None
        self.has_log = False
        self.local_uuid = ""
        self.uuid = ""

    def add_module_dict(self, module):
        module_list = []
        for _, sub_module in module.named_modules():
            if sub_module != module:
                module_list.append(sub_module)
        self.module_dict[module] = module_list
    
    def is_child_module(self, module):
        for item in self.module_dict.items():
            if module in item[1]:
                return True
        return False

perf_dump_state = PerfDumpState()
perf_dump_enable = False


class CustomRotatingFileHandler(RotatingFileHandler):
    def doRollover(self):
        super().doRollover()

        fd = os.open(self.baseFilename, DEFAULT_FALGS, DEFAULT_PERMISSION)
        os.fchmod(fd, DEFAULT_PERMISSION)

        with os.fdopen(fd, 'w') as f:
            f.write(f"[LOCALUUID]:{perf_dump_state.local_uuid}\n")
            f.write("[FRAMEWORK]:PyTorch\n")
            f.write(f"[UUID]:{perf_dump_state.uuid}\n")
            f.close()
        os.close(fd)


def _is_loss_module(module):
    return isinstance(module, torch.nn.modules.loss._Loss)


def _validate_path(path):
    if os.path.isabs(path) and os.path.exists(path):
        return True
    else:
        return False
    

def _get_perf_dump_path():
    perf_dump_path = os.environ.get("PERF_DUMP_PATH")
    if perf_dump_path and _validate_path(perf_dump_path):
        return perf_dump_path
    else:
        raise RuntimeError("PERF_DUMP_PATH is empty or invalid." + pta_error(ErrCode.VALUE))


def delete_pref_pt_logs(perf_dump_path, device_id):
    log_pattern = os.path.join(perf_dump_path, f"perf_pt_*_{device_id}.log*")
    log_files = glob.glob(log_pattern)
    
    for log_file in log_files:
        if os.path.islink(log_file):
            continue
        try:
            os.remove(log_file)
        except Exception as e:
            raise RuntimeError(f"Failed to delete {log_file}. Please delete it manually." + pta_error(ErrCode.SYSCALL)) from e


def _get_uuid():
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT")

    if master_addr is None or master_port is None:
        return "127.0.0.1_8888"
    
    return master_addr + "_" + master_port
    

def _setup_logger(name, path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    file_handler = CustomRotatingFileHandler(path, maxBytes=50 * 1024 * 1024, backupCount=3)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.propagate = False


def _perf_dump_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        global perf_dump_enable
        global perf_dump_state

        if not torch.npu.is_initialized():
            return func(self, *args, **kwargs)

        if perf_dump_enable:
            if not perf_dump_state.has_log:
                perf_dump_path = _get_perf_dump_path()
                pid = os.getpid()
                device_id = torch_npu.npu.current_device()
                delete_pref_pt_logs(perf_dump_path, device_id)
                perf_dump_state.local_uuid = uuid.uuid4()
                perf_dump_state.uuid = _get_uuid()
                perf_dump_state.log_file_name = os.path.join(perf_dump_path, f"perf_pt_{pid}_{device_id}.log")
                _setup_logger("perf_logger", perf_dump_state.log_file_name)
                logger = logging.getLogger("perf_logger")
                logger.info(f"[LOCALUUID]:{perf_dump_state.local_uuid}")
                logger.info("[FRAMEWORK]:PyTorch")
                logger.info(f"[UUID]:{perf_dump_state.uuid}")
                os.chmod(perf_dump_state.log_file_name, DEFAULT_PERMISSION)
                perf_dump_state.has_log = True

            if perf_dump_state.is_outer_call:
                if not perf_dump_state.is_child_module(self) and not _is_loss_module(self):
                    current_time = int(time.time() * 1000)
                    logger = logging.getLogger("perf_logger")
                    if perf_dump_state.last_time is not None:
                        logger.info(f"[STEPTIME]:{perf_dump_state.last_time},{current_time}")
                    perf_dump_state.last_time = current_time
                    perf_dump_state.add_module_dict(self)
                perf_dump_state.is_outer_call = False
                self.visited = True

        tmp = func(self, *args, **kwargs)

        if perf_dump_enable:
            if hasattr(self, "visited") and self.visited:
                perf_dump_state.is_outer_call = True
                self.visited = False

        return tmp
    return wrapper


@_perf_dump_decorator
@_silent_check_decorator
@_matmul_silent_check_decorator
def _custom_call(self, *args, **kwargs):
    return original_call(self, *args, **kwargs)


def _parse_config(config):
    config_dict = {}
    if config:
        config_items = config.split(',')
        for item in config_items:
            key_value = item.split(':')
            if len(key_value) == 2:
                key, value = key_value
                config_dict[key] = value
    return config_dict


def _prase_asd_config(asd_config):
    # checksum
    with_checksum_str = asd_config.get("with_checksum", "false")
    if with_checksum_str not in ["true", "false"]:
        raise ValueError("NPU_ASD_CONFIG-with_checksum should be true or false. For details, 0 as `with checksum closed`, 1 as `with checksum opened`." + pta_error(ErrCode.VALUE))
    with_checksum = with_checksum_str == "true"
    matmul_check.set_with_checksum(with_checksum)

    # cooldown
    cooldown = asd_config.get("cooldown", "5")
    if cooldown.isdigit() and cooldown != "0":
        matmul_check.set_cooldown(int(cooldown))
    else:
        warnings.warn(f"Warning: NPU_ASD_CONFIG-cooldown is invalid, use the default value of 5.")

    # strikes_num
    strikes_num = asd_config.get("strikes_num", "3")
    if strikes_num.isdigit() and strikes_num != "0":
        matmul_check.set_strikes_num(int(strikes_num))
    else:
        warnings.warn(f"Warning: NPU_ASD_CONFIG-strikes_num is invalid, use the default value of 3.")

    # strikes_window
    strikes_window = asd_config.get("strikes_window", "480")
    if strikes_window.isdigit() and strikes_window != "0":
        matmul_check.set_strikes_window(int(strikes_window))
    else:
        warnings.warn(f"Warning: NPU_ASD_CONFIG-strikes_window is invalid, use the default value of 480.")

    # checksum_cooldown
    checksum_cooldown = asd_config.get("checksum_cooldown", "180")
    if checksum_cooldown.isdigit() and checksum_cooldown != "0":
        matmul_check.set_checksum_cooldown(int(checksum_cooldown))
    else:
        warnings.warn(f"Warning: NPU_ASD_CONFIG-checksum_cooldown is invalid, use the default value of 180.")

    # upper_thresh1
    upper_thresh1 = asd_config.get("upper_thresh1", "1000000")
    if upper_thresh1.isdigit() and int(upper_thresh1) >= 3:
        matmul_check.set_upper_thresh1(int(upper_thresh1))
    else:
        warnings.warn(f"Warning: NPU_ASD_CONFIG-upper_thresh1 is invalid, use the default value of 1000000.")

    # upper_thresh2
    upper_thresh2 = asd_config.get("upper_thresh2", "100")
    if upper_thresh2.isdigit() and int(upper_thresh2) >= 3:
        matmul_check.set_upper_thresh2(int(upper_thresh2))
    else:
        warnings.warn(f"Warning: NPU_ASD_CONFIG-upper_thresh2 is invalid, use the default value of 100.")

    # grad_sample_interval
    grad_sample_interval = asd_config.get("grad_sample_interval", "3")
    if grad_sample_interval.isdigit() and grad_sample_interval != "0":
        matmul_check.set_grad_sample_interval(int(grad_sample_interval))
    else:
        warnings.warn(f"Warning: NPU_ASD_CONFIG-grad_sample_interval is invalid, use the default value of 3.")


def add_perf_dump_patch():
    global perf_dump_enable

    perf_dump_config = os.getenv("PERF_DUMP_CONFIG")
    config_dict = _parse_config(perf_dump_config)
    enable_value = config_dict.get("enable", "false")
    perf_dump_enable = enable_value.lower() == "true"

    asd_enable = 0
    asd_config = os.getenv("NPU_ASD_CONFIG", None)
    if asd_config is not None:
        asd_config_dict = _parse_config(asd_config)
        asd_config_enable = asd_config_dict.get("enable", "false")
        if asd_config_enable not in ["true", "false"]:
            raise ValueError("NPU_ASD_CONFIG-enable should be true or false. For details, false as `ASD closed`, true as `ASD opened`." + pta_error(ErrCode.VALUE))
        if asd_config_enable == "true":
            warnings.warn(f'Silent data corruption check may take up 1.5GB device memory, please make sure there are enough free space in device')
            _prase_asd_config(asd_config_dict)
            asd_enable = 1
            matmul_check.set_matmul_hook_enable(asd_enable)
            loggerSilent.info(f"Silent check 3.0 version will be enabled. The checksum enable is {matmul_check.get_with_checksum()}, "
                              f"cooldown is {matmul_check.get_cooldown()}, strikes_num is {matmul_check.get_strikes_num()}, strikes_window is {matmul_check.get_strikes_window()}, "
                              f"checksum_cooldown is {matmul_check.get_checksum_cooldown()}, upper_thresh1 is {matmul_check.get_upper_thresh1()}, "
                              f"upper_thresh2 is {matmul_check.get_upper_thresh2()}. grad_sample_interval is {matmul_check.get_grad_sample_interval()}.")
    else:
        asd_value = os.getenv("NPU_ASD_ENABLE", "0")
        if torch_npu._C._get_silent_check_version() == 1:
            if asd_value == "1":
                warnings.warn(f"Warning: CANN version lower than 8.0.RC3 and currently does not support silent check 2.0 version or later. It will switch to 1.0 version.")
        else:
            if asd_value not in ["0", "1", "2", "3"]:
                raise ValueError("NPU_ASD_ENABLE should be 0, 1, 2 or 3. For details, 0 as `ASD closed`, "
                                "1 as `ASD opened, print error logs`, "
                                "2 as `ASD opened, print error logs and raise exception`, "
                                "3 as `ASD opened, print debug logs and raise exception`" + pta_error(ErrCode.VALUE))
            asd_enable = int(asd_value)
            if asd_enable:
                warnings.warn(f"Warning: Silent check 2.0 version will be enabled. The asd_detect is {asd_enable}. It is recommended to enable silent check v3 using the NPU_ASD_CONFIG.\n"
                              "Silent data corruption check may take up 1.5GB device memory, please make sure there are enough free space in device. ")
                silent_check.set_check_enable(asd_enable)

    if perf_dump_enable or asd_enable:
        Module.__call__ = _custom_call
