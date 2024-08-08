import os
import stat
import logging
from logging.handlers import RotatingFileHandler
import uuid
import time
import glob
import warnings
import torch
from torch.nn import Module

import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error


original_call = Module.__call__
DEFAULT_FALGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
DEFAULT_PERMISSION = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP


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
first_forward = True
input_hook_flag = True
weight_hook_flag = True
asd_enable = 1


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
    if os.path.isabs(path) and os.path.exists(path) and not os.path.islink(path):
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


def input_hook(grad):
    torch_npu._C._npu_set_call_state("forward")
    return grad


def output_hook(grad):
    torch_npu._C._npu_set_call_state("backward")
    return grad


def _custom_call(self, *args, **kwargs):    
    global perf_dump_enable
    global perf_dump_state

    global asd_enable
    global first_forward
    global input_hook_flag
    global weight_hook_flag

    if not torch.npu.is_initialized():
        return original_call(self, *args, **kwargs)

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

    if asd_enable:
        if input_hook_flag:
            for x in args:
                if isinstance(x, torch.Tensor):
                    if x.requires_grad:
                        x.register_hook(input_hook)
                        input_hook_flag = False
                        break

        if weight_hook_flag:
            for param_name, param in self._parameters.items():
                if isinstance(param, torch.Tensor):
                    if param.requires_grad:
                        param.register_hook(input_hook)
                        weight_hook_flag = False
                        break

        if first_forward:
            first_forward = False
            self.outer = True
            if self.training:
                torch_npu._C._npu_set_module_train_state("train")
            else:
                torch_npu._C._npu_set_module_train_state("infer")

    tmp = original_call(self, *args, **kwargs)

    if perf_dump_enable:
        if hasattr(self, "visited") and self.visited:
            perf_dump_state.is_outer_call = True
            self.visited = False

    if asd_enable:
        if hasattr(self, "outer") and self.outer:
            if isinstance(tmp, torch.Tensor):
                if tmp.requires_grad:
                    tmp.register_hook(output_hook)
            input_hook_flag = True
            weight_hook_flag = True
            first_forward = True
            self.outer = False

    return tmp


def _parse_perf_config():
    perf_dump_config = os.getenv("PERF_DUMP_CONFIG")
    config_dict = {}
    if perf_dump_config:
        config_items = perf_dump_config.split(',')
        for item in config_items:
            key_value = item.split(':')
            if len(key_value) == 2:
                key, value = key_value
                config_dict[key] = value
    return config_dict


def add_perf_dump_patch():
    global perf_dump_enable
    global asd_enable

    config_dict = _parse_perf_config()
    enable_value = config_dict.get("enable", "false")
    perf_dump_enable = enable_value.lower() == "true"

    asd_value = os.getenv("NPU_ASD_ENABLE", "0")
    if asd_value not in ["0", "1", "2", "3"]:
        raise ValueError("NPU_ASD_ENABLE should be 0, 1, 2 or 3. For details, 0 as `ASD closed`, "
                         "1 as `ASD opened, print error logs` "
                         "2 as `ASD opened, print error logs and raise exception`, "
                         "3 as `ASD opened, print debug logs and raise exception`" + pta_error(ErrCode.VALUE))
    asd_enable = int(asd_value)

    if asd_enable and not torch_npu._C._npu_support_silentClientV2():        
        warnings.warn(f"Warning: CANN version lower than 8.0.RC3 and currently does not support silent check 2.0 version. It will switch to 1.0 version.")
        asd_enable = 0

    if perf_dump_enable or asd_enable:
        Module.__call__ = _custom_call
