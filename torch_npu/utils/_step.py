import os
import logging
import uuid
import time
import torch
from torch.nn import Module

import torch_npu
from torch_npu.utils.error_code import ErrCode, pta_error


original_call = Module.__call__


class PerfDumpState:
    def __init__(self):
        self.module_dict = {}
        self.is_outer_call = True
        self.log_file_name = ""
        self.last_time = None
        self.has_log = False

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
    

def _setup_logger(name, path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.propagate = False


def _custom_call(self, *args, **kwargs):
    global perf_dump_state

    if not torch.npu.is_initialized():
        return original_call(self, *args, **kwargs)
    
    if not perf_dump_state.has_log:
        perf_dump_path = _get_perf_dump_path()
        pid = os.getpid()
        device_id = torch_npu.npu.current_device()
        random_uuid = uuid.uuid4()
        perf_dump_state.log_file_name = os.path.join(perf_dump_path, f"perf_pt_{pid}_{device_id}.log")
        _setup_logger("perf_logger", perf_dump_state.log_file_name)
        logger = logging.getLogger("perf_logger")
        logger.info(f"[UUID]:{random_uuid}")
        logger.info("[FRAMEWORK]:PyTorch")
        perf_dump_state.has_log = True

    if perf_dump_state.is_outer_call:
        if not perf_dump_state.is_child_module(self) and not _is_loss_module(self):
            current_time = int(time.time() * 1000)
            logger = logging.getLogger("perf_logger")
            logger.info(f"[STEPTIME]:{perf_dump_state.last_time},{current_time}")
            perf_dump_state.last_time = current_time
            perf_dump_state.add_module_dict(self)
        perf_dump_state.is_outer_call = False
        self.visited = True

    tmp = original_call(self, *args, **kwargs)

    if hasattr(self, "visited") and self.visited:
        perf_dump_state.is_outer_call = True
        self.visited = False
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
    config_dict = _parse_perf_config()
    enable_value = config_dict.get("enable", "false")
    perf_dump_enable = enable_value.lower() == "true"

    if perf_dump_enable:
        Module.__call__ = _custom_call
