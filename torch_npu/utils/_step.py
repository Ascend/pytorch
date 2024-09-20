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
from torch_npu.asd.asd import _silent_fault_detector_v2


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
IS_IN_BACKWARD = 0


def input_hook(idx, asd_flag):
    def hook(grad):
        global IS_IN_BACKWARD

        if idx != "":
            IS_IN_BACKWARD = IS_IN_BACKWARD & 1  # 011 & 001 = 001
            _silent_fault_detector_v2.silent_fault_check(idx, asd_flag, grad)
        else:
            IS_IN_BACKWARD = IS_IN_BACKWARD & 2  # 011 & 010 = 010

        if not IS_IN_BACKWARD:
            torch_npu._C._npu_set_call_state("forward")
        return
    return hook


def output_hook(grad):
    global IS_IN_BACKWARD
    IS_IN_BACKWARD = 3  # 011
    torch_npu._C._npu_set_call_state("backward")
    return grad


def _is_inner_module(module):
    return len(module._modules) == 0


class SilentCheckState:
    def __init__(self):
        self.init_param()
        self.init_marks = {}
        self.weight_hook_handles = {}
        self.last_weight_hook_handles = {}
        self.dtype_support = True

    def init_param(self):
        self.first_forward = True
        self.input_hook_flag = False
        self.is_training = False
        self.first_module_id = ""
        self.first_weight = None
        self.last_weight = None
        self.last_tensor = None
        self.last_tensor_id = None
        self.first_tensor_id = None

    def init_module_info(self, module_id, training):
        self.first_module_id = module_id
        self.first_forward = False
        self.is_training = training
        if self.is_training:
            torch_npu._C._npu_set_module_train_state("train")
        else:
            torch_npu._C._npu_set_module_train_state("infer")

    def check_tensor_dtype(self, tensor):
        if not self.dtype_support:
            return
        if isinstance(tensor, torch.Tensor) and tensor.requires_grad and tensor.dtype == torch.float16:
            self.dtype_support = False

    def check_dtype(self, module, *args):
        for x in args:
            self.check_tensor_dtype(x)
        for param_name, param in module._parameters.items():
            self.check_tensor_dtype(param)

    def search_first_weight(self, module):
        # Search the first weight
        if not self.init_marks.get(self.first_module_id, False) and self.first_weight is None:
            for param_name, param in module._parameters.items():
                if isinstance(param, torch.Tensor) and param.requires_grad:
                    self.first_weight = param
                    break

    def register_input_hook_before_call(self, asd_flag, *args):
        # Search the first tensor (if the first tensor is input)
        if self.is_training and not self.input_hook_flag:
            for x in args:
                if isinstance(x, torch.Tensor) and x.requires_grad:
                    x.register_hook(input_hook(self.first_module_id, asd_flag))
                    self.input_hook_flag = True
                    break

    def register_input_hook_after_call(self, output):
        # Search the first tensor (if the first tensor is output of an inner module)
        if not self.input_hook_flag:
            if isinstance(output, torch.Tensor) and output.requires_grad:
                output.register_hook(input_hook(self.first_module_id, asd_enable))
                self.input_hook_flag = True
                self.first_tensor_id = id(output)

    def search_last_weight(self, module):
        # Search the last weight (only in inner module)
        if not self.init_marks.get(self.first_module_id, False) and _is_inner_module(module):
            for param_name, param in module._parameters.items():
                if isinstance(param, torch.Tensor) and param.requires_grad:
                    self.last_weight = param

    def search_last_tensor(self, output):
        # Search the last tensor
        if isinstance(output, torch.Tensor) and output.requires_grad:
            self.last_tensor_id = id(output)
            self.last_tensor = output

    def init_all_hook(self, asd_flag):
        if self.is_training:
            # Otherwise, there is only one weight in the outer module
            if self.first_tensor_id != self.last_tensor_id:
                if self.last_tensor is not None:
                    self.last_tensor.register_hook(output_hook)
                if self.last_weight_hook_handles.get(self.first_module_id, None) is None:
                    if self.last_weight is not None:
                        last_weight_handle = self.last_weight.register_hook(output_hook)
                        self.last_weight_hook_handles[self.first_module_id] = last_weight_handle
                if self.weight_hook_handles.get(self.first_module_id, None) is None:
                    if self.first_weight is not None:
                        first_weight_handle = self.first_weight.register_hook(input_hook("", asd_flag))
                        self.weight_hook_handles[self.first_module_id] = first_weight_handle
            self.init_marks[self.first_module_id] = True


silent_check = SilentCheckState()
asd_enable = 0


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


def _custom_call(self, *args, **kwargs):    
    global perf_dump_enable
    global perf_dump_state

    global asd_enable
    global silent_check
    global IS_IN_BACKWARD

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

    if asd_enable and not IS_IN_BACKWARD:
        if silent_check.first_forward:
            silent_check.init_module_info(id(self), self.training)
            self.outer = True

        if silent_check.is_training and not silent_check.init_marks.get(silent_check.first_module_id, False):
            silent_check.check_dtype(self, *args)
            if not silent_check.dtype_support:
                for value in silent_check.weight_hook_handles.values():
                    if value is not None:
                        value.remove()
                for value in silent_check.last_weight_hook_handles.values():
                    if value is not None:
                        value.remove()
                asd_enable = 0
                warnings.warn(f"Warning: Module has unsupported dtype tensor, silent check will be closed.")

        # Search the first tensor (if the first tensor is input)
        silent_check.register_input_hook_before_call(asd_enable, *args)

    tmp = original_call(self, *args, **kwargs)

    if asd_enable and silent_check.is_training and not IS_IN_BACKWARD:
        # Search the first weight
        silent_check.search_first_weight(self)

        # Search the first tensor (if the first tensor is output of an inner module)
        silent_check.register_input_hook_after_call(tmp)

        # Search the last weight (only in inner module)
        silent_check.search_last_weight(self)
        
        # Search the last tensor
        silent_check.search_last_tensor(tmp)

    if perf_dump_enable:
        if hasattr(self, "visited") and self.visited:
            perf_dump_state.is_outer_call = True
            self.visited = False

    if asd_enable and not IS_IN_BACKWARD:
        if hasattr(self, "outer") and self.outer:
            silent_check.init_all_hook(asd_enable)
            silent_check.init_param()
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
