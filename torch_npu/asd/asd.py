import os
from functools import wraps, partial
import logging
import time
import warnings
import threading
import math
import torch
from torch.nn.functional import layer_norm as origin_layernorm
from torch.nn.functional import embedding as origin_embedding

import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error
from ._silent_fault_data import SilentFaultData, SilentFaultDataV2

__all__ = []


loggerSilent = logging.getLogger("torch_npu.silent_check")


def _Singleton(cls):
    _instances = {}

    def _singleton(*args, **kwargs):
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]
    return _singleton


@_Singleton
class _SilentFaultDetector:
    def __init__(self):
        self.silent_data_dict = dict()
        self.loss_scale = 1.0
        self.global_step = 0
        self.min_step = 7
        self.set_loss_scale_flag = False
        self.idx = None
        self.step = 0
        self.low_step = None
        self.high_step = None

    def set_asd_loss_scale(self, loss_scale=1.0):
        if loss_scale == 0:
            raise ValueError("loss scale cannot be 0." + pta_error(ErrCode.VALUE))
        self.set_loss_scale_flag = True
        self.loss_scale = loss_scale

    def silent_fault_check(self, grad):
        if self.low_step is None or self.high_step is None:
            self.low_step = torch.tensor(0, dtype=torch.int32).npu()
            self.high_step = torch.tensor(self.min_step, dtype=torch.int32).npu()
        if grad.dtype == torch.float16:
            if not self.set_loss_scale_flag:
                return 
            else:
                grad = grad.float() / self.loss_scale

        val = torch.norm(grad)
        idx = self.idx

        if idx not in self.silent_data_dict:
            self.silent_data_dict[idx] = SilentFaultData()

        sfda = self.silent_data_dict[idx]
        
        if self.global_step <= self.min_step:
            self.step += 1
            self.global_step = self.step // (len(self.silent_data_dict) + 1)
            step_tensor = self.low_step
        else:
            step_tensor = self.high_step

        torch_npu._npu_silent_check(grad, val, sfda.pre_val, sfda.min_val, sfda.max_val, step_tensor, self.min_step,
                                    sfda.upper_thresh[0], sfda.sigma_thresh[0], sfda.upper_thresh[1], sfda.sigma_thresh[1])

    def silent_fault_check_hook(self, weight):
        def hook(grad):
            self.idx = id(weight)
            self.silent_fault_check(grad)
            return
        return hook


_silent_fault_detector = _SilentFaultDetector()


def _patch_layernorm(input_layernorm, normalized_shape, *args, **kwargs):
    if input_layernorm is not None and input_layernorm.requires_grad and input_layernorm._backward_hooks is None:
        if "weight" in kwargs:
            input_layernorm.register_hook(_silent_fault_detector.silent_fault_check_hook(kwargs["weight"]))
        elif len(args) > 0:
            input_layernorm.register_hook(_silent_fault_detector.silent_fault_check_hook(args[0]))
    return origin_layernorm(input_layernorm, normalized_shape, *args, **kwargs)


def _patch_embedding(input_embedding, weight, *args, **kwargs):
    if weight is not None and weight.requires_grad and weight._backward_hooks is None:
        weight.register_hook(_silent_fault_detector.silent_fault_check_hook(weight))
    return origin_embedding(input_embedding, weight, *args, **kwargs)


def _asd_patch():
    env_value = os.getenv("NPU_ASD_ENABLE", "0")

    if env_value.isdigit() and int(env_value) and torch_npu._C._get_silent_check_version() > 1:
        return

    if env_value not in ["0", "1"]:
        raise ValueError("NPU_ASD_ENABLE should be 0 or 1!" + pta_error(ErrCode.VALUE))

    if int(env_value):
        torch.nn.functional.layer_norm = _patch_layernorm
        torch.nn.functional.embedding = _patch_embedding


@_Singleton
class _SilentFaultDetectorV2:
    def __init__(self):
        self.silent_data_dict = dict()
        self.min_step = 100

    def silent_fault_check(self, idx, asd_flag, grad):
        if grad is None:
            return
        if grad.dtype != torch.bfloat16 and grad.dtype != torch.float32:
            return

        val = torch.norm(grad)

        if idx not in self.silent_data_dict:
            self.silent_data_dict[idx] = SilentFaultDataV2()

        sfda = self.silent_data_dict[idx]

        torch_npu._npu_silent_check_v2(val, grad, sfda.check_tensor, sfda.step_tensor, self.min_step, sfda.upper_thresh[0],
                                       sfda.sigma_thresh[0], sfda.upper_thresh[1], sfda.sigma_thresh[1], asd_flag)


_silent_fault_detector_v2 = _SilentFaultDetectorV2()
IS_IN_BACKWARD = False


def _input_hook(idx, asd_flag):
    def hook(grad):
        global IS_IN_BACKWARD
        loggerSilent.debug(f"input_hook: IS_IN_BACKWARD is {IS_IN_BACKWARD}, will change to False. idx is {idx}, flag is {asd_flag}")
        IS_IN_BACKWARD = False
        torch_npu._C._npu_set_call_state("forward")
        _silent_fault_detector_v2.silent_fault_check(idx, asd_flag, grad)
        return
    return hook


def _output_hook(grad):
    global IS_IN_BACKWARD
    loggerSilent.debug(f"output_hook: IS_IN_BACKWARD is {IS_IN_BACKWARD}, will change to True.")
    IS_IN_BACKWARD = True
    torch_npu._C._npu_set_call_state("backward")
    return grad


def _is_inner_module(module):
    return len(module._modules) == 0


class _SilentCheckState:
    def __init__(self):
        self.init_param()
        self.init_marks = {}
        self.weight_hook_handles = {}
        self.last_weight_hook_handles = {}
        self.dtype_support = True
        self.check_enable = 0

    def set_check_enable(self, enable):
        self.check_enable = enable

    def get_check_enable(self):
        return self.check_enable

    def init_param(self):
        self.first_forward = True
        self.input_hook_flag = False
        self.is_training = False
        self.first_module_id = ""
        self.first_weight = None
        self.first_weight_id = None
        self.last_weight = None
        self.last_weight_id = None

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
        for _, param in module._parameters.items():
            self.check_tensor_dtype(param)

    def search_first_weight(self, module):
        # Search the first weight
        if not self.init_marks.get(self.first_module_id, False) and self.first_weight is None:
            for _, param in module._parameters.items():
                if isinstance(param, torch.Tensor) and param.requires_grad:
                    self.first_weight = param
                    self.first_weight_id = id(param)
                    break

    def search_last_weight(self, module):
        # Search the last weight (only in inner module)
        if not self.init_marks.get(self.first_module_id, False) and _is_inner_module(module):
            for _, param in module._parameters.items():
                if isinstance(param, torch.Tensor) and param.requires_grad:
                    self.last_weight = param
                    self.last_weight_id = id(param)

    def init_all_hook(self):
        if self.is_training:
            if self.last_weight is not None and self.first_weight is not None:
                # Otherwise, there is only one weight in the outer module
                if self.first_weight_id != self.last_weight_id:
                    loggerSilent.debug(f"init_all_hook: module init, first_module_id is {self.first_module_id}.")
                    if self.last_weight_hook_handles.get(self.first_module_id, None) is None:
                        last_weight_handle = self.last_weight.register_hook(_output_hook)
                        self.last_weight_hook_handles[self.first_module_id] = last_weight_handle
                    if self.weight_hook_handles.get(self.first_module_id, None) is None:
                        first_weight_handle = self.first_weight.register_hook(_input_hook(self.first_module_id, self.check_enable))
                        self.weight_hook_handles[self.first_module_id] = first_weight_handle
                else:
                    loggerSilent.debug(f"init_all_hook: module only have one weight, first_module_id is {self.first_module_id}.")
            self.init_marks[self.first_module_id] = True


silent_check = _SilentCheckState()


def _silent_check_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        global silent_check
        global IS_IN_BACKWARD

        if not torch.npu.is_initialized():
            return func(self, *args, **kwargs)

        if silent_check.get_check_enable() and not IS_IN_BACKWARD:
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
                    silent_check.set_check_enable(0)
                    warnings.warn(f"Warning: Module has unsupported dtype tensor, silent check will be closed.")

        tmp = func(self, *args, **kwargs)

        if silent_check.get_check_enable() and silent_check.is_training and not IS_IN_BACKWARD:
            # Search the first weight
            silent_check.search_first_weight(self)

            # Search the last weight (only in inner module)
            silent_check.search_last_weight(self)

        if silent_check.get_check_enable() and not IS_IN_BACKWARD:
            if hasattr(self, "outer") and self.outer:
                silent_check.init_all_hook()
                silent_check.init_param()
                self.outer = False

        return tmp
    return wrapper


class _MatmulSilentCheck:
    def __init__(self):
        self.init_param()
        self.init_marks = {}
        self.check_stat = {}
        self.hook_dict = {}
        self.registered_modules = []
        self.visited_modules_id = []
        self.matmul_hook_enable = 0
        self.matmul_with_bf16 = False
        self.statistic_value = None
        self.is_outer_call = True
        # link to checksum
        self.matmul_trigger = False
        self.checksum_enable = False
        self.checksum_result = None
        self.checksum_state = None
        self.checksum_state_thread_running = False
        self.checksum_state_thread = None
        # Use another thread to receive the statistic value and detect SDC
        self.check_thread_running = False
        self.check_thread = None
        self._lock = None
        self.queue_len = 1024
        self.statistic_cpu_value = None
        self.name_list = ["" for _ in range(self.queue_len)]
        self.head_index = 0
        self.tail_index = 0
        self.history_abnormal_list = []
        # Parameter filtering
        self.filter_index = -1
        self.filter_interval = 3
        self.invalid_grad_sum = 0
        # Threshold
        self.with_checksum = False
        self.cooldown = 5 # default 5 min cooldown
        self.strikes_num = 3 # default 3 times
        self.strikes_window = 480 # default 480 min
        self.checksum_cooldown = 180 # default 180 min
        self.upper_thresh1 = 1000000 # default 1000000
        self.upper_thresh2 = 100 # default 100
        self.store = None
        self.rank = None

    def init_param(self):
        self.first_forward = True
        self.is_training = False
        self.first_module_id = ""

    def init_module_info(self, module_id, training):
        self.first_module_id = module_id
        self.first_forward = False
        self.is_training = training

    def set_matmul_hook_enable(self, enable):
        self.matmul_hook_enable = enable

    def get_matmul_hook_enable(self):
        return self.matmul_hook_enable

    def set_with_checksum(self, enable):
        self.with_checksum = enable

    def get_with_checksum(self):
        return self.with_checksum

    def set_cooldown(self, cooldown):
        self.cooldown = cooldown

    def get_cooldown(self):
        return self.cooldown

    def set_strikes_num(self, strikes_num):
        self.strikes_num = strikes_num

    def get_strikes_num(self):
        return self.strikes_num

    def set_strikes_window(self, strikes_window):
        self.strikes_window = strikes_window

    def get_strikes_window(self):
        return self.strikes_window

    def set_checksum_cooldown(self, checksum_cooldown):
        self.checksum_cooldown = checksum_cooldown

    def get_checksum_cooldown(self):
        return self.checksum_cooldown

    def set_upper_thresh1(self, upper_thresh1):
        self.upper_thresh1 = upper_thresh1

    def get_upper_thresh1(self):
        return self.upper_thresh1

    def set_upper_thresh2(self, upper_thresh2):
        self.upper_thresh2 = upper_thresh2

    def get_upper_thresh2(self):
        return self.upper_thresh2

    def set_grad_sample_interval(self, grad_sample_interval):
        self.filter_interval = grad_sample_interval

    def get_grad_sample_interval(self):
        return self.filter_interval

    @property
    def lock(self):
        if self._lock is None:
            self._lock = threading.Lock()
        return self._lock

    def init_stream(self):
        if self.statistic_cpu_value is None:
            self.statistic_value = torch.tensor(0., device=f"npu:{torch_npu.npu.current_device()}")
            self.checksum_state = 0
            self.statistic_cpu_value = torch.zeros((self.queue_len,), device='cpu', dtype=torch.float32).pin_memory()
            self.statistic_cpu_value.fill_(-1)
        if self.store is None:
            if torch.distributed.is_initialized():
                self.store = torch.distributed.distributed_c10d._get_default_store()
                self.rank = torch.distributed.get_rank()
                if self.rank == 0:
                    for i in range(1, torch.distributed.get_world_size()):
                        self.store.set(f"rank_{i}_info_log", "")
                        self.store.set(f"rank_{i}_warn_log", "")

    def parameter_filtering(self):
        self.filter_index = (self.filter_index + 1) % self.filter_interval
        return self.filter_index == 0
    
    def register_module_hook(self, module, name):
        self.check_stat[name + "_backward"] = {'avg': 0, 'pre_val': 0, 'step': 0, 'none_zero_step': 0}
        hook = partial(self.module_hook, name=name + "_backward")
        self.hook_dict[name + "_backward"] = module.register_full_backward_hook(hook)
        self.registered_modules.append(name)
    
    def module_hook(self, module, grad_input, grad_output, name):
        for _, param in module.named_parameters():
            if param.dim() >= 2:
                if param.grad is not None:
                    self._detect_grad(param.grad.detach(), name)
                    self.invalid_grad_sum = 0
                elif hasattr(param, 'main_grad') and param.main_grad is not None:
                    self._detect_grad(param.main_grad.detach(), name)
                    self.invalid_grad_sum = 0
                else:
                    self.invalid_grad_sum += 1
                    if self.invalid_grad_sum > max(10, len(self.registered_modules)):
                        warnings.warn(f"There is no available grad for detection, and the silent check feature may not take effect.")
                        self.invalid_grad_sum = 0

    def _detect_grad(self, grad, name):
        if grad.dtype != torch.bfloat16 and grad.dtype != torch.float32:
            return

        if self.matmul_hook_enable >= 1:
            with torch.no_grad():
                self.statistic_value.fill_(torch.pow(torch.norm(grad, float('inf')), 2).detach().float())

                #Asynchronously copy the value to host
                self.lock.acquire()
                self.statistic_cpu_value[self.tail_index].copy_(self.statistic_value.data, non_blocking=True)
                self.name_list[self.tail_index] = name
                self.tail_index = (self.tail_index + 1) % self.queue_len
                self.lock.release()
            if self.tail_index == self.head_index:
                # The queue is full, synchronize to empty the queue
                torch_npu.npu.synchronize()

    def _async_detect(self):
        while self.check_thread_running:
            if hasattr(torch, "npu") and torch.npu.is_initialized() and torch.distributed.is_initialized():
                break
            time.sleep(10)
        if not self.check_thread_running:
            return
        local_rank = os.getenv("LOCAL_RANK", "-1")
        if local_rank.isdigit():
            torch.npu.set_device(int(local_rank))

        while self.check_thread_running:
            self.lock.acquire()
            val = self.statistic_cpu_value[self.head_index].item()
            name = self.name_list[self.head_index]
            while val != -1 and name != "":
                loggerSilent.debug(f"[silent data] name:{name}, val: {val}, pre_val: {self.check_stat[name]['pre_val']}, avg: {self.check_stat[name]['avg']}, bp time: {self.check_stat[name]['step']}, none_zero_step: {self.check_stat[name]['none_zero_step']}")
                result, self.check_stat[name]['avg'], self.check_stat[name]['none_zero_step'] = self._silent_check(
                    val, self.check_stat[name]['pre_val'], self.check_stat[name]['avg'], self.check_stat[name]['none_zero_step'],
                    self.upper_thresh1, self.upper_thresh2
                )

                if result:
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    new_abnormal = {'time_str': current_time,
                                    'time': time.time(),
                                    'name': name,
                                    'rank': self.rank,
                                    'val': val,
                                    'pre_val': self.check_stat[name]['pre_val'],
                                    'avg': self.check_stat[name]['avg'],
                                    'step': self.check_stat[name]['step'],
                                    'none_zero_step': self.check_stat[name]['none_zero_step'],
                                    'counted': True,
                                    'striked': False}
                    self._abnormal_process(new_abnormal)
                self.check_stat[name]['step'] += 1
                self.check_stat[name]['pre_val'] = val

                self.statistic_cpu_value[self.head_index].fill_(-1)
                self.name_list[self.head_index] = ""
                self.head_index = (self.head_index + 1) % self.queue_len
                val = self.statistic_cpu_value[self.head_index].item()
                name = self.name_list[self.head_index]

            self.lock.release()
            time.sleep(0.1)

    def _silent_check(self, val, pre_val, avg, none_zero_step, alpha1=1e6, alpha2=1e2):
        if val == 0:
            return False, avg, none_zero_step
        elif math.isnan(val) or math.isinf(val):
            return True, avg, none_zero_step
        else:
            if none_zero_step >= 10 and avg != 0:
                thres = avg * alpha1 / (1 - 0.99 ** none_zero_step)
                thres2 = avg * alpha2 / (1 - 0.99 ** none_zero_step)
            else:
                thres = val
                thres2 = val
            if val > thres and abs(val - pre_val) > thres:
                return True, avg, none_zero_step
            else:
                if val <= thres2:
                    none_zero_step += 1
                    avg = avg * 0.99 + val * 0.01
                return False, avg, none_zero_step

    def _abnormal_process(self, new_abnormal):
        counting_abnormal_pos = []
        i = len(self.history_abnormal_list) - 1
        if i < 0:
            self._generate_event_log(new_abnormal)
            self.history_abnormal_list.append(new_abnormal)
            if self.strikes_num == 1:
                self._generate_warning_log(counting_abnormal_pos, new_abnormal)
                new_abnormal['striked'] = True
                if self.with_checksum:
                    self.checksum_state = 1
                    if not self.matmul_with_bf16:
                        warnings.warn(f"Warning: Module has no supported dtype grad, checksum will not to be linked.")
            return
        while i >= 0:
            old_abnormal = self.history_abnormal_list[i]
            old_time = old_abnormal['time']
            new_time = new_abnormal['time']
            if old_abnormal['counted'] and abs(new_time - old_time) >= self.cooldown * 60:
                # A new counted abnormal
                self._generate_event_log(new_abnormal)
                if self.strikes_num == 1:
                    self._generate_warning_log(counting_abnormal_pos, new_abnormal)
                    new_abnormal['striked'] = True
                    if self.with_checksum:
                        self.checksum_state = 1
                        if not self.matmul_with_bf16:
                            warnings.warn(f"Warning: Module has no supported dtype grad, checksum will not to be linked.")
                    break
                counting_abnormal_pos.append(i)
                i -= 1
                while i >= 0:
                    old_abnormal = self.history_abnormal_list[i]
                    if old_abnormal['counted'] and not old_abnormal['striked']:
                        counting_abnormal_pos.append(i)
                    if len(counting_abnormal_pos) == self.strikes_num - 1:
                        break
                    i -= 1
                if len(counting_abnormal_pos) == self.strikes_num - 1 and abs(new_abnormal['time'] - old_abnormal['time']) <= self.strikes_window * 60:
                    # Three strikes
                    self._generate_warning_log(counting_abnormal_pos, new_abnormal)
                    for index in counting_abnormal_pos:
                        self.history_abnormal_list[index]['striked'] = True
                    new_abnormal['striked'] = True

                    if self.with_checksum:
                        self.checksum_state = 1
                        if not self.matmul_with_bf16:
                            warnings.warn(f"Warning: Module has no supported dtype grad, checksum will not to be linked.")
                break
            elif not old_abnormal['counted']:
                # Keep tracing the last counted abnormal
                i -= 1
            else:
                # A new not-counted abnormal
                new_abnormal['counted'] = False
                break
        self.history_abnormal_list.append(new_abnormal)
        # remove expired exception
        current_time = time.time()
        first_expired_index = 0
        for abnormal in self.history_abnormal_list:
            if abs(current_time - abnormal['time']) <= self.strikes_window * 60:
                break
            first_expired_index += 1
        if first_expired_index > 0:
            del self.history_abnormal_list[:first_expired_index]

    def _generate_event_log(self, new_abnormal):
        info_str = f"[Event][{new_abnormal['time_str']}] [Rank {new_abnormal['rank']}]: A grad-norm spike may happen, "
        info_str = info_str + f"param name {new_abnormal['name']}, abnormal value {new_abnormal['val']}, previous value {new_abnormal['pre_val']}, "
        info_str = info_str + f"history avg {new_abnormal['avg']}, bp time {new_abnormal['step']}, normal count {new_abnormal['none_zero_step']}."
        loggerSilent.info(info_str)
        if self.store is not None and self.rank is not None and self.rank != 0:
            current_log = self.store.get(f"rank_{self.rank}_info_log").decode()
            self.store.set(f"rank_{self.rank}_info_log", current_log + "\n" + info_str if current_log != "" else info_str)

    def _generate_warning_log(self, counting_abnormal_pos, new_abnormal):
        warning_str = f"[Warning][{new_abnormal['time_str']}] [Rank {new_abnormal['rank']}]: feature detection detects abnormal results!"
        index = 0
        for pos in reversed(counting_abnormal_pos):
            warning_str = warning_str + "\n" + f"Grad-norm spike: index {index}, time {self.history_abnormal_list[pos]['time_str']}, param name {self.history_abnormal_list[pos]['name']}, abnormal value {self.history_abnormal_list[pos]['val']}, previous value {self.history_abnormal_list[pos]['pre_val']}, "
            warning_str = warning_str + f"history avg {self.history_abnormal_list[pos]['avg']}, bp time {self.history_abnormal_list[pos]['step']}, normal count {self.history_abnormal_list[pos]['none_zero_step']}."
            index += 1
        warning_str = warning_str + "\n" + f"Grad-norm spike: index {index}, time {new_abnormal['time_str']}, param name {new_abnormal['name']}, abnormal value {new_abnormal['val']}, previous value {new_abnormal['pre_val']}, "
        warning_str = warning_str + f"history avg {new_abnormal['avg']}, bp time {new_abnormal['step']}, normal count {new_abnormal['none_zero_step']}."
        loggerSilent.warning(warning_str)
        if self.store is not None and self.rank is not None and self.rank != 0:
            current_log = self.store.get(f"rank_{self.rank}_warn_log").decode()
            self.store.set(f"rank_{self.rank}_warn_log", current_log + "\n" + warning_str if current_log != "" else warning_str)

    def _generate_silent_log(self):
        warning_str = f"[Warning][Rank {self.rank}]: The result of Matmul checksum is abnormal!"
        loggerSilent.warning(warning_str)
        if self.store is not None and self.rank is not None and self.rank != 0:
            current_log = self.store.get(f"rank_{self.rank}_warn_log").decode()
            self.store.set(f"rank_{self.rank}_warn_log", current_log + "\n" + warning_str if current_log != "" else warning_str)

    def _tcp_comm_checksum_state(self):
        while self.checksum_state_thread_running:
            if hasattr(torch, "npu") and torch.npu.is_initialized() and torch.distributed.is_initialized() and self.store is not None:
                break
            time.sleep(10)
        if not self.checksum_state_thread_running:
            return
        local_rank = os.getenv("LOCAL_RANK", "-1")
        self.rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        if local_rank.isdigit():
            torch.npu.set_device(int(local_rank))

        last_checksum_time = None
        if self.rank == 0:
            self.store.add('counter2', world_size)
        while self.checksum_state_thread_running:
            if self.rank == 0:
                for i in range(1, world_size):
                    msg = self.store.get(f"rank_{i}_warn_log").decode()
                    if msg != "":
                        loggerSilent.warning(msg)
                        self.store.set(f"rank_{i}_warn_log", "")
                    msg = self.store.get(f"rank_{i}_info_log").decode()
                    if msg != "":
                        loggerSilent.info(msg)
                        self.store.set(f"rank_{i}_info_log", "")

            if not self.with_checksum or not self.matmul_with_bf16:
                time.sleep(10)
                continue

            self.store.add('checksum_state', self.checksum_state)
            if self.rank == 0:
                self.store.add('counter2', 0 - world_size)
            self.store.add('counter', 1)

            while int(self.store.get('counter').decode()) < world_size and self.checksum_state_thread_running:
                time.sleep(0.1)

            global_state = int(self.store.get('checksum_state').decode())
            if global_state:
                now_time = time.time()
                if last_checksum_time is None or abs(now_time - last_checksum_time) > self.checksum_cooldown * 60:
                    loggerSilent.info(f'[Info] Rank {self.rank}: feature detection detects abnormal results, checksum is on.')
                    last_checksum_time = now_time
                    if self.checksum_result is None:
                        self.checksum_result = torch.tensor(False, dtype=torch.bool, device='npu')
                    else:
                        self.checksum_result.fill_(False)
                    self.checksum_enable = True
                    time.sleep(self.cooldown * 60)
                    if self.checksum_result:
                        self._generate_silent_log()
                    self.checksum_enable = False
                    loggerSilent.info(f'[Info] Rank {self.rank}: checksum is off')
                self.checksum_state = 0
            self.store.add('counter2', 1)

            while int(self.store.get('counter2').decode()) < world_size and self.checksum_state_thread_running:
                time.sleep(0.1)
            
            if self.rank == 0:
                self.store.add('checksum_state', 0 - global_state)
                self.store.add('counter', 0 - world_size)

            time.sleep(10)

    def __getstate__(self):
        self._cleanup()
        state = self.__dict__.copy()
        state['_lock'] = None
        state['store'] = None
        return state
    
    def __setstate(self, state):
        self.__dict__.update(state)
        self.store = None

    def _startup(self):        
        if not self.check_thread_running:
            self.check_thread_running = True
            self.check_thread = threading.Thread(
                target=self._async_detect,
                daemon=True
            )
            self.check_thread.start()

        if not self.checksum_state_thread_running:
            self.checksum_state_thread_running = True
            self.checksum_state_thread = threading.Thread(
                target=self._tcp_comm_checksum_state,
                daemon=True
            )
            self.checksum_state_thread.start()

    def _cleanup(self):
        if self.check_thread_running:
            self.check_thread_running = False
            self.check_thread.join()
            self.check_thread = None

        if self.checksum_state_thread_running:
            self.checksum_state_thread_running = False
            self.checksum_state_thread.join()
            self.checksum_state_thread = None


matmul_check = _MatmulSilentCheck()


def _trigger_matmul_decorator(func):
    @wraps(func)
    def wrapper(a, b, *args, **kwargs):
        global matmul_check
        result = func(a, b, *args, **kwargs)
        if matmul_check.checksum_enable and a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16:
            checksum = torch_npu.matmul_checksum(a, b, result)
            matmul_check.checksum_result.logical_or_(checksum)
        return result
    return wrapper


def _trigger_tensor_matmul_decorator(func):
    @wraps(func)
    def wrapper(self, other):
        global matmul_check
        result = func(self, other)
        if matmul_check.checksum_enable and other.dtype == torch.bfloat16 and self.dtype == torch.bfloat16:
            checksum = torch_npu.matmul_checksum(self, other, result)
            matmul_check.checksum_result.logical_or_(checksum)
        return result
    return wrapper


def _matmul_silent_check_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        global matmul_check

        if not torch.npu.is_initialized():
            return func(self, *args, **kwargs)

        if matmul_check.get_matmul_hook_enable() and matmul_check.first_forward:
            matmul_check.init_stream()
            matmul_check.init_module_info(id(self), self.training)
            self.matmul_check_outer = True

            matmul_check._startup()
            if matmul_check.with_checksum and not matmul_check.matmul_trigger:
                original_matmul = torch.matmul
                original_tensor_matmul = torch.Tensor.matmul
                torch_npu.asd.checksum.matmul = original_matmul
                torch.matmul = _trigger_matmul_decorator(original_matmul)
                torch.Tensor.matmul = _trigger_tensor_matmul_decorator(original_tensor_matmul)
                matmul_check.matmul_trigger = True

            if matmul_check.is_training and not matmul_check.init_marks.get(matmul_check.first_module_id, False):
                for name, module in self.named_modules():
                    if matmul_check.get_matmul_hook_enable() == 0:
                        break
                    if len(module._modules) == 0 and name not in matmul_check.registered_modules and id(module) not in matmul_check.visited_modules_id:
                        matmul_check.visited_modules_id.append(id(module))
                        for _, param in module.named_parameters():
                            if not isinstance(param, torch.Tensor) or param.dim() < 2:
                                continue
                            if matmul_check.parameter_filtering():
                                matmul_check.register_module_hook(module, name)
                            # check dtype
                            if param.dtype == torch.float16:
                                for value in matmul_check.hook_dict.values():
                                    if value is not None:
                                        value.remove()
                                matmul_check.set_matmul_hook_enable(0)
                                break
                            if param.dtype == torch.bfloat16:
                                matmul_check.matmul_with_bf16 = True

                matmul_check.init_marks[matmul_check.first_module_id] = True

        tmp = func(self, *args, **kwargs)
        
        if matmul_check.get_matmul_hook_enable():
            if hasattr(self, "matmul_check_outer") and self.matmul_check_outer:
                matmul_check.init_param()
                self.matmul_check_outer = False

        return tmp
    return wrapper
