import os
import torch
import torch_npu


def parse_thresh(env_var_name, default_value, min_value):
    env_var = os.environ.get(env_var_name, default=default_value)
    thresh = [value.strip() for value in env_var.split(",")]
    if len(thresh) != 2 or not all(value.isdigit() for value in thresh):
        thresh = default_value.split(",")
    thresh = [max(int(value), min_value) for value in thresh]
    if thresh[0] <= thresh[1]:
        thresh = [int(value) for value in default_value.split(",")]
        
    return thresh


def get_thresh():
    upper_thresh = parse_thresh("NPU_ASD_UPPER_THRESH", "1000000,10000", 3)
    sigma_thresh = parse_thresh("NPU_ASD_SIGMA_THRESH", "100000,5000", 3)
    return upper_thresh, sigma_thresh


class SilentFaultData:
    def __init__(self):
        self.pre_val = torch.tensor(0).float().npu()
        self.max_val = torch.tensor(-10 ** 10).float().npu()
        self.min_val = torch.tensor(10 ** 10).float().npu()
        self.upper_thresh, self.sigma_thresh = get_thresh()


class SilentFaultDataV2:
    def __init__(self):
        self.step_tensor = torch.zeros(1, dtype=torch.int64).npu()
        self.check_tensor = torch.zeros(3, dtype=torch.float).npu()
        self.upper_thresh, self.sigma_thresh = get_thresh()
