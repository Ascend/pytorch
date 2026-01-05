import torch
import torch_npu


torch_npu._afd.attention_worker_scheduler_ = torch.ops.npu.attention_worker_scheduler_
torch_npu._afd.attention_worker_scheduler = torch.ops.npu.attention_worker_scheduler
torch_npu._afd.ffn_worker_scheduler_ = torch.ops.npu.ffn_worker_scheduler_
torch_npu._afd.ffn_worker_scheduler = torch.ops.npu.ffn_worker_scheduler