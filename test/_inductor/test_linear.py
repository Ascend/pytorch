import torch
import torch_npu


def main(squeeze):
    clone = torch.ops.aten.clone.default(squeeze, memory_format=torch.contiguous_format)
    select = torch.ops.aten.select.int(clone, 0, 0)
    select_1 = torch.ops.aten.select.int(clone, 0, 1)
    select_2 = torch.ops.aten.select.int(clone, 0, 2)
    view_6 = torch.ops.aten.view.default(select, [128, 2400, 16])
    permute_2 = torch.ops.aten.permute.default(view_6, [1, 0, 2])
    view_7 = torch.ops.aten.view.default(select_1, [128, 2400, 16])
    permute_3 = torch.ops.aten.permute.default(view_7, [1, 0, 2])
    view_8 = torch.ops.aten.view.default(select_2, [128, 2400, 16])
    permute_4 = torch.ops.aten.permute.default(view_8, [1, 0, 2])
    mul_1 = torch.ops.aten.mul.Tensor(permute_2, 0.25)
    unsqueeze_default_21 = torch.ops.aten.unsqueeze.default(mul_1, 0)
    unsqueeze_default_22 = torch.ops.aten.unsqueeze.default(permute_3, 0)
    unsqueeze_default_23 = torch.ops.aten.unsqueeze.default(permute_4, 0)
    return (unsqueeze_default_21, unsqueeze_default_22, unsqueeze_default_23)

if __name__ == "__main__":
    squeeze = torch.randn((3, 128, 300, 128), device='npu', dtype=torch.float32)
    view_6 = torch.randn((128, 2400, 16), device='npu', dtype=torch.float32)
    view_7 = torch.randn((128, 2400, 16), device='npu', dtype=torch.float32)
    view_8 = torch.randn((128, 2400, 16), device='npu', dtype=torch.float32)
    arg5_1 = torch.randn((128, 128), device='npu', dtype=torch.float32)
    arg6_1 = torch.randn((128), device='npu', dtype=torch.float32)
    arg8_1 = torch.randn((384, 128), device='npu', dtype=torch.float32)
    arg7_1 = torch.randn((384), device='npu', dtype=torch.float32)
    func = torch.compile(main, backend='inductor', dynamic=False)
    func(squeeze)