import torch
import torch.nn as nn
import torch_npu


class LabelSmoothingCrossEntropy(nn.Module):
    """CrossEntropy with LabelSmoothing using npu api.

    Paper: [Rethinking the Inception Architecture for Computer Vision]

    Args:
        smooth_factor (float): default 0. If label_smoothing using, using 0.1([0, 1]) instead.
        num_classes (float): classes numbers using for onehot.

    Returns:
        float: tensors of shape (k, 5) and (k, 1). Labels are 0-based.
    """

    def __init__(self, num_classes=1000, smooth_factor=0.):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.on_value = 1.0 - smooth_factor
        self.off_value = 1.0 * smooth_factor / (num_classes - 1)

    def forward(self, pred, target):
        one_hot_label = torch_npu.npu_one_hot(target.int(), -1, pred.size(1), self.on_value, self.off_value)
        loss = torch_npu.npu_softmax_cross_entropy_with_logits(pred, one_hot_label)

        loss = torch.mean(loss, [0], keepdim=False, dtype=torch.float32)
        return loss