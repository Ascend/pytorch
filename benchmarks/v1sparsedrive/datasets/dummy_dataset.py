import torch
from torch.utils.data import Dataset


class DummySparseDriveDataset(Dataset):
    """
    模拟后续真实 dataloader 输出格式：
    {
        "inputs": Tensor[3, H, W],
        "data_samples": {
            "sample_idx": int,
            "img_shape": (H, W),
            "gt_det_boxes": Tensor[num_det_boxes, 11],
            "gt_det_labels": Tensor[num_det_boxes],
            "gt_map_points": Tensor[num_map_instances, 40],
            "gt_map_labels": Tensor[num_map_instances],
            "gt_motion": Tensor[24],
            "gt_plan": Tensor[12],
        }
    }
    """
    def __init__(
        self,
        length=20,
        image_size=(256, 704),
        num_det_boxes=20,
        num_map_instances=10,
        num_det_classes=10,
        num_map_classes=3,
    ):
        self.length = length
        self.image_size = image_size
        self.num_det_boxes = num_det_boxes
        self.num_map_instances = num_map_instances
        self.num_det_classes = num_det_classes
        self.num_map_classes = num_map_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        h, w = self.image_size

        image = torch.randn(3, h, w)

        gt_det_boxes = torch.randn(self.num_det_boxes, 11)
        gt_det_labels = torch.randint(0, self.num_det_classes, (self.num_det_boxes,))

        gt_map_points = torch.randn(self.num_map_instances, 40)
        gt_map_labels = torch.randint(0, self.num_map_classes, (self.num_map_instances,))

        gt_motion = torch.randn(24)
        gt_plan = torch.randn(12)

        sample = {
            "inputs": image,
            "data_samples": {
                "sample_idx": idx,
                "img_shape": (h, w),
                "gt_det_boxes": gt_det_boxes,
                "gt_det_labels": gt_det_labels,
                "gt_map_points": gt_map_points,
                "gt_map_labels": gt_map_labels,
                "gt_motion": gt_motion,
                "gt_plan": gt_plan,
            }
        }
        return sample


def simple_collate(batch):
    inputs = torch.stack([item["inputs"] for item in batch], dim=0)
    data_samples = [item["data_samples"] for item in batch]
    return {
        "inputs": inputs,
        "data_samples": data_samples,
    }