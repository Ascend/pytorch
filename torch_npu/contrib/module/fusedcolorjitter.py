__all__ = []

import warnings
import random
from math import sin, cos, pi
import numbers
import numpy as np
import torch

from torch_npu.utils._error_code import ErrCode, ops_error

warnings.filterwarnings(action='once', category=FutureWarning)


class _FusedColorJitterApply(object):
    def __init__(self,
                 hue=0.0,
                 saturation=1.0,
                 contrast=0.0,
                 brightness=0.0,
                 is_normalized=False,
                 force_return_array=False):
        self.hue = hue
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness
        self.is_normalized = is_normalized
        self.force_return_array = force_return_array
        self.half_range = 127.5 if not is_normalized else 0.5

    def hue_saturation_matrix(self, hue, saturation):
        """
        Single matrix transform for both hue and saturation change.
        Derived by transforming first to YIQ, then do the modification, and transform back to RGB.
        """
        const_mat = np.array([[0.299, 0.299, 0.299],
                              [0.587, 0.587, 0.587],
                              [0.114, 0.114, 0.114],
                              ], dtype=np.float32)
        sch_mat = np.array([[0.701, -0.299, -0.300],
                            [-0.587, 0.413, -0.588],
                            [-0.114, -0.114, 0.886],
                            ], dtype=np.float32)
        ssh_mat = np.array([[0.168, -0.328, 1.250],
                            [0.330, 0.035, -1.050],
                            [-0.497, 0.292, -0.203],
                            ], dtype=np.float32)
        sch = saturation * cos(hue * 255. * pi / 180.0)
        ssh = saturation * sin(hue * 255. * pi / 180.0)
        m = const_mat + sch * sch_mat + ssh * ssh_mat
        return m

    def get_random_transform_matrix(self, hue=0.05, saturation=0.5, contrast=0.5, brightness=0.125):
        hue = random.uniform(-hue, hue)
        saturation = random.uniform(max(0, 1. - saturation), 1 + saturation)
        contrast = random.uniform(max(0, 1. - contrast), 1 + contrast)
        brightness = random.uniform(max(0, 1. - brightness), 1 + brightness)

        transform_matrix = self.hue_saturation_matrix(hue, saturation)
        transform_matrix = transform_matrix * brightness * contrast
        transform_offset = (1. - contrast) * brightness * self.half_range
        return transform_matrix, transform_offset

    def apply_image_transform(self, img, transform_matrix, transform_offset):
        H, W, C = img.shape
        if C != 3:
            if C == 4:
                img = img[:, :, :3]
            elif C == 1:
                img = img.repeat(3, axis=-1)
            else:
                raise ValueError('Unknow format using.. Currnet shape is {}'.format(img.shape) + 
                                 ops_error(ErrCode.VALUE))
            H, W, C = img.shape
        img = np.matmul(img.reshape(-1, 3), transform_matrix) + transform_offset
        return img.reshape(H, W, C)

    def __call__(self, img):
        from PIL import Image

        transform_matrix, transform_offset = self.get_random_transform_matrix(
            self.hue, self.saturation, self.contrast, self.brightness
        )

        if isinstance(img, Image.Image):
            img = np.asarray(img, dtype=np.float32)
            return_img = True
            self.raw_type = np.uint8
        else:
            self.raw_type = img.dtype
            return_img = False
        img = self.apply_image_transform(img, transform_matrix, transform_offset)

        img = img.clip(0., 1. if self.is_normalized else 255.).astype(self.raw_type)

        if return_img and not self.force_return_array:
            return Image.fromarray(img, mode='RGB')

        return img


class FusedColorJitter(torch.nn.Module):
    """Randomly change the brightness, contrast, saturation and hue of an image.

    Unlike the native torchvision.transforms.ColorJitter,
    FusedColorJitter completes the adjustment of the image's brightness, contrast, saturation, and hue,
    through a matmul and an add operation, approximately 20% performance acceleration can be achieved.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        warnings.warn("torch_npu.contrib.module.FusedColorJitter is deprecated. "
                      "Please use torchvision.transforms.ColorJitter for replacement.", FutureWarning)
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

        self.transformer = _FusedColorJitterApply(brightness, contrast, saturation, hue)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name) + 
                                 ops_error(ErrCode.VALUE))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound) + ops_error(ErrCode.VALUE))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name) + 
                            ops_error(ErrCode.TYPE))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def forward(self, img):
        return self.transformer(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string
