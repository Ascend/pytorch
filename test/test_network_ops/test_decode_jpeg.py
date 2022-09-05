# Copyright (c) 2022, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from struct import pack, unpack_from
import base64

import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

SAFEBLOCK = 1024 * 1024


def i8(c):
    return c if c.__class__ is int else c[0]


def i16(c, o=0):
    return unpack_from(">H", c, o)[0]


def _safe_read(fp, size):
    if size <= 0:
        return b""
    if size <= SAFEBLOCK:
        data = fp.read(size)
        if len(data) < size:
            raise OSError("Truncated File Read")
        return data
    data = []
    remaining_size = size
    while remaining_size > 0:
        block = fp.read(min(remaining_size, SAFEBLOCK))
        if not block:
            break
        data.append(block)
        remaining_size -= len(block)
    if sum(len(d) for d in data) < size:
        raise OSError("Truncated File Read")
    return b"".join(data)


def skip(fp):
    n = i16(fp.read(2)) - 2
    _safe_read(fp, n)


def app(fp):
    n = i16(fp.read(2)) - 2
    s = _safe_read(fp, n)


def com(fp):
    n = i16(fp.read(2)) - 2
    s = _safe_read(fp, n)


def sof(fp):
    n = i16(fp.read(2)) - 2
    s = _safe_read(fp, n)
    h, w, c = i16(s, 1), i16(s, 3), i8(s[5])
    return h, w, c


def dqt(fp):
    n = i16(fp.read(2)) - 2
    s = _safe_read(fp, n)


MARKER = {
    0xFFC0: ("SOF0", "Baseline DCT", sof),
    0xFFC1: ("SOF1", "Extended Sequential DCT", sof),
    0xFFC2: ("SOF2", "Progressive DCT", sof),
    0xFFC3: ("SOF3", "Spatial lossless", sof),
    0xFFC4: ("DHT", "Define Huffman table", skip),
    0xFFC5: ("SOF5", "Differential sequential DCT", sof),
    0xFFC6: ("SOF6", "Differential progressive DCT", sof),
    0xFFC7: ("SOF7", "Differential spatial", sof),
    0xFFC8: ("JPG", "Extension", None),
    0xFFC9: ("SOF9", "Extended sequential DCT (AC)", sof),
    0xFFCA: ("SOF10", "Progressive DCT (AC)", sof),
    0xFFCB: ("SOF11", "Spatial lossless DCT (AC)", sof),
    0xFFCC: ("DAC", "Define arithmetic coding conditioning", skip),
    0xFFCD: ("SOF13", "Differential sequential DCT (AC)", sof),
    0xFFCE: ("SOF14", "Differential progressive DCT (AC)", sof),
    0xFFCF: ("SOF15", "Differential spatial (AC)", sof),
    0xFFD0: ("RST0", "Restart 0", None),
    0xFFD1: ("RST1", "Restart 1", None),
    0xFFD2: ("RST2", "Restart 2", None),
    0xFFD3: ("RST3", "Restart 3", None),
    0xFFD4: ("RST4", "Restart 4", None),
    0xFFD5: ("RST5", "Restart 5", None),
    0xFFD6: ("RST6", "Restart 6", None),
    0xFFD7: ("RST7", "Restart 7", None),
    0xFFD8: ("SOI", "Start of image", None),
    0xFFD9: ("EOI", "End of image", None),
    0xFFDA: ("SOS", "Start of scan", skip),
    0xFFDB: ("DQT", "Define quantization table", dqt),
    0xFFDC: ("DNL", "Define number of lines", skip),
    0xFFDD: ("DRI", "Define restart interval", skip),
    0xFFDE: ("DHP", "Define hierarchical progression", sof),
    0xFFDF: ("EXP", "Expand reference component", skip),
    0xFFE0: ("APP0", "Application segment 0", app),
    0xFFE1: ("APP1", "Application segment 1", app),
    0xFFE2: ("APP2", "Application segment 2", app),
    0xFFE3: ("APP3", "Application segment 3", app),
    0xFFE4: ("APP4", "Application segment 4", app),
    0xFFE5: ("APP5", "Application segment 5", app),
    0xFFE6: ("APP6", "Application segment 6", app),
    0xFFE7: ("APP7", "Application segment 7", app),
    0xFFE8: ("APP8", "Application segment 8", app),
    0xFFE9: ("APP9", "Application segment 9", app),
    0xFFEA: ("APP10", "Application segment 10", app),
    0xFFEB: ("APP11", "Application segment 11", app),
    0xFFEC: ("APP12", "Application segment 12", app),
    0xFFED: ("APP13", "Application segment 13", app),
    0xFFEE: ("APP14", "Application segment 14", app),
    0xFFEF: ("APP15", "Application segment 15", app),
    0xFFF0: ("JPG0", "Extension 0", None),
    0xFFF1: ("JPG1", "Extension 1", None),
    0xFFF2: ("JPG2", "Extension 2", None),
    0xFFF3: ("JPG3", "Extension 3", None),
    0xFFF4: ("JPG4", "Extension 4", None),
    0xFFF5: ("JPG5", "Extension 5", None),
    0xFFF6: ("JPG6", "Extension 6", None),
    0xFFF7: ("JPG7", "Extension 7", None),
    0xFFF8: ("JPG8", "Extension 8", None),
    0xFFF9: ("JPG9", "Extension 9", None),
    0xFFFA: ("JPG10", "Extension 10", None),
    0xFFFB: ("JPG11", "Extension 11", None),
    0xFFFC: ("JPG12", "Extension 12", None),
    0xFFFD: ("JPG13", "Extension 13", None),
    0xFFFE: ("COM", "Comment", com),
}


def extract_jpeg_shpae(fp):
    s = fp.read(3)
    s = b"\xFF"

    while True:
        i = s[0]
        if i == 0xFF:
            s = s + fp.read(1)
            i = i16(s)
        else:
            s = fp.read(1)
            continue

        if i in MARKER:
            name, description, handler = MARKER[i]
            if handler is not None:
                if name[:3] == "SOF":
                    h, w, c = handler(fp)
                    break
                else:
                    handler(fp)
            s = fp.read(1)
        elif i == 0 or i == 0xFFFF:
            s = b"\xff"
        elif i == 0xFF00:
            s = fp.read(1)
        else:
            raise SyntaxError("no marker found")

    return h, w, c


class TestDecodeJpeg(TestCase):
    def result_error(self, img, image_shape, channels):
        error_name = "result error"
        if img.shape[0] != image_shape[0] or img.shape[1] != image_shape[1] or img.shape[2] != channels:
            self.fail("shape error")
        if img.dtype != torch.uint8:
            self.fail("dtype error")
        if img.device.type != "npu":
            self.fail("device error")

    def test_decode_jpeg(self, device="npu"):
        path = "../../docs/en/PyTorch Network Model Porting and Training Guide/figures/pth-file.jpg"
        
        with open(path, "rb") as f:
            f.seek(0)
            image_shape = extract_jpeg_shpae(f)

            f.seek(0)
            bytes_string = f.read()
            arr = np.frombuffer(bytes_string, dtype=np.uint8)
            addr = 16
            length = len(bytes_string)
            addr_arr = list(map(int, pack('<Q', addr)))
            len_arr = list(map(int, pack('<Q', length)))
            arr = np.hstack((addr_arr, len_arr, arr, [0]))
            arr = np.array(arr, dtype=np.uint8)
            bytes_tensor = torch.as_tensor(arr.copy()).npu()

            channels = 3
            output = torch_npu.decode_jpeg(bytes_tensor, image_shape, channels=channels)

            self.result_error(output, image_shape, channels)


if __name__ == '__main__':
    run_tests()
