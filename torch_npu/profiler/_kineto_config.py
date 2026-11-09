# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
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

import json

from torch._C._profiler import _ExperimentalConfig as ExperimentalConfig

from .experimental_config import supported_export_type, supported_profiler_level


class _NPUExperimentalConfig(ExperimentalConfig):
    """Ascend-specific options for ``torch.profiler.profile``.

    Only ``backend="aclprof"`` selects the ACLprof stub. ``profiler_level``
    and ``export_type`` are retained for future implementations; they do not
    enable collection or export. ``None`` leaves these options unspecified.
    Exports accept an ExportType string or a nonempty list of those strings.
    Invalid levels/exports raise ValueError; invalid backends fall back to MSPTI.
    """

    def __init__(self, *, backend: str = "mspti", profiler_level=None, export_type=None, **kwargs):
        config = {"backend": backend}
        if profiler_level is not None:
            if not isinstance(profiler_level, str) or profiler_level not in supported_profiler_level():
                raise ValueError("profiler_level must be a supported ProfilerLevel value")
            config["profiler_level"] = profiler_level
        if export_type is not None:
            exports = [export_type] if isinstance(export_type, str) else export_type
            if not isinstance(exports, list) or not exports or any(
                not isinstance(value, str) or value not in supported_export_type() for value in exports
            ):
                raise ValueError("export_type must be an ExportType value or a nonempty list of them")
            config["export_type"] = sorted(set(exports))
        super().__init__(
            custom_profiler_config=json.dumps(config, separators=(",", ":")),
            **kwargs,
        )

    def __str__(self):
        # Older PyTorch builds expose this field only through their pickle state.
        custom_config = getattr(self, "custom_profiler_config", None)
        if custom_config is not None:
            return custom_config
        return self.__getstate__()[11]
