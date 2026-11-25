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
import uuid

from torch._C._profiler import _ExperimentalConfig as ExperimentalConfig

from .experimental_config import _ExperimentalConfig, AiCMetrics, supported_export_type, supported_profiler_level


def _custom_profiler_config_state(config):
    error = "Incompatible PyTorch ExperimentalConfig: cannot reliably read custom_profiler_config"
    marker = f"__torch_npu_custom_config_{uuid.uuid4().hex}__"
    try:
        reference = ExperimentalConfig(custom_profiler_config=marker).__getstate__()
    except TypeError as cause:
        raise RuntimeError(error) from cause
    state = config.__getstate__()
    if (
        type(reference) not in (tuple, list)
        or type(state) is not type(reference)
        or list(map(type, state)) != list(map(type, reference))
    ):
        raise RuntimeError(error)
    positions = [i for i, value in enumerate(reference) if isinstance(value, str) and value == marker]
    if len(positions) != 1:
        raise RuntimeError(error)
    return state, positions[0]


def _custom_profiler_config(config):
    try:
        raw = config.custom_profiler_config
    except AttributeError:
        state, index = _custom_profiler_config_state(config)
        raw = state[index]
    if not isinstance(raw, str):
        raise RuntimeError("Incompatible PyTorch ExperimentalConfig: cannot reliably read custom_profiler_config")
    return raw


class _NPUExperimentalConfig(ExperimentalConfig):
    """Ascend options for ``torch.profiler.profile``.

    An NPU trace handler selects ACLprof and supplies its output directory.
    Without that handler the backend selection and native options are preserved.
    Device settings reuse the legacy configuration validation. ACLprof produces
    native files per active window and the handler can analyze them as text or DB.
    CPU/Python collection is separate.
    """

    _npu_config = True

    def __init__(self, *, backend: str = "mspti", profiler_level=None, export_type=None,
                 aic_metrics=AiCMetrics.PipeUtilization, l2_cache=False, mstx=False,
                 record_op_args=False, op_attr=False, host_sys=None,
                 mstx_domain_include=None, mstx_domain_exclude=None,
                 sys_io=False, sys_interconnection=False, **kwargs):
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
        self._device_options = dict(
            aic_metrics=aic_metrics, l2_cache=l2_cache, mstx=mstx,
            record_op_args=record_op_args, op_attr=op_attr, host_sys=host_sys,
            mstx_domain_include=mstx_domain_include, mstx_domain_exclude=mstx_domain_exclude,
            sys_io=sys_io, sys_interconnection=sys_interconnection,
        )
        super().__init__(
            custom_profiler_config=json.dumps(config, separators=(",", ":")),
            **kwargs,
        )

    def __reduce_ex__(self, protocol):
        return self._restore, (self.__getstate__(), {"_device_options": getattr(self, "_device_options", {})})

    @classmethod
    def _restore(cls, state, attributes):
        config = cls.__new__(cls)
        config.__setstate__(state)
        config._device_options = attributes.get("_device_options", {})
        return config

    @property
    def _backend(self):
        return json.loads(str(self)).get("backend")

    def _get_device_config(self, level, export_type=None):
        device = _ExperimentalConfig(profiler_level=level, export_type=export_type or "text",
                                     **getattr(self, "_device_options", {}))
        if device.record_op_args:
            raise ValueError("ACLprof native capture does not collect operator arguments")
        return {
            "aic_metrics": device._aic_metrics, "l2_cache": device._l2_cache,
            "mstx": device.tx_enabled, "record_op_args": device.record_op_args,
            "op_attr": device._op_attr, "host_sys": device._host_sys,
            "mstx_domain_include": device._mstx_domain_include,
            "mstx_domain_exclude": device._mstx_domain_exclude,
            "sys_io": device._sys_io, "sys_interconnection": device._sys_interconnection,
        }

    def to_custom_profiler_config(self, overrides=None):
        config = json.loads(_custom_profiler_config(self))
        config.update(overrides or {})
        config = {**self._get_device_config(config.get("profiler_level", "Level0"), config.get("export_type")), **config}
        carrier = ExperimentalConfig.__new__(ExperimentalConfig)
        state, index = _custom_profiler_config_state(self)
        values = list(state)
        values[index] = json.dumps(config, separators=(",", ":"))
        carrier.__setstate__(type(state)(values))
        return carrier

    def __str__(self):
        return _custom_profiler_config(self)
