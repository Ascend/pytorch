# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""The on-disk record format shared by NPU compiler cache artifacts.

An identity is written next to an artifact so that a later process can tell
whether the artifact is still the one it needs. This module owns only the
*format*: canonical serialization, hashing, and the ``Manifest`` envelope.

Nothing here is DVM-specific, and nothing here decides whether a saved compile
package may be reused -- that is decided by the environment alone (see
:mod:`.system_info`). The DVM producer identity that fills these values lives in
``torch_npu._inductor.dvm.persistence.identity``.
"""
import dataclasses
import hashlib
import json
import math
from typing import Any

# Initial, unreleased format. A backend scope check is applied by the owning
# backend's callers; identity checks still reject incompatible artifacts.
SCHEMA = 1


class PackageCompatibilityError(RuntimeError):
    """An unusable cache record with a reason; never an indication that execution is safe."""


class IdentityUnavailable(PackageCompatibilityError):
    pass


def canonical(value: Any):
    """Never use object addresses, Python hash(), or arbitrary repr() in a key."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: canonical(v) for k, v in sorted(value.items())}
        # Accuracy tables legitimately use torch.dtype keys.
        return {"mapping": sorted([[canonical(k), canonical(v)] for k, v in value.items()], key=json_bytes)}
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return {"set": sorted((canonical(v) for v in value), key=json_bytes)}
    # Configs can contain torch dtypes / operators and cacheable custom passes.
    import torch

    if isinstance(value, (torch.dtype, torch.device)):
        return {type(value).__name__: str(value)}
    if isinstance(value, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
        return {"operator": str(value)}
    if isinstance(value, torch._ops.HigherOrderOperator):
        return {"higher_order_operator": value.name()}
    uuid = getattr(value, "uuid", None)
    if callable(uuid):
        identifier = uuid()
        if identifier is None:
            raise IdentityUnavailable(f"NPU_CONFIG_UUID_UNAVAILABLE:{type(value).__qualname__}")
        return {"pass_uuid": canonical(identifier)}
    raise IdentityUnavailable(f"NPU_CONFIG_UNSERIALIZABLE:{type(value).__qualname__}")


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def stable_hash(value):
    return hashlib.sha256(json_bytes(canonical(value))).hexdigest()


@dataclasses.dataclass(frozen=True)
class Manifest:
    """The identity a cache artifact carries.

    ``request`` describes how the artifact was produced (backend, codegen knobs,
    initialization state, target ordinals). It keys lower caches and scopes restore
    to the right backend; it is not compared against the loading process.
    ``identity`` describes the compiler installation and is what makes an artifact
    from another chip or CANN release unusable.
    """
    schema: int
    request_json: str
    identity_json: str

    @classmethod
    def create(cls, request, identity):
        return cls(SCHEMA, json_bytes(canonical(request)).decode(),
                   json_bytes(canonical(identity)).decode())

    @property
    def request(self):
        return json.loads(self.request_json)

    @property
    def identity(self):
        return json.loads(self.identity_json)

    def check_supported(self, backend):
        """Format and scope gate: can these bytes be interpreted by ``backend``?"""
        if self.schema != SCHEMA:
            raise PackageCompatibilityError("NPU_MANIFEST_UNSUPPORTED")
        if self.request.get("backend") != backend:
            raise PackageCompatibilityError("NPU_BACKEND_OUT_OF_SCOPE")
