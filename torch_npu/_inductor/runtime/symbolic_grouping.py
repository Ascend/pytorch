from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Literal

import sympy
import torch


JsonScalar = int | float | bool | str | None


@dataclasses.dataclass(frozen=True, slots=True)
class GroupFeatureSpec:
    name: str
    source: Literal["primary_axis", "outer_product", "reduction_product", "axis"]
    axis_names: tuple[str, ...]
    buckets: tuple[int, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class KernelVariant:
    variant_id: str
    constexpr_kwargs: tuple[tuple[str, int], ...]
    num_warps: int
    num_stages: int
    extra_compile_kwargs: tuple[tuple[str, JsonScalar], ...]


@dataclasses.dataclass(frozen=True, slots=True)
class LaunchPolicy:
    policy_id: str
    group_id: int
    static_blocks: tuple[tuple[str, int], ...]
    runtime_block_rules: tuple[tuple[str, tuple[tuple[str, JsonScalar], ...]], ...]
    grid_target: int


@dataclasses.dataclass(frozen=True, slots=True)
class GroupedCandidate:
    group_id: int
    variant_id: str
    policy_id: str


class UnsupportedGroupedPlan(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True, slots=True)
class GroupedKernelMeta:
    enabled: bool
    template: str
    workload: str | None
    primary_group_axis: str | None
    static_split_axes: tuple[str, ...]
    secondary_runtime_symbolic_axes: tuple[str, ...]
    group_features: tuple[GroupFeatureSpec, ...]
    runtime_block_arg_names: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "template": self.template,
            "workload": self.workload,
            "primary_group_axis": self.primary_group_axis,
            "static_split_axes": list(self.static_split_axes),
            "secondary_runtime_symbolic_axes": list(
                self.secondary_runtime_symbolic_axes
            ),
            "group_features": [
                {
                    "name": spec.name,
                    "source": spec.source,
                    "axis_names": list(spec.axis_names),
                    "buckets": list(spec.buckets),
                }
                for spec in self.group_features
            ],
            "runtime_block_arg_names": list(self.runtime_block_arg_names),
        }

    @staticmethod
    def from_payload(payload: Mapping[str, object]) -> "GroupedKernelMeta":
        return GroupedKernelMeta(
            enabled=_require_bool(payload.get("enabled"), "enabled"),
            template=_require_str(payload.get("template"), "template"),
            workload=_require_group_workload(payload.get("workload")),
            primary_group_axis=_require_optional_str(
                payload.get("primary_group_axis"), "primary_group_axis"
            ),
            static_split_axes=_require_str_tuple(
                payload.get("static_split_axes"), "static_split_axes"
            ),
            secondary_runtime_symbolic_axes=_require_str_tuple(
                payload.get(
                    "secondary_runtime_symbolic_axes"
                ),
                "secondary_runtime_symbolic_axes",
            ),
            group_features=tuple(
                _group_feature_spec_from_payload(item)
                for item in _require_sequence(
                    payload.get("group_features"), "group_features"
                )
            ),
            runtime_block_arg_names=_require_str_tuple(
                payload.get("runtime_block_arg_names"), "runtime_block_arg_names"
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class GroupedBenchmarkArgFootprint:
    group_id: int
    name: str
    kind: str
    dtype: str
    size: tuple[int, ...]
    stride: tuple[int, ...]
    num_bytes: int


@dataclasses.dataclass(frozen=True, slots=True)
class GroupedBenchmarkFootprint:
    synthetic_bytes: int
    mutated_clone_bytes: int
    total_bytes: int
    group_bytes: tuple[tuple[int, int], ...]
    largest_group_id: int | None
    dominant_arg: GroupedBenchmarkArgFootprint | None


def is_runtime_symbolic_length(length) -> bool:
    return not isinstance(length, (int, sympy.Integer))


def bucketize(value: int, buckets: tuple[int, ...]) -> int:
    for idx, upper in enumerate(buckets):
        if value <= upper:
            return idx
    return len(buckets)


def make_group_id(
    feature_values: tuple[int, ...], feature_specs: tuple[GroupFeatureSpec, ...]
) -> int:
    if len(feature_values) != len(feature_specs):
        raise ValueError(
            "feature_values and feature_specs must have the same length: "
            f"{len(feature_values)} != {len(feature_specs)}"
        )

    group_id = 0
    stride = 1
    for value, spec in zip(feature_values, feature_specs):
        group_id += bucketize(value, spec.buckets) * stride
        stride *= len(spec.buckets) + 1
    return group_id


def decode_group_bucket_indices(
    group_features, group_id: int
) -> tuple[int, ...]:
    bucket_indices = []
    remaining = int(group_id)
    for feature_spec in tuple(group_features):
        buckets = tuple(_feature_field(feature_spec, "buckets"))
        radix = len(buckets) + 1
        bucket_indices.append(remaining % radix)
        remaining //= radix
    return tuple(bucket_indices)


def find_primary_feature_index(group_features, primary_group_axis: str) -> int:
    for feature_idx, feature_spec in enumerate(tuple(group_features)):
        axis_names = tuple(_feature_field(feature_spec, "axis_names"))
        if primary_group_axis in axis_names:
            return feature_idx
    raise ValueError(
        f"primary_group_axis {primary_group_axis} does not appear in group_features"
    )


def is_open_bucket_group(
    group_features,
    primary_feature_index: int,
    group_id: int,
) -> bool:
    feature_specs = tuple(group_features)
    bucket_indices = decode_group_bucket_indices(feature_specs, group_id)
    buckets = tuple(_feature_field(feature_specs[primary_feature_index], "buckets"))
    return bucket_indices[primary_feature_index] == len(buckets)


def _feature_field(feature_spec, field_name: str):
    if isinstance(feature_spec, Mapping):
        return feature_spec[field_name]
    return getattr(feature_spec, field_name)


def build_group_representatives(
    group_features,
    axis_names,
    axis_static_values,
) -> dict[str, object]:
    group_features = tuple(group_features)
    axis_names = tuple(axis_names)
    axis_static_values = {
        axis_name: int(axis_value)
        for axis_name, axis_value in axis_static_values
    }

    seen_feature_axes = set()
    for feature_spec in group_features:
        for axis_name in tuple(_feature_field(feature_spec, "axis_names")):
            if axis_name in seen_feature_axes:
                raise UnsupportedGroupedPlan(
                    f"axis {axis_name} appears in multiple group features"
                )
            seen_feature_axes.add(axis_name)

    def bucket_bounds(feature_spec, bucket_idx: int):
        buckets = tuple(_feature_field(feature_spec, "buckets"))
        if not buckets:
            raise UnsupportedGroupedPlan(
                f"group feature {_feature_field(feature_spec, 'name')} must define non-empty buckets"
            )
        lower = int(buckets[bucket_idx - 1]) if bucket_idx > 0 else 0
        upper = int(buckets[bucket_idx]) if bucket_idx < len(buckets) else None
        return lower, upper

    def representative_feature_value(feature_spec, bucket_idx: int) -> int:
        lower, upper = bucket_bounds(feature_spec, bucket_idx)
        if upper is None:
            return lower * 2
        return upper

    def choose_symbolic_axis_value(static_factor: int, bucket_idx: int, feature_spec):
        lower, upper = bucket_bounds(feature_spec, bucket_idx)
        dyn_min = lower // static_factor + 1
        if upper is None:
            target = max(lower + 1, lower * 2)
            return max(1, (target + static_factor - 1) // static_factor)
        dyn_max = upper // static_factor
        if dyn_min > dyn_max:
            return None
        return max(1, dyn_max)

    def representative_for_feature(feature_spec, bucket_idx: int):
        feature_axis_names = tuple(_feature_field(feature_spec, "axis_names"))
        source = _feature_field(feature_spec, "source")
        if not feature_axis_names:
            return representative_feature_value(feature_spec, bucket_idx), ()
        if source in ("primary_axis", "axis"):
            if len(feature_axis_names) != 1:
                raise UnsupportedGroupedPlan(
                    f"group feature {_feature_field(feature_spec, 'name')} expects one axis"
                )
            value = representative_feature_value(feature_spec, bucket_idx)
            return value, ((feature_axis_names[0], value),)
        if source not in ("outer_product", "reduction_product"):
            raise UnsupportedGroupedPlan(
                f"group feature source {source} is not supported"
            )

        static_factor = 1
        static_axis_values = []
        symbolic_axis_names = []
        for axis_name in feature_axis_names:
            if axis_name in axis_static_values:
                axis_value = axis_static_values[axis_name]
                if axis_value <= 0:
                    raise UnsupportedGroupedPlan(
                        f"axis {axis_name} has non-positive static value {axis_value}"
                    )
                static_factor *= axis_value
                static_axis_values.append((axis_name, axis_value))
            else:
                symbolic_axis_names.append(axis_name)

        if len(symbolic_axis_names) > 1:
            raise UnsupportedGroupedPlan(
                f"group feature {_feature_field(feature_spec, 'name')} has multiple symbolic axes"
            )
        if len(symbolic_axis_names) == 0:
            feature_value = static_factor
            lower, upper = bucket_bounds(feature_spec, bucket_idx)
            if feature_value <= lower or (upper is not None and feature_value > upper):
                return None
            return feature_value, tuple(static_axis_values)

        symbolic_axis_value = choose_symbolic_axis_value(
            static_factor, bucket_idx, feature_spec
        )
        if symbolic_axis_value is None:
            return None
        axis_values = [
            (axis_name, symbolic_axis_value)
            if axis_name == symbolic_axis_names[0]
            else (axis_name, axis_static_values[axis_name])
            for axis_name in feature_axis_names
        ]
        return symbolic_axis_value * static_factor, tuple(axis_values)

    def representative_for_group(group_id: int):
        feature_values = []
        axis_values = {}
        remaining = group_id
        for feature_spec in group_features:
            buckets = tuple(_feature_field(feature_spec, "buckets"))
            radix = len(buckets) + 1
            bucket_idx = remaining % radix
            remaining //= radix
            representative = representative_for_feature(feature_spec, bucket_idx)
            if representative is None:
                return None
            feature_value, feature_axis_values = representative
            feature_values.append(feature_value)
            for axis_name, axis_value in feature_axis_values:
                axis_values[axis_name] = int(axis_value)

        for axis_name in axis_names:
            if axis_name in axis_static_values:
                axis_values.setdefault(axis_name, axis_static_values[axis_name])
            else:
                axis_values.setdefault(axis_name, 1)

        return (
            tuple(feature_values),
            tuple((axis_name, int(axis_values[axis_name])) for axis_name in axis_names),
        )

    group_id_count = 1
    for feature_spec in group_features:
        group_id_count *= len(tuple(_feature_field(feature_spec, "buckets"))) + 1

    representatives = tuple(
        representative_for_group(group_id) for group_id in range(group_id_count)
    )
    reachable_group_ids = tuple(
        group_id
        for group_id, representative in enumerate(representatives)
        if representative is not None
    )
    if not reachable_group_ids:
        raise UnsupportedGroupedPlan("grouped plan has no reachable groups")

    plan = {
        "group_id_count": group_id_count,
        "reachable_group_ids": reachable_group_ids,
        "unreachable_group_ids": tuple(
            group_id
            for group_id, representative in enumerate(representatives)
            if representative is None
        ),
        "benchmark_feature_inputs_by_group": tuple(
            representative[0] if representative is not None else ()
            for representative in representatives
        ),
        "benchmark_axis_values_by_group": tuple(
            representative[1] if representative is not None else ()
            for representative in representatives
        ),
    }
    return plan


def evaluate_grouped_benchmark_expr(expr, axis_env) -> int:
    if isinstance(expr, int):
        return int(expr)
    if not isinstance(expr, Mapping):
        raise UnsupportedGroupedPlan(
            f"unsupported grouped benchmark expression: {expr}"
        )
    if "const" in expr:
        return int(expr["const"])
    if "axis_name" in expr:
        axis_name = expr["axis_name"]
        if axis_name not in axis_env:
            raise UnsupportedGroupedPlan(
                f"benchmark axis environment is missing {axis_name}"
            )
        return int(axis_env[axis_name])
    if "runtime_arg_index" in expr:
        raise UnsupportedGroupedPlan(
            "runtime_arg_index cannot be bounded during grouped codegen"
        )
    if "mul" in expr:
        product = 1
        for operand in expr["mul"]:
            product *= evaluate_grouped_benchmark_expr(operand, axis_env)
        return product
    if "add" in expr:
        return sum(
            evaluate_grouped_benchmark_expr(operand, axis_env)
            for operand in expr["add"]
        )
    if "floordiv" in expr:
        operands = tuple(
            evaluate_grouped_benchmark_expr(operand, axis_env)
            for operand in expr["floordiv"]
        )
        if len(operands) != 2 or operands[1] == 0:
            raise UnsupportedGroupedPlan(
                f"invalid grouped benchmark floordiv expression: {expr}"
            )
        return operands[0] // operands[1]
    raise UnsupportedGroupedPlan(
        f"unsupported grouped benchmark expression: {expr}"
    )


def required_storage_numel(size, stride) -> int:
    size = tuple(int(dim) for dim in size)
    stride = tuple(int(dim_stride) for dim_stride in stride)
    if len(size) != len(stride):
        raise UnsupportedGroupedPlan(
            f"benchmark size/stride rank mismatch: {size} vs {stride}"
        )
    if any(dim < 0 for dim in size):
        raise UnsupportedGroupedPlan(f"negative benchmark tensor size: {size}")
    if any(dim_stride < 0 for dim_stride in stride):
        raise UnsupportedGroupedPlan(
            f"negative benchmark tensor stride: {stride}"
        )
    if any(dim == 0 for dim in size):
        return 0
    return 1 + sum(
        (dim - 1) * dim_stride
        for dim, dim_stride in zip(size, stride)
    )


def _grouped_benchmark_dtype_itemsize(dtype) -> tuple[str, int]:
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise UnsupportedGroupedPlan(f"unknown grouped benchmark dtype: {dtype}")
    return str(dtype), int(dtype.itemsize)


def estimate_grouped_benchmark_footprint(
    group_representatives,
    ordered_arg_specs,
    mutated_arg_names=(),
) -> GroupedBenchmarkFootprint:
    reachable = tuple(group_representatives["reachable_group_ids"])
    axis_values = tuple(
        group_representatives["benchmark_axis_values_by_group"]
    )
    ordered_arg_specs = tuple(ordered_arg_specs)
    mutated_names = set(mutated_arg_names)
    synthetic_bytes = 0
    mutated_clone_bytes = 0
    group_bytes = []
    dominant_arg = None

    for group_id in reachable:
        axis_env = dict(axis_values[group_id])
        current_bytes = 0
        current_mutated_bytes = 0
        for spec in ordered_arg_specs:
            kind = spec.get("kind")
            source = spec.get("source")
            if source == "runtime_arg" or kind == "size":
                continue
            if kind == "tensor" and source in ("buffer", "constant"):
                size = tuple(
                    evaluate_grouped_benchmark_expr(expr, axis_env)
                    for expr in spec["size_exprs"]
                )
                stride = tuple(
                    evaluate_grouped_benchmark_expr(expr, axis_env)
                    for expr in spec["stride_exprs"]
                )
                dtype_name, itemsize = _grouped_benchmark_dtype_itemsize(
                    spec["dtype"]
                )
                num_bytes = required_storage_numel(size, stride) * itemsize
            elif kind == "workspace" and source == "workspace":
                count = evaluate_grouped_benchmark_expr(
                    spec["count_expr"], axis_env
                )
                if count < 0:
                    raise UnsupportedGroupedPlan(
                        f"negative grouped benchmark workspace count: {count}"
                    )
                dtype_name, itemsize = _grouped_benchmark_dtype_itemsize(
                    spec["dtype"]
                )
                size, stride, num_bytes = (count,), (1,), count * itemsize
            else:
                raise UnsupportedGroupedPlan(
                    f"unsupported grouped benchmark arg: kind={kind}, source={source}"
                )

            current_bytes += num_bytes
            if spec.get("name") in mutated_names:
                current_mutated_bytes += num_bytes
            if dominant_arg is None or num_bytes > dominant_arg.num_bytes:
                dominant_arg = GroupedBenchmarkArgFootprint(
                    group_id=int(group_id),
                    name=str(spec.get("name", "unknown")),
                    kind=str(kind),
                    dtype=dtype_name,
                    size=size,
                    stride=stride,
                    num_bytes=int(num_bytes),
                )

        group_bytes.append((int(group_id), int(current_bytes)))
        synthetic_bytes += current_bytes
        mutated_clone_bytes = max(mutated_clone_bytes, current_mutated_bytes)

    largest_group_id = (
        max(group_bytes, key=lambda item: item[1])[0]
        if group_bytes
        else None
    )
    return GroupedBenchmarkFootprint(
        int(synthetic_bytes),
        int(mutated_clone_bytes),
        int(synthetic_bytes + mutated_clone_bytes),
        tuple(group_bytes),
        largest_group_id,
        dominant_arg,
    )


def serialize_grouped_plan(plan: GroupedKernelMeta) -> dict[str, object]:
    return plan.to_payload()


def deserialize_grouped_plan(payload: dict[str, object]) -> GroupedKernelMeta:
    return GroupedKernelMeta.from_payload(payload)


def _group_feature_spec_from_payload(payload: object) -> GroupFeatureSpec:
    mapping = _require_mapping(payload, "group_features[]")
    return GroupFeatureSpec(
        name=_require_str(mapping.get("name"), "group_features[].name"),
        source=_require_str(mapping.get("source"), "group_features[].source"),
        axis_names=_require_str_tuple(
            mapping.get("axis_names"), "group_features[].axis_names"
        ),
        buckets=_require_int_tuple(mapping.get("buckets"), "group_features[].buckets"),
    )


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping, got {type(value).__name__}")
    return value


def _require_sequence(value: object, field_name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a sequence, got {type(value).__name__}")
    return value


def _require_str(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    return value


def _require_optional_str(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, field_name)


def _require_group_workload(value: object) -> str | None:
    workload = _require_optional_str(value, "workload")
    if workload not in (None, "elementwise"):
        raise ValueError(f"unsupported group workload: {workload}")
    return workload


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool, got {type(value).__name__}")
    return value


def _require_str_tuple(value: object, field_name: str) -> tuple[str, ...]:
    items = _require_sequence(value, field_name)
    return tuple(_require_str(item, field_name) for item in items)


def _require_int_tuple(value: object, field_name: str) -> tuple[int, ...]:
    items = _require_sequence(value, field_name)
    result = []
    for item in items:
        if not isinstance(item, int):
            raise TypeError(
                f"{field_name} must contain ints, got {type(item).__name__}"
            )
        result.append(item)
    return tuple(result)
