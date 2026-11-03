import hashlib
import os
import re
import shutil
import warnings
from dataclasses import dataclass


_CXX_SUFFIXES = frozenset({".cc", ".cpp", ".cxx"})
_ASC_SUFFIXES = frozenset({".asc"})
_SUPPORTED_SUFFIXES = _CXX_SUFFIXES | _ASC_SUFFIXES
_NPU_ARCH_ENV_VARS = ("TORCH_NPU_ARCH_LIST", "TORCH_NPU_ARCH")
_BISHENG_OWNED_FLAGS = ("-c", "-dc", "-o", "-shared", "--npu-arch")


@dataclass(frozen=True)
class NpuSource:
    source: str
    object: str
    language: str


@dataclass(frozen=True)
class NpuBuildSpec:
    name: str
    build_directory: str
    output_path: str
    sources: tuple
    cxx: str
    bisheng: str
    npu_arch: str
    cxx_flags: tuple
    asc_flags: tuple
    dlink_flags: tuple
    link_flags: tuple
    extra_link_inputs: tuple
    depends: tuple
    npu_dlink: bool


def source_language(source):
    suffix = os.path.splitext(os.fspath(source))[1]
    normalized_suffix = suffix.lower()
    if normalized_suffix in _CXX_SUFFIXES:
        return "cxx"
    if normalized_suffix in _ASC_SUFFIXES:
        return "asc"
    supported = ", ".join(sorted(_SUPPORTED_SUFFIXES))
    raise ValueError(f"Unsupported extension source suffix {suffix!r}; supported: {supported}")


def contains_asc_sources(sources):
    return any(os.path.splitext(os.fspath(source))[1].lower() in _ASC_SUFFIXES
               for source in sources)


def validate_source_set(extension_name, sources, require_files):
    if not sources:
        raise ValueError(f"NpuExtension {extension_name!r} requires at least one source")

    physical_sources = set()
    for source in sources:
        source = os.fspath(source)
        if "\x00" in source or "\n" in source or "\r" in source:
            raise ValueError(
                f"NpuExtension {extension_name!r} has an invalid source path: {source!r}"
            )
        try:
            source_language(source)
        except ValueError as error:
            raise ValueError(
                f"NpuExtension {extension_name!r} source {source!r}: {error}"
            ) from error

        physical_source = os.path.normcase(os.path.realpath(os.path.abspath(source)))
        if physical_source in physical_sources:
            raise ValueError(
                f"NpuExtension {extension_name!r} contains duplicate source {source!r}"
            )
        physical_sources.add(physical_source)
        if require_files and not os.path.isfile(physical_source):
            raise ValueError(
                f"NpuExtension {extension_name!r} source is not a file: {source!r}"
            )


def normalize_npu_arch(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("npu_arch must be a string or None")
    value = value.strip()
    if not value:
        raise ValueError("npu_arch must not be empty")
    if any(character.isspace() for character in value) or "," in value or ";" in value:
        raise ValueError("npu_arch must contain exactly one architecture")
    return value


def architecture_from_device_name(device_name):
    normalized = str(device_name).lower()
    if "950" in normalized:
        return "dav-3510"
    if "910" in normalized:
        return "dav-2201"
    raise ValueError(
        f"Unsupported NPU device {device_name!r}; pass npu_arch explicitly"
    )


def resolve_npu_arch(explicit=None, environ=None, device_name=None):
    explicit = normalize_npu_arch(explicit)
    if explicit is not None:
        return explicit

    environ = os.environ if environ is None else environ
    for variable in _NPU_ARCH_ENV_VARS:
        raw_value = environ.get(variable)
        if not raw_value:
            continue
        values = [value for value in re.split(r"[;,\s]+", raw_value.strip()) if value]
        if len(values) != 1:
            raise ValueError(
                f"{variable} must contain exactly one architecture for the MVP"
            )
        architecture = normalize_npu_arch(values[0])
        if variable == "TORCH_NPU_ARCH_LIST":
            fallback_architecture = environ.get("TORCH_NPU_ARCH")
            if fallback_architecture:
                warnings.warn(
                    f"Both TORCH_NPU_ARCH_LIST={raw_value!r} and "
                    f"TORCH_NPU_ARCH={fallback_architecture!r} are set; using "
                    f"TORCH_NPU_ARCH_LIST={architecture!r} and ignoring "
                    "TORCH_NPU_ARCH",
                    stacklevel=2,
                )
        return architecture

    if device_name is not None:
        return architecture_from_device_name(device_name)
    raise RuntimeError(
        "Unable to determine NPU architecture; pass npu_arch or set TORCH_NPU_ARCH"
    )


def find_bisheng(explicit=None, environ=None):
    environ = os.environ if environ is None else environ
    candidates = [
        ("Explicit Bisheng path", explicit, True),
        ("BISHENG path", environ.get("BISHENG"), True),
        ("PATH Bisheng", shutil.which("bisheng"), False),
    ]
    for source, candidate, warn_on_failure in candidates:
        if not candidate:
            continue
        path = os.path.abspath(os.path.expanduser(candidate))
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
        if warn_on_failure:
            warnings.warn(
                f"{source} {path!r} was not found or is not executable; "
                "continuing with fallback Bisheng discovery",
                stacklevel=2,
            )
    raise RuntimeError(
        "Unable to find an executable Bisheng compiler; set BISHENG or initialize CANN"
    )


def object_path(source, build_directory):
    absolute_source = os.path.realpath(os.path.abspath(os.fspath(source)))
    digest = hashlib.sha256(os.fsencode(os.path.normcase(absolute_source))).hexdigest()[:12]
    basename = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(absolute_source))
    return os.path.join(os.path.abspath(build_directory), "obj", f"{basename}.{digest}.o")


def make_npu_sources(sources, build_directory):
    return tuple(
        NpuSource(
            source=os.path.realpath(os.path.abspath(os.fspath(source))),
            object=object_path(source, build_directory),
            language=source_language(source),
        )
        for source in sources
    )


def validate_bisheng_user_flags(flags):
    for flag in flags:
        if flag in _BISHENG_OWNED_FLAGS or flag.startswith("--npu-arch="):
            raise ValueError(
                f"Bisheng flag {flag!r} is owned by BuildExtension and cannot be overridden"
            )


def split_compile_args(extra_compile_args):
    if extra_compile_args is None:
        return [], []
    if isinstance(extra_compile_args, dict):
        copied = {key: list(value) for key, value in extra_compile_args.items()}
        dlink_flags = copied.pop("bisheng_dlink", [])
        validate_bisheng_user_flags(copied.get("bisheng", []))
        validate_bisheng_user_flags(dlink_flags)
        return copied, dlink_flags
    if isinstance(extra_compile_args, (list, tuple)):
        copied = list(extra_compile_args)
        validate_bisheng_user_flags(copied)
        return copied, []
    raise TypeError("extra_compile_args must be a list, tuple, dict, or None")
