"""Generate a typed, fixed-ABI shim for an AutoFuse wrapper.so.

AutoFuse wrappers are generated C++ functions whose argument lists differ from
kernel to kernel.  A function pointer obtained from ``dlsym`` does not carry
the C type information needed to call such a function safely.  This module
reads the generated wrapper declaration during FX graph construction and
compiles a very small shim which performs the typed call.  The shim only
loads/calls ``wrapper.so``; it never calls the legacy adapter ABI or a kernel
symbol directly.
"""

from __future__ import annotations

import hashlib
import numbers
import os
import re
import shutil
import subprocess
import tempfile
import fcntl
from pathlib import Path
from typing import Iterable, Sequence


class AutofuseCodegenError(RuntimeError):
    """Raised when a wrapper signature cannot be safely adapted."""


_TYPE_ALIASES = {
    "void *": "void*",
    "void*": "void*",
    "const void *": "const void*",
    "const void*": "const void*",
    "char *": "char*",
    "char*": "char*",
    "const char *": "const char*",
    "const char*": "const char*",
    "int": "int",
    "int32_t": "int32_t",
    "int64_t": "int64_t",
    "uint32_t": "uint32_t",
    "uint64_t": "uint64_t",
    "bool": "bool",
    "float": "float",
    "double": "double",
}

_DECL_RE = re.compile(
    r"(?:extern\s+\"C\"\s+)?(?P<ret>[A-Za-z_][\w\s:*&<>]*)\s+"
    r"(?P<name>wrapper(?:_task_queue01|_task_queue2)?)\s*\(",
    re.MULTILINE,
)
_CODEGEN_VERSION = "fxrt-wrapper-shim-v4"


def _strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", source)


def _matching_paren(source: str, open_pos: int) -> int:
    """Return the closing parenthesis matching ``open_pos``."""
    depth = 0
    for pos in range(open_pos, len(source)):
        char = source[pos]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return pos
    raise AutofuseCodegenError("unterminated wrapper declaration")


def _split_params(params: str) -> list[str]:
    """Split a C++ parameter list while respecting nested delimiters."""
    result: list[str] = []
    start = 0
    depth = 0
    for pos, char in enumerate(params):
        if char in "(<[{":
            depth += 1
        elif char in ")>]}" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            result.append(params[start:pos].strip())
            start = pos + 1
    tail = params[start:].strip()
    if tail:
        result.append(tail)
    return result


def _normalize_type(param: str) -> str:
    """Normalize one generated wrapper parameter to a supported C type."""
    # Generated declarations have simple "type name" parameters.  Removing
    # the final identifier preserves pointer qualifiers in all supported forms.
    param = re.sub(r"\s*=.*$", "", param).strip()
    match = re.match(r"(?P<type>.+?)(?:\s+|)(?P<name>[A-Za-z_]\w*)$", param)
    if not match:
        raise AutofuseCodegenError(f"cannot parse wrapper parameter: {param!r}")
    type_text = " ".join(match.group("type").split())
    type_text = type_text.replace(" *", "*")
    if type_text not in _TYPE_ALIASES:
        raise AutofuseCodegenError(f"unsupported wrapper parameter type: {type_text!r}")
    return _TYPE_ALIASES[type_text]


def _find_wrapper_declaration(source: str) -> tuple[str, list[str]]:
    """Find and parse the exported AutoFuse wrapper declaration."""
    source = _strip_comments(source)
    matches = list(_DECL_RE.finditer(source))
    # Prefer the dispatching wrapper.  The task-queue entry points have the
    # same signature, and are only used as a fallback for older artifacts.
    matches.sort(key=lambda m: (m.group("name") != "wrapper", m.start()))
    if not matches:
        raise AutofuseCodegenError("wrapper source does not export a wrapper function")
    match = matches[0]
    close_pos = _matching_paren(source, match.end() - 1)
    params = _split_params(source[match.end() : close_pos])
    if not params or params == ["void"]:
        raise AutofuseCodegenError("wrapper must have kernel arguments, stream and kernel_key")
    return_type = " ".join(match.group("ret").split())
    if return_type not in {"int", "int64_t"}:
        raise AutofuseCodegenError(f"unsupported wrapper return type: {return_type!r}")
    return _TYPE_ALIASES.get(return_type, return_type), [_normalize_type(param) for param in params]


def _source_candidates(wrapper_path: str) -> Iterable[Path]:
    wrapper = Path(wrapper_path)
    yield wrapper.with_name("inductor_wrapper.cpp")
    yield wrapper.with_suffix(".cpp")


def read_wrapper_signature(wrapper_path: str) -> tuple[str, tuple[str, ...]]:
    """Read the generated wrapper declaration next to ``wrapper.so``."""
    for candidate in _source_candidates(wrapper_path):
        if candidate.is_file():
            return _find_wrapper_declaration(candidate.read_text(encoding="utf-8"))
    raise AutofuseCodegenError(
        f"cannot find generated wrapper source beside {wrapper_path}; "
        "a typed AutoFuse shim cannot be generated without its signature"
    )


def _cpp_type_for_payload(type_name: str, index: int) -> str:
    """Render the typed C++ expression for one fixed-ABI argument payload."""
    payload = f"args[{index}]"
    if type_name in {"void*", "const void*", "char*", "const char*"}:
        return f"reinterpret_cast<{type_name}>({payload})"
    if type_name == "bool":
        return f"static_cast<bool>({payload} != 0)"
    if type_name == "float":
        return f"static_cast<float>(fxrt_decode_double({payload}))"
    if type_name == "double":
        return f"fxrt_decode_double({payload})"
    return f"static_cast<{type_name}>({payload})"


def _render_stub(signature: Sequence[str], return_type: str) -> str:
    args = ",\n        ".join(
        f"{_cpp_type_for_payload(type_name, index)}" for index, type_name in enumerate(signature[:-2])
    )
    # The last two wrapper parameters are supplied by the fixed shim ABI.
    if args:
        args += ",\n        stream,\n        const_cast<char *>(kernel_key)"
    else:
        args = "stream,\n        const_cast<char *>(kernel_key)"
    return f'''#include <cstdint>\n#include <cstdlib>\n#include <cstring>\n#include <dlfcn.h>\n#include <map>\n#include <mutex>\n#include <string>\n\nusing InitFunc = int (*)(const char*, const char*);\nusing LaunchFunc = {return_type} (*)({", ".join(signature)});\nusing ReleaseFunc = int (*)(char*);\n\nstruct FxrtAutofuseState {{\n  void* handle;\n  LaunchFunc launch;\n  ReleaseFunc release;\n  std::string state_key;\n  std::string kernel_key;\n  uint32_t refs;\n}};\nstruct FxrtAutofuseContext {{\n  FxrtAutofuseState* state;\n}};\n\nstatic std::mutex g_fxrt_autofuse_mutex;\nstatic std::map<std::string, FxrtAutofuseState*> g_fxrt_autofuse_states;\n\nextern "C" uint32_t fxrt_autofuse_arg_num() {{ return {len(signature) - 2}U; }}\nextern "C" uint32_t fxrt_autofuse_abi_version() {{ return 1U; }}\n\nstatic float fxrt_decode_float(uint64_t payload) {{\n  float value; std::memcpy(&value, &payload, sizeof(value)); return value;\n}}\nstatic double fxrt_decode_double(uint64_t payload) {{\n  double value; std::memcpy(&value, &payload, sizeof(value)); return value;\n}}\n\nextern "C" int64_t fxrt_autofuse_init(\n    void** out_context, const char* wrapper_so, const char* kernel_so, const char* kernel_key) {{\n  if (out_context == nullptr || wrapper_so == nullptr || kernel_so == nullptr) return -1;\n  const std::string key = std::string(wrapper_so) + "\\n" + kernel_so + "\\n" +\n                          (kernel_key == nullptr ? "" : kernel_key);\n  const std::string selected_key = kernel_key == nullptr ? "" : kernel_key;\n  std::lock_guard<std::mutex> lock(g_fxrt_autofuse_mutex);\n  const auto found = g_fxrt_autofuse_states.find(key);\n  if (found != g_fxrt_autofuse_states.end()) {{\n    ++found->second->refs;\n    *out_context = new FxrtAutofuseContext{{found->second}};\n    return 0;\n  }}\n\n  void* handle = dlopen(wrapper_so, RTLD_NOW | RTLD_LOCAL);\n  if (handle == nullptr) return -2;\n  auto init = reinterpret_cast<InitFunc>(dlsym(handle, "init"));\n  auto launch = reinterpret_cast<LaunchFunc>(dlsym(handle, "wrapper"));\n  if (launch == nullptr) {{\n    const char* symbol = (std::getenv("TASK_QUEUE_ENABLE") != nullptr &&\n                          std::strtol(std::getenv("TASK_QUEUE_ENABLE"), nullptr, 10) == 2)\n                             ? "wrapper_task_queue2" : "wrapper_task_queue01";\n    launch = reinterpret_cast<LaunchFunc>(dlsym(handle, symbol));\n  }}\n  auto release = reinterpret_cast<ReleaseFunc>(dlsym(handle, "release"));\n  if (init == nullptr || launch == nullptr) {{ dlclose(handle); return -3; }}\n  const int ret = init(kernel_so, selected_key.empty() ? nullptr : selected_key.c_str());\n  if (ret != 0) {{ dlclose(handle); return ret; }}\n  auto* state = new FxrtAutofuseState{{handle, launch, release, key, selected_key, 1U}};\n  g_fxrt_autofuse_states.emplace(key, state);\n  *out_context = new FxrtAutofuseContext{{state}};\n  return 0;\n}}\n\nextern "C" int64_t fxrt_autofuse_launch(\n    void* opaque, const uint64_t* args, uint32_t arg_num, void* stream, const char* kernel_key) {{\n  if (opaque == nullptr || args == nullptr || arg_num != {len(signature) - 2}U) return -1;\n  auto* context = static_cast<FxrtAutofuseContext*>(opaque);\n  if (context->state == nullptr || context->state->launch == nullptr) return -2;\n  auto launch = context->state->launch;\n  return static_cast<int64_t>(launch(\n        {args}));\n}}\n\nextern "C" int64_t fxrt_autofuse_finalize(void* opaque, const char* kernel_key) {{\n  if (opaque == nullptr) return 0;\n  auto* context = static_cast<FxrtAutofuseContext*>(opaque);\n  if (context->state == nullptr) {{ delete context; return 0; }}\n  std::lock_guard<std::mutex> lock(g_fxrt_autofuse_mutex);\n  auto* state = context->state;\n  if (state->refs > 1U) {{\n    --state->refs;\n    delete context;\n    return 0;\n  }}\n  g_fxrt_autofuse_states.erase(state->state_key);\n  int64_t ret = 0;\n  // Defer release until all contexts using this wrapper have gone away.\n  // Some generated wrappers close every static kernel except the selected key.\n  if (g_fxrt_autofuse_states.empty() && state->release != nullptr && !state->kernel_key.empty()) {{\n    ret = state->release(const_cast<char*>(state->kernel_key.c_str()));\n  }}\n  dlclose(state->handle);\n  delete state;\n  delete context;\n  return ret;\n}}\n'''


def _validate_signature(signature: Sequence[str]) -> None:
    if len(signature) < 2:
        raise AutofuseCodegenError("wrapper signature must end with stream and kernel_key")
    if signature[-2] not in {"void*", "const void*"}:
        raise AutofuseCodegenError(f"wrapper stream parameter must be void*, got {signature[-2]!r}")
    if signature[-1] not in {"char*", "const char*"}:
        raise AutofuseCodegenError(f"wrapper kernel_key parameter must be char*, got {signature[-1]!r}")


def generate_wrapper_stub(wrapper_path: str, cache_dir: str | None = None) -> tuple[str, tuple[str, ...]]:
    """Generate and cache a typed wrapper shim; return ``(path, arg_types)``."""
    wrapper_path = os.path.realpath(wrapper_path)
    if not os.path.isfile(wrapper_path):
        raise AutofuseCodegenError(f"wrapper.so does not exist: {wrapper_path}")
    source_path = next((p for p in _source_candidates(wrapper_path) if p.is_file()), None)
    if source_path is None:
        raise AutofuseCodegenError(f"wrapper source is missing for {wrapper_path}")
    return_type, full_signature = read_wrapper_signature(wrapper_path)
    _validate_signature(full_signature)
    arg_types = tuple(full_signature[:-2])
    digest = hashlib.sha256()
    for path in (Path(wrapper_path), source_path):
        stat = path.stat()
        digest.update(str(path).encode())
        digest.update(str(stat.st_mtime_ns).encode())
        digest.update(str(stat.st_size).encode())
    digest.update(repr((return_type, full_signature)).encode())
    digest.update(_CODEGEN_VERSION.encode())
    key = digest.hexdigest()[:32]
    cache_value = cache_dir or os.environ.get("FXRT_AUTOFUSE_CODEGEN_CACHE")
    root = Path(cache_value) if cache_value else Path(tempfile.gettempdir()) / "fxrt_autofuse_codegen"
    target_dir = root / key
    target = target_dir / "stub.so"
    if target.is_file():
        return str(target), arg_types
    target_dir.mkdir(parents=True, exist_ok=True)
    with (target_dir / "stub.lock").open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if target.is_file():
            return str(target), arg_types
        source = target_dir / "stub.cpp"
        generated_source = _render_stub(full_signature, return_type)
        generated_source = generated_source.replace(
            "  void* handle = dlopen(wrapper_so, RTLD_NOW | RTLD_LOCAL);\n",
            "  int dlopen_flags = RTLD_NOW | RTLD_LOCAL;\n"
            "#ifdef RTLD_NODELETE\n"
            "  // AutoFuse's Python launcher can own the same wrapper, and torch-npu may\n"
            "  // still have callbacks whose code lives in it when an FXRT graph dies.\n"
            "  // Keep wrapper text/static state alive for the process lifetime.\n"
            "  dlopen_flags |= RTLD_NODELETE;\n"
            "#endif\n"
            "  void* handle = dlopen(wrapper_so, dlopen_flags);\n",
        )
        source.write_text(generated_source, encoding="utf-8")
        compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++")
        if compiler is None:
            raise AutofuseCodegenError("cannot generate AutoFuse wrapper shim: g++/c++ is unavailable")
        tmp_target = target_dir / f"stub.so.tmp.{os.getpid()}"
        command = [compiler, "-shared", "-fPIC", "-std=c++17", "-O2", str(source), "-ldl", "-o", str(tmp_target)]
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True)
        except OSError as exc:
            raise AutofuseCodegenError(f"failed to invoke {compiler}: {exc}") from exc
        if result.returncode != 0:
            raise AutofuseCodegenError(
                f"failed to compile AutoFuse wrapper shim for {wrapper_path}: {result.stderr.strip()}"
            )
        os.replace(tmp_target, target)
    return str(target), arg_types


def validate_wrapper_arguments(arg_types: Sequence[str], arguments: Sequence[object]) -> None:
    """Validate the FX example values against the generated C parameter kinds."""
    if len(arg_types) != len(arguments):
        raise AutofuseCodegenError(
            f"wrapper expects {len(arg_types)} kernel arguments, got {len(arguments)}"
        )
    for index, (type_name, value) in enumerate(zip(arg_types, arguments)):
        type_name = type_name.replace("const ", "")
        value_type = type(value)
        type_name_hint = value_type.__name__
        module_hint = value_type.__module__
        is_tensor = hasattr(value, "data_ptr") and hasattr(value, "shape")
        is_symbol = type_name_hint.startswith("Sym") or module_hint.startswith("sympy")
        if type_name == "void*":
            valid = is_tensor or value is None
        elif type_name in {"int", "int32_t", "int64_t", "uint32_t", "uint64_t"}:
            valid = isinstance(value, numbers.Integral) or is_symbol
        elif type_name == "bool":
            valid = isinstance(value, bool) or type_name_hint == "SymBool"
        elif type_name in {"float", "double"}:
            valid = (isinstance(value, numbers.Real) and not isinstance(value, bool)) or type_name_hint == "SymFloat"
        else:
            valid = False
        if not valid:
            raise AutofuseCodegenError(
                f"wrapper argument {index} has C type {type_name!r}, "
                f"but FX value has type {value_type!r}"
            )


__all__ = [
    "AutofuseCodegenError",
    "generate_wrapper_stub",
    "read_wrapper_signature",
    "validate_wrapper_arguments",
]
