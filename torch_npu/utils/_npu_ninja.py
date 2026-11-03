import os
import shlex
import sys

import torch.utils.cpp_extension as TorchExtension


def _reject_unsafe_text(value):
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"Ninja values must not contain NUL or newlines: {value!r}")


def _ninja_escape_path(value):
    value = os.fspath(value)
    _reject_unsafe_text(value)
    return value.replace("$", "$$").replace(" ", "$ ").replace(":", "$:")


def _shell_quote(value):
    value = os.fspath(value)
    _reject_unsafe_text(value)
    return shlex.quote(value)


def _ninja_shell_token(value):
    return _shell_quote(value).replace("$", "$$")


def _join_shell_tokens(values):
    return " ".join(_ninja_shell_token(value) for value in values)


def _render_ninja_file(spec):
    depfile_filter = os.path.join(os.path.dirname(__file__), "_npu_depfile.py")
    lines = [
        "ninja_required_version = 1.3",
        f"cxx = {_ninja_shell_token(spec.cxx)}",
        f"bisheng = {_ninja_shell_token(spec.bisheng)}",
        f"python = {_ninja_shell_token(sys.executable)}",
        f"depfile_filter = {_ninja_shell_token(depfile_filter)}",
        f"npu_arch = {_ninja_shell_token(spec.npu_arch)}",
        f"cxx_flags = {_join_shell_tokens(spec.cxx_flags)}",
        f"asc_flags = {_join_shell_tokens(spec.asc_flags)}",
        f"dlink_flags = {_join_shell_tokens(spec.dlink_flags)}",
        f"ldflags = {_join_shell_tokens(spec.link_flags)}",
        "",
        "rule cxx_compile",
        "  command = $cxx $cxx_flags -MMD -MF $out.d -c $in -o $out",
        "  depfile = $out.d",
        "  deps = gcc",
        "",
        "rule asc_compile",
        "  command = $bisheng -x asc -c --npu-arch=$npu_arch $asc_flags "
        "-MMD -MF $out.raw.d -MQ $out $in -o $out && "
        "$python $depfile_filter $out.raw.d $out.d $out",
        "  depfile = $out.d",
        "  deps = gcc",
        "",
        "rule asc_compile_rdc",
        "  command = $bisheng -x asc -dc --npu-arch=$npu_arch $asc_flags "
        "-MMD -MF $out.raw.d -MQ $out $in -o $out && "
        "$python $depfile_filter $out.raw.d $out.d $out",
        "  depfile = $out.d",
        "  deps = gcc",
        "",
        "rule npu_link",
        "  command = $bisheng -shared $in $dlink_flags $ldflags -o $out",
        "",
    ]

    for source in spec.sources:
        if source.language == "cxx":
            rule = "cxx_compile"
            dependencies = spec.depends
        else:
            rule = "asc_compile_rdc" if spec.npu_dlink else "asc_compile"
            dependencies = (*spec.depends, depfile_filter)
        implicit_dependencies = ""
        if dependencies:
            implicit_dependencies = " | " + " ".join(
                _ninja_escape_path(dependency) for dependency in dependencies
            )
        lines.append(
            f"build {_ninja_escape_path(source.object)}: {rule} "
            f"{_ninja_escape_path(source.source)}{implicit_dependencies}"
        )

    link_inputs = [source.object for source in spec.sources]
    link_inputs.extend(spec.extra_link_inputs)
    lines.extend([
        f"build {_ninja_escape_path(spec.output_path)}: npu_link "
        + " ".join(_ninja_escape_path(value) for value in link_inputs),
        "",
        f"default {_ninja_escape_path(spec.output_path)}",
        "",
    ])
    return "\n".join(lines)


def _write_ninja_file(spec):
    os.makedirs(os.path.join(spec.build_directory, "obj"), exist_ok=True)
    os.makedirs(os.path.dirname(spec.output_path), exist_ok=True)
    build_file = os.path.join(spec.build_directory, "build.ninja")
    TorchExtension._maybe_write(build_file, _render_ninja_file(spec))
    return build_file


def _run_ninja_build(spec, verbose=False):
    TorchExtension._run_ninja_build(
        spec.build_directory,
        verbose,
        f"Error compiling objects for extension {spec.name!r}",
    )
