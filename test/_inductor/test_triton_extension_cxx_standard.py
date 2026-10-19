import shutil
import subprocess
from unittest import mock

import pytest

from torch_npu._inductor.triton_compat import (
    _with_cxx20,
    patch_triton_extension_cxx_standard,
)


def _builder(command, *, kernel_launcher="torch"):
    return subprocess.run(command, capture_output=True, text=True)


def _precompile_builder(header_path, gch_path):
    return subprocess.run(
        [
            shutil.which("g++"), "-x", "c++-header", str(header_path),
            "-std=c++17", "-o", str(gch_path),
        ],
        capture_output=True,
        text=True,
    )


def test_precompile_patch_when_extension_builder_is_already_patched():
    from triton.backends.ascend import driver, utils

    existing_builder = _with_cxx20(_builder)
    with mock.patch.object(utils, "_build_npu_ext", existing_builder), mock.patch.object(
        utils, "_precompile_npu_ext", _precompile_builder, create=True
    ), mock.patch.object(
        driver, "_precompile_npu_ext", _precompile_builder, create=True
    ), mock.patch.object(subprocess, "run") as run:
        patch_triton_extension_cxx_standard()
        assert utils._build_npu_ext is existing_builder
        patched = utils._precompile_npu_ext
        driver_patched = driver._precompile_npu_ext
        assert getattr(patched, "_torch_npu_cxx20", False)
        assert getattr(driver_patched, "_torch_npu_cxx20", False)
        driver._precompile_npu_ext("precompiled.h", "precompiled.h.gch")
        assert "-std=c++20" in run.call_args.args[0]
        patch_triton_extension_cxx_standard()
        assert utils._precompile_npu_ext is patched
        assert driver._precompile_npu_ext is driver_patched


def test_precompile_patch_compiles_cxx20_header(tmp_path):
    from triton.backends.ascend import driver, utils

    if shutil.which("g++") is None:
        pytest.skip("g++ is required to compile the regression header")
    header = tmp_path / "precompiled.h"
    header.write_text(
        '#if __cplusplus < 202002L\n#error C++20 is required\n#endif\n'
        '#include <compare>\n'
        'struct Value { int x; auto operator<=>(const Value&) const = default; };\n'
    )
    gch = tmp_path / "precompiled.h.gch"
    # Negative control: reproduce the CI failure before applying compatibility.
    before = _precompile_builder(header, gch)
    assert before.returncode != 0
    assert "C++20 is required" in before.stderr
    with mock.patch.object(
        utils, "_precompile_npu_ext", _precompile_builder, create=True
    ), mock.patch.object(
        driver, "_precompile_npu_ext", _precompile_builder, create=True
    ):
        patch_triton_extension_cxx_standard()
        result = driver._precompile_npu_ext(header, gch)
        assert result.returncode == 0, result.stderr
        assert gch.is_file()


def test_precompile_patch_without_extension_builder():
    from triton.backends.ascend import driver, utils

    with mock.patch.object(utils, "_build_npu_ext", None), mock.patch.object(
        utils, "_precompile_npu_ext", _precompile_builder, create=True
    ), mock.patch.object(
        driver, "_precompile_npu_ext", _precompile_builder, create=True
    ), mock.patch.object(subprocess, "run") as run:
        patch_triton_extension_cxx_standard()
        driver._precompile_npu_ext("precompiled.h", "precompiled.h.gch")
        assert "-std=c++20" in run.call_args.args[0]


def test_patch_independent_driver_precompile_alias():
    from triton.backends.ascend import driver, utils

    driver_builder = _with_cxx20(_precompile_builder)
    utils_builder = _precompile_builder
    with mock.patch.object(utils, "_build_npu_ext", _builder), mock.patch.object(
        utils, "_precompile_npu_ext", utils_builder, create=True
    ), mock.patch.object(driver, "_precompile_npu_ext", driver_builder, create=True), mock.patch.object(
        subprocess, "run"
    ) as run:
        patch_triton_extension_cxx_standard()
        assert utils._precompile_npu_ext is not driver._precompile_npu_ext
        assert driver._precompile_npu_ext is driver_builder
        driver._precompile_npu_ext("precompiled.h", "precompiled.h.gch")
        assert "-std=c++20" in run.call_args.args[0]


def test_builder_standard_is_local():
    builder = _with_cxx20(_builder)
    command = ["g++", "-std=c++17", "-x", "c++-header", "precompiled.h"]
    with mock.patch.object(subprocess, "run") as run:
        builder(command)
        assert run.call_args.args[0][1] == "-std=c++20"
        assert command[1] == "-std=c++17"
        assert _builder.__globals__["subprocess"] is subprocess
        builder(command, kernel_launcher="mindspore")
        assert run.call_args.args[0][1] == "-std=c++17"
        builder(["g++", "-std=c++23", "launcher.cpp"])
        assert run.call_args.args[0][1] == "-std=c++23"


def test_registered_driver_uses_cxx20_without_initializing_runtime():
    from triton.backends import backends

    namespace = backends["ascend"].driver.__init__.__globals__
    with mock.patch.dict(namespace, {
        "_build_npu_ext": _builder,
        "_precompile_npu_ext": _precompile_builder,
    }), mock.patch.object(
        subprocess, "run"
    ) as run:
        patch_triton_extension_cxx_standard()
        builder = namespace["_build_npu_ext"]
        builder(["g++", "-std=c++17", "npu_utils.cpp"])
        assert "-std=c++20" in run.call_args.args[0]
        namespace["_precompile_npu_ext"]("precompiled.h", "precompiled.h.gch")
        assert "-std=c++20" in run.call_args.args[0]
        patch_triton_extension_cxx_standard()
        assert namespace["_build_npu_ext"] is builder


def test_patch_is_idempotent_for_driver_and_utils_builders():
    from triton.backends.ascend import driver, utils

    with mock.patch.object(utils, "_build_npu_ext", _builder), mock.patch.object(
        driver, "_build_npu_ext", _builder
    ):
        patch_triton_extension_cxx_standard()
        utils_patched = utils._build_npu_ext
        driver_patched = driver._build_npu_ext
        assert getattr(utils_patched, "_torch_npu_cxx20", False)
        assert getattr(driver_patched, "_torch_npu_cxx20", False)
        patch_triton_extension_cxx_standard()
        assert utils_patched is utils._build_npu_ext
        assert driver_patched is driver._build_npu_ext


def test_old_torch_does_not_patch():
    from triton.backends.ascend import utils

    with mock.patch("torch.__version__", "2.12.0"), mock.patch.object(
        utils, "_build_npu_ext", _builder
    ), mock.patch.object(
        utils, "_precompile_npu_ext", _precompile_builder, create=True
    ):
        patch_triton_extension_cxx_standard()
        assert utils._build_npu_ext is _builder
        assert utils._precompile_npu_ext is _precompile_builder
