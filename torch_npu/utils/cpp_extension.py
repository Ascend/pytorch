import copy
import os
import re
import subprocess
import sysconfig

import torch
import torch.utils.cpp_extension as TorchExtension
from setuptools._distutils.ccompiler import gen_preprocess_options

import torch_npu
from torch_npu.utils._npu_extension import (
    NpuBuildSpec,
    contains_asc_sources,
    find_bisheng,
    make_npu_sources,
    normalize_npu_arch,
    resolve_npu_arch,
    split_compile_args,
    validate_source_set,
)
import torch_npu.utils._npu_ninja as NpuNinja


__all__ = ["NpuExtension", "BuildExtension", "NpuBuildExtension"]


PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.realpath(torch_npu.__file__))


def NpuExtension(name, sources, *args, **kwargs):
    """Create a :class:`setuptools.Extension` for C++ and ASC sources.

    ``NpuExtension`` follows the setuptools/PyTorch extension API while adding
    NPU build metadata consumed by :class:`BuildExtension`.
    """
    npu_arch = kwargs.pop("npu_arch", None)
    npu_dlink = kwargs.pop("npu_dlink", False)
    copied_sources = copy.deepcopy(list(sources))
    copied_kwargs = copy.deepcopy(kwargs)
    has_asc_sources = contains_asc_sources(copied_sources)
    dlink_flags = []
    if has_asc_sources:
        if not isinstance(npu_dlink, bool):
            raise TypeError("npu_dlink must be a bool")
        validate_source_set(name, copied_sources, require_files=False)
        compile_args, dlink_flags = split_compile_args(
            copied_kwargs.get("extra_compile_args")
        )
        copied_kwargs["extra_compile_args"] = compile_args

    torch_npu_dir = PYTORCH_NPU_INSTALL_PATH
    include_dirs = list(copied_kwargs.get("include_dirs") or [])
    include_dirs += [
        os.path.join(torch_npu_dir, "include"),
        os.path.join(torch_npu_dir, "include", "third_party", "hccl", "inc"),
    ]
    if has_asc_sources:
        include_dirs.append(sysconfig.get_path("include"))
    copied_kwargs["include_dirs"] = include_dirs

    library_dirs = list(copied_kwargs.get("library_dirs") or [])
    library_dirs.append(os.path.join(torch_npu_dir, "lib"))
    copied_kwargs["library_dirs"] = library_dirs

    extension = TorchExtension.CppExtension(
        name, copied_sources, *args, **copied_kwargs
    )
    extension.libraries.append("torch_npu")
    extension._npu_arch = normalize_npu_arch(npu_arch) if has_asc_sources else None
    extension._npu_dlink = npu_dlink if has_asc_sources else False
    extension._npu_dlink_flags = dlink_flags
    extension._npu_has_asc = has_asc_sources
    return extension


class BuildExtension(TorchExtension.BuildExtension):
    """Build ASC sources declared with NpuExtension, or delegate C++ to PyTorch."""

    def build_extensions(self):
        asc_extensions = [
            ext for ext in self.extensions
            if contains_asc_sources(ext.sources)
        ]
        if asc_extensions:
            for extension in asc_extensions:
                if getattr(extension, "_npu_has_asc", False) is not True or any(
                    not hasattr(extension, name)
                    for name in ("_npu_arch", "_npu_dlink", "_npu_dlink_flags")
                ):
                    raise ValueError(
                        f"Extension {extension.name!r} contains .asc sources but lacks "
                        "NpuExtension ASC metadata. Create it with NpuExtension "
                        "and declare all ASC sources at construction time; use "
                        "torch_npu.utils.cpp_extension.BuildExtension to build it."
                    )
            if not TorchExtension.IS_LINUX:
                raise RuntimeError("ASC extensions are currently supported only on Linux")
            if not self.use_ninja:
                raise RuntimeError(
                    "Ninja is required to build NpuExtension modules containing .asc sources"
                )
            if not TorchExtension.is_ninja_available():
                raise RuntimeError("Ninja is required to build ASC extensions")
            device_name = None
            if any(getattr(extension, "_npu_arch", None) is None
                   for extension in asc_extensions):
                try:
                    if torch.npu.is_available():
                        device_name = torch.npu.get_device_name(0)
                except (AttributeError, RuntimeError):
                    device_name = None
            for extension in asc_extensions:
                validate_source_set(extension.name, extension.sources, require_files=True)
                extension._npu_arch = resolve_npu_arch(
                    getattr(extension, "_npu_arch", None), device_name=device_name
                )
                extension._npu_bisheng = find_bisheng()
                self._check_bisheng(extension._npu_bisheng)
            if ".asc" not in self.compiler.src_extensions:
                self.compiler.src_extensions.append(".asc")
        return super().build_extensions()

    @staticmethod
    def _check_bisheng(bisheng):
        try:
            subprocess.run(
                [bisheng, "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=False,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                f"Unable to execute Bisheng compiler at {bisheng!r}"
            ) from error

    def build_extension(self, ext):
        if not contains_asc_sources(ext.sources):
            return super().build_extension(ext)
        return self._build_npu_extension(ext)

    def _build_npu_extension(self, ext):
        output_path = os.path.abspath(self.get_ext_fullpath(ext.name))
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", ext.name)
        build_directory = os.path.abspath(
            os.path.join(self.build_temp, "npu", safe_name)
        )
        preprocess_flags = gen_preprocess_options(
            list(ext.define_macros or []) + [(name,) for name in (ext.undef_macros or [])],
            [os.path.abspath(path) for path in (ext.include_dirs or [])],
        )

        cxx_compiler_command = getattr(self.compiler, "compiler_cxx", None)
        if not cxx_compiler_command:
            cxx_compiler_command = self.compiler.compiler_so
        cxx = cxx_compiler_command[0]
        cxx_flags = list(cxx_compiler_command[1:])
        if "-fPIC" not in cxx_flags:
            cxx_flags.append("-fPIC")
        cxx_flags.extend(preprocess_flags)

        extra_compile_args = ext.extra_compile_args or []
        if isinstance(extra_compile_args, dict):
            user_cxx_flags = list(extra_compile_args.get("cxx", []))
            user_asc_flags = list(extra_compile_args.get("bisheng", []))
        else:
            user_cxx_flags = list(extra_compile_args)
            user_asc_flags = list(extra_compile_args)
        cxx_flags.extend(user_cxx_flags)
        asc_flags = ["-fPIC", *preprocess_flags, *user_asc_flags]
        if not any(flag.startswith("-std=") for flag in cxx_flags):
            cxx_flags.append("-std=c++17")
        if not any(flag.startswith("-std=") for flag in asc_flags):
            asc_flags.append("-std=c++17")

        extra_link_inputs = tuple(
            os.path.realpath(os.path.abspath(path))
            for path in (ext.extra_objects or [])
        )
        link_flags = []
        link_flags.extend(
            f"-L{os.path.abspath(path)}" for path in (ext.library_dirs or [])
        )
        link_flags.extend(
            f"-Wl,-rpath,{os.path.abspath(path)}"
            for path in (ext.runtime_library_dirs or [])
        )
        link_flags.extend(f"-l{library}" for library in (ext.libraries or []))
        link_flags.extend(ext.extra_link_args or [])
        depends = tuple(
            os.path.realpath(os.path.abspath(path)) for path in (ext.depends or [])
        )

        spec = NpuBuildSpec(
            name=ext.name,
            build_directory=build_directory,
            output_path=output_path,
            sources=make_npu_sources(ext.sources, build_directory),
            cxx=cxx,
            bisheng=ext._npu_bisheng,
            npu_arch=ext._npu_arch,
            cxx_flags=tuple(cxx_flags),
            asc_flags=tuple(asc_flags),
            dlink_flags=tuple(ext._npu_dlink_flags),
            link_flags=tuple(link_flags),
            extra_link_inputs=extra_link_inputs,
            depends=depends,
            npu_dlink=ext._npu_dlink,
        )
        NpuNinja._write_ninja_file(spec)
        NpuNinja._run_ninja_build(spec, verbose=self.verbose)
        if not os.path.isfile(output_path):
            raise RuntimeError(
                f"NPU extension build completed without output: {output_path}"
            )


NpuBuildExtension = BuildExtension
