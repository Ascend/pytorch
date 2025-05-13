import importlib
import inspect
import json
import os
import re
import warnings
from typing import Callable
from itertools import chain
from pathlib import Path
import pkgutil

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch._utils_internal import get_file_path_2
import torch_npu


NOT_IMPORT_LIST = [
    "torch_npu.npu.BoolTensor",
    "torch_npu.npu.ByteTensor",
    "torch_npu.npu.CharTensor",
    "torch_npu.npu.DoubleTensor",
    "torch_npu.npu.FloatTensor",
    "torch_npu.npu.HalfTensor",
    "torch_npu.npu.IntTensor",
    "torch_npu.npu.LongTensor",
    "torch_npu.npu.ShortTensor",
    "torch_npu.npu.BFloat16Tensor",
    "torch_npu.distributed._verify_params_across_processes",
    "torch_npu.dynamo.torchair.ge.attr.Bool",
    "torch_npu.dynamo.torchair.ge.attr.DataType",
    "torch_npu.dynamo.torchair.ge.attr.Float",
    "torch_npu.dynamo.torchair.ge.attr.Int",
    "torch_npu.dynamo.torchair.ge.attr.ListBool",
    "torch_npu.dynamo.torchair.ge.attr.ListDataType",
    "torch_npu.dynamo.torchair.ge.attr.ListFloat",
    "torch_npu.dynamo.torchair.ge.attr.ListInt",
    "torch_npu.dynamo.torchair.ge.attr.ListListFloat",
    "torch_npu.dynamo.torchair.ge.attr.ListListInt",
    "torch_npu.dynamo.torchair.ge.attr.ListStr",
    "torch_npu.dynamo.torchair.ge.attr.Str",
    "torch_npu.profiler.ProfilerActivity"
]


def set_failure_list(api_str, value, signature, failure_list):
    failure_list.append(f"# {api_str}:")
    failure_list.append(f"  - function signature is different: ")
    failure_list.append(f"    - the base signature is {value}.")
    failure_list.append(f"    - now it is {signature}.")


def is_not_compatibility_for_cpp_api(base_signature: str, file: str):
    content = []
    with open(file, mode="r") as fp:
        subs = ""
        start_concat = False
        for line in fp.readlines():
            if "(" in line and ")" not in line and not subs:
                start_concat = True
            if start_concat:
                subs += line
            if ")" in line and "(" not in line and start_concat:
                start_concat = False
                subs = re.sub("(?<=\\()[ \n]+", "", subs)
                subs = re.sub("(?<=,)[ \n]+", " ", subs)
                line = subs
                subs = ""
            if not start_concat:
                content.append(line)
    text = "".join(content)

    base_signature = re.escape(base_signature)
    if not re.search(rf" +{base_signature}", text):
        return True
    else:
        return False


def is_not_compatibility(base_str, new_str, api_str=None):
    base_io_params = base_str.split("->")
    new_io_params = new_str.split("->")
    base_input_params = base_io_params[0].strip()
    new_input_params = new_io_params[0].strip()
    base_out_params = "" if len(base_io_params) == 1 else base_io_params[1].strip()
    new_out_params = "" if len(new_io_params) == 1 else new_io_params[1].strip()

    # output params
    if base_out_params != new_out_params:
        return True

    base_params = base_input_params[1:-1].split(",")
    new_params = new_input_params[1:-1].split(",")
    base_diff_params = set(base_params) - set(new_params)
    # special case
    if api_str == "torch_npu.profiler.profiler.analyse" and base_diff_params:
        delete_special = [elem for elem in base_diff_params if "max_process_number" not in elem]
        base_diff_params = delete_special

    # case: delete/different default value/different parameter name/different parameter dtype
    if base_diff_params:
        return True

    new_diff_params = set(new_params) - set(base_params)
    for elem in new_diff_params:
        # case: add params
        if "=" not in elem:
            return True

    # case: position parameters
    base_arr = [elem for elem in base_params if "=" not in elem]
    new_arr = [elem for elem in new_params if "=" not in elem]
    i = 0
    while i < len(base_arr):
        if base_arr[i] != new_arr[i]:
            return True
        i += 1

    return False


def api_signature(obj, api_str, content, base_schema, failure_list):
    signature = inspect.signature(obj)
    signature = str(signature)
    if re.search("Any = <module 'pickle' from .+.py'>", signature):
        signature = re.sub("Any = <module 'pickle' from .+\\.py'>", "Any = <module 'pickle'>", signature)
    if re.search(" at 0x[\\da-zA-Z]+>", signature):
        signature = re.sub(" at 0x[\\da-zA-Z]+>", ">", signature)
    if api_str in base_schema.keys():
        value = base_schema[api_str]["signature"]
        if is_not_compatibility(value, signature, api_str=api_str):
            set_failure_list(api_str, value, signature, failure_list)
    content[api_str] = {"signature": signature}


def func_in_class(obj, content, modname, elem, base_schema, failure_list):
    class_variables = [attribute for attribute in obj.__dict__.keys() if not attribute.startswith('__')
                       and callable(getattr(obj, attribute))]
    for variable in class_variables:
        if variable in ["_backward_cls"]:
            continue
        func = getattr(obj, variable)
        api_str = f"{modname}.{elem}.{variable}"
        api_signature(func, api_str, content, base_schema, failure_list)


def func_from_yaml(content, base_schema, failure_list):
    torch_npu_path = torch_npu.__path__[0]
    yaml_path = os.path.join(torch_npu_path, "csrc/aten/npu_native_functions.yaml")
    with open(yaml_path, 'r') as f:
        for line in f.readlines():
            if " func:" in line:
                strings = line.split(" func:")
                if len(strings) < 2:
                    continue
                func = strings[1].strip()
                if "(" in func:
                    func_name = func.split("(")[0]
                    signature = func.split(func_name)[1]
                else:
                    func_name = func
                    signature = ""
                if "func: " + func_name in base_schema:
                    value = base_schema["func: " + func_name]["signature"]
                    if is_not_compatibility(value, signature, api_str=func_name):
                        set_failure_list("func: " + func_name, value, signature, failure_list)
                content["func: " + func_name] = {"signature": signature}


def _find_all_importables(pkg):
    """Find all importables in the project.

    Return them in order.
    """
    return sorted(
        set(
            chain.from_iterable(
                _discover_path_importables(Path(p), pkg.__name__)
                for p in pkg.__path__
            ),
        ),
    )


def _discover_path_importables(pkg_pth, pkg_name):
    """Yield all importables under a given path and package.

    This is like pkgutil.walk_packages, but does *not* skip over namespace
    packages.
    """
    for dir_path, _d, file_names in os.walk(pkg_pth):
        pkg_dir_path = Path(dir_path)

        if pkg_dir_path.parts[-1] == '__pycache__':
            continue

        if all(Path(_).suffix != '.py' for _ in file_names):
            continue

        rel_pt = pkg_dir_path.relative_to(pkg_pth)
        pkg_pref = '.'.join((pkg_name,) + rel_pt.parts)
        yield from (
            pkg_path
            for _, pkg_path, _ in pkgutil.walk_packages(
            (str(pkg_dir_path),), prefix=f'{pkg_pref}.',
        )
        )


class TestPublicApiCompatibility(TestCase):
    @staticmethod
    def _is_mod_public(modname):
        split_strs = modname.split('.')
        for elem in split_strs:
            if elem.startswith("_"):
                return False
        return True

    def test_api_compatibility(self):
        failure_list = []

        try:
            file_abspath = os.path.abspath(__file__)
            air_path = 'third_party/torchair/torchair/tests/st/allowlist_for_publicAPI.json'
            with open(
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file_abspath))),
                                 air_path)) as json_file_torchair:
                allow_dict_torchair = json.load(json_file_torchair)
                update_allow_dict_torchair = {f"torch_npu.dynamo.{key}": value for key, value in
                                              allow_dict_torchair.items()}
        except Exception:
            update_allow_dict_torchair = {}
            warnings.warn(
                "if you are debugging UT file in clone repo, please recursively update the torchair submodule")

        with open(get_file_path_2(os.path.dirname(os.path.dirname(__file__)),
                                  'allowlist_for_publicAPI.json')) as json_file:
            allow_dict = json.load(json_file)
            for modname in allow_dict["being_migrated"]:
                if modname in allow_dict:
                    allow_dict[allow_dict["being_migrated"][modname]] = allow_dict[modname]

        if update_allow_dict_torchair:
            allow_dict.update(update_allow_dict_torchair)

        with open(
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'deprecated_apis.json')) as json_file:
            deprecated_dict = json.load(json_file)

        # load torch_npu_schema.json
        base_schema = {}
        with open(get_file_path_2(os.path.dirname(os.path.dirname(__file__)), "torch_npu_schema.json")) as fp:
            base_schema0 = json.load(fp)
            for key, value in base_schema0.items():
                if not key.startswith("torch_c_func:") and not key.startswith("torch_npu_public_env:"):
                    base_schema[key] = value

        content = {}

        def test_module(modname):
            try:
                if "__main__" in modname or \
                        modname in ["torch_npu.dynamo.torchair.core._backend",
                                    "torch_npu.dynamo.torchair.core._torchair"]:
                    return
                mod = importlib.import_module(modname)
            except Exception:
                # It is ok to ignore here as we have a test above that ensures
                # this should never happen
                return

            if not self._is_mod_public(modname):
                return

            def check_one_element(elem, modname, mod, *, is_public):
                if f"{modname}.{elem}" in NOT_IMPORT_LIST:
                    return
                obj = getattr(mod, elem)
                if not (isinstance(obj, (Callable, torch.dtype)) or inspect.isclass(obj)):
                    return
                if modname in deprecated_dict and elem in deprecated_dict[modname]:
                    return
                elem_module = getattr(obj, '__module__', None)

                modname = allow_dict["being_migrated"].get(modname, modname)
                elem_modname_starts_with_mod = elem_module is not None and \
                                               elem_module.startswith(modname) and \
                                               '._' not in elem_module

                looks_public = not elem.startswith('_') and elem_modname_starts_with_mod
                is_public_api = False
                if is_public != looks_public:
                    if modname in allow_dict and elem in allow_dict[modname]:
                        is_public_api = True
                elif is_public and looks_public:
                    is_public_api = True
                if is_public_api:
                    api_str = f"{modname}.{elem}"
                    api_signature(obj, api_str, content, base_schema, failure_list)

                    # function in class
                    if inspect.isclass(obj):
                        func_in_class(obj, content, modname, elem, base_schema, failure_list)

            if hasattr(mod, '__all__'):
                public_api = mod.__all__
                all_api = dir(mod)
                for elem in all_api:
                    check_one_element(elem, modname, mod, is_public=elem in public_api)
            else:
                all_api = dir(mod)
                for elem in all_api:
                    if not elem.startswith('_'):
                        check_one_element(elem, modname, mod, is_public=True)

        for modname in _find_all_importables(torch_npu):
            test_module(modname)

        test_module('torch_npu')

        # functions from npu_native_functions_by_codegen.yaml
        func_from_yaml(content, base_schema, failure_list)

        base_funcs = base_schema.keys()
        now_funcs = content.keys()
        deleted_apis = set(base_funcs) - set(now_funcs)
        for func in deleted_apis:
            failure_list.append(f"# {func}:")
            failure_list.append(f"  - {func} has been deleted.")

        msg = "All the APIs below do not meet the compatibility guidelines. "
        msg += "If the change timeline has been reached, you can modify the torch_npu_schema.json to make it OK."
        msg += "\n\nFull list:\n"
        msg += "\n".join(map(str, failure_list))
        # empty lists are considered false in python
        self.assertTrue(not failure_list, msg)

    def test_torch_cpp_api_compatibility(self):
        torch_npu_path = os.path.abspath(os.path.dirname(torch_npu.__file__))

        with open(get_file_path_2(os.path.dirname(os.path.dirname(__file__)), "torch_npu_schema.json")) as fp:
            base_schema = json.load(fp)

        failure_list = []
        special_type = ["char *"]

        for key, value in base_schema.items():
            if key.startswith("torch_c_func"):
                if "(" in key:
                    func = key.split("(")[0].split("::")[-1]
                elif "::" in key:
                    func = key.split("::")[-1]
                else:
                    func = key.replace("torch_c_func: ", "")
                signature = value["signature"].replace("c10_npu::", "")
                input_out = signature.split(" -> ")
                input_params = input_out[0]
                out_type = input_out[1] if len(input_out) > 1 else ""
                if "namespace" in value.keys():
                    func = value["namespace"] + func
                if out_type and out_type not in special_type:
                    base_sign = out_type + " " + func + input_params
                else:
                    base_sign = out_type + func + input_params
                file0 = value["file"]
                file1 = os.path.join(torch_npu_path, "include", file0)
                if is_not_compatibility_for_cpp_api(base_sign, file1):
                    failure_list.append(f"# {key}:")
                    failure_list.append(f"  - the signature '{base_sign}' has been changed in the file '{file0}'")

        msg = "All the C++ APIs below do not meet the compatibility guidelines. "
        msg += "If the change timeline has been reached, you can modify the torch_npu_schema.json to make it OK."
        msg += "\n\nFull list:\n"
        msg += "\n".join(map(str, failure_list))

        self.assertTrue(not failure_list, msg)

    def test_public_environments(self):
        torch_npu_path = os.path.abspath(os.path.dirname(torch_npu.__file__))

        with open(get_file_path_2(os.path.dirname(os.path.dirname(__file__)), "torch_npu_schema.json")) as fp:
            base_schema = json.load(fp)
        failure_list = []

        file = "torch_npu/csrc/core/npu/register/OptionsManager.h"
        path = os.path.join(torch_npu_path, "include", file)

        with open(path, mode='r') as fp:
            context = fp.read()

        for key, value in base_schema.items():
            if key.startswith("torch_npu_public_env"):
                base_mode = value["mode"]
                if base_mode not in context:
                    key = key.replace("torch_npu_public_env: ", "")
                    failure_list.append(f"# {key}:")
                    failure_list.append(f"  - the mode of the environment variable {key} has been changed.")

        msg = "All the environment variable's mode below do not meet the compatibility guidelines. "
        msg += "If the change timeline has been reached, you can modify the torch_npu_schema.json to make it OK."
        msg += "\n\nFull list:\n"
        msg += "\n".join(map(str, failure_list))
        # empty lists are considered false in python
        self.assertTrue(not failure_list, msg)


if __name__ == '__main__':
    run_tests()
