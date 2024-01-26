"""
This file is mainly used to transform failed test names into DecorateInfo Object.
"""

from pathlib import Path
import os
import yaml

from torchgen.code_template import CodeTemplate
from torchgen.gen import FileManager

from codegen.autograd.utils import VERSION_PART
from codegen.utils import PathManager

project_path = Path(os.path.dirname(__file__)).parent
op_plugin_info_path = os.path.realpath(os.path.join(
    project_path,
    f'third_party/op-plugin/test/test_v{VERSION_PART[0]}r{VERSION_PART[1]}_ops', 
    "unsupported_ops_info.yaml"))
torch_npu_info_path = os.path.realpath(os.path.join(project_path, "test", "unsupported_ops_info.yaml"))

unsupported_dict_path = op_plugin_info_path if os.path.exists(op_plugin_info_path) else torch_npu_info_path
PathManager.check_directory_path_readable(unsupported_dict_path)
with open(unsupported_dict_path, "r") as f:
    unsupported_summary_dict = yaml.safe_load(f)
skip_list = dict()


#class2test is a mapping from test class to test names. As is shown in test_ops.py.
class2test = {
            "TestCommon":[
                "test_compare_cpu",
                "test_dtypes",
                "test_errors",
                "test_multiple_devices",
                "test_non_standard_bool_values",
                "test_noncontiguous_samples",
                "test_numpy_ref",
                "test_out",
                "test_out_integral_dtype",
                "test_out_warning",
                "test_pointwise_tag_coverage",
                "test_variant_consistency_eager",
                "test_complex_half_reference_testing"
            ],
            "TestCompositeCompliance":[
                "test_backward",
                "test_forward_ad",
                "test_operator",
            ],
            "TestMathBits":[
                "test_conj_view",
                "test_neg_conj_view",
                "test_neg_view",
            ],
            "TestFakeTensor":[
                "test_fake",
                "test_fake_autocast",
                "test_fake_crossref_backward_amp",
                "test_fake_crossref_backward_no_amp",
                "test_pointwise_ops",
            ],
            "TestTags":["test_tags"]
}


def get_class_name(func_name):
    for key, value in class2test.items():
        if func_name in value:
            return key
    raise RuntimeError("Can't find corresponding class name of test: {}".format(func_name))


#There are two kinds of templates. One is with Dtype Info, the other is without.
def update_skip_list(class_name, op_name, func_name, dtype):
    skip_reason = "npu test skipped!"
    template = f"""\nDecorateInfo(unittest.skip("{skip_reason}"), \'{class_name}\', \'{func_name}\', dtypes=(torch.{dtype},))"""
    template_ = f"""\nDecorateInfo(unittest.skip("{skip_reason}"), \'{class_name}\', \'{func_name}\')"""
    if dtype:
        if op_name not in skip_list:
            skip_list[op_name] = [template]
        else:
            skip_list[op_name].append(template)
    else:
        if op_name not in skip_list:
            skip_list[op_name] = [template_]
        else:
            skip_list[op_name].append(template_)


def gen_ops_info(summary_dict):
    for op_name in summary_dict:
        for func_name in summary_dict[op_name]:
            for dtype in summary_dict[op_name][func_name]:
                update_skip_list(get_class_name(func_name), op_name, func_name, dtype)
    skip_template = CodeTemplate(
        """\n'${op_name}': [${decorators}]"""
    )
    fm = FileManager(os.path.join("torch_npu", "testing"), os.path.join("codegen", "templates"), False)

    fm.write_with_template(f"_npu_testing_utils.py", "npu_testing_utils.py", lambda:{
        "skip_detail": [skip_template.substitute(op_name=op, decorators=doc) for op, doc in skip_list.items()]
    })

if __name__ == "__main__":
    gen_ops_info(unsupported_summary_dict)
