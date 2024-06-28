import re
from pathlib import Path
from .base import AccurateTest


class OpStrategy(AccurateTest):
    """
    通过对适配层的代码的识别
    """

    def identify(self, modify_file):
        """
        通过对于算子实现文件的文件名解析获取其单元测试的名字，比如：
        BinaryCrossEntropyWithLogitsBackwardKernelNpu.cpp
        针对这个文件，先识别关键字BinaryCrossEntropyWithLogitsBackward
        然后，获取其正则表达式*binary*cross*entropy*with*logits*backward*识别到符合要求的测试用例
        具体方法：通过大写字母切分关键字，再识别包含所有这些关键字的测试文件名。
        """
        filename = Path(modify_file).name
        if filename.find('KernelNpu') >= 0:
            feature_line = filename.split('KernelNpu')[0]
            features = re.findall('[A-Z][^A-Z]*', feature_line)
            regex = '*' + '*'.join([f"{feature.lower()}" for feature in features]) + '*'
            return AccurateTest.find_ut_by_regex(regex)
        return []
