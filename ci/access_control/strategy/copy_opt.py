from .base import AccurateTest


class CopyOptStrategy(AccurateTest):
    """
    通过识别非连续转连续的测试用例
    """

    def identify(self, modify_file):
        if modify_file.find('contiguous') > 0:
            regex = '*contiguous*'
            return AccurateTest.find_ut_by_regex(regex)
        return []
