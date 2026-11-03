import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from setuptools import Distribution, Extension
import torch.utils.cpp_extension as TorchExtension

from torch_npu.utils import cpp_extension


class TestNpuExtensionEntry(unittest.TestCase):
    def command(self, extensions):
        command = cpp_extension.BuildExtension(
            Distribution({'ext_modules': extensions}), use_ninja=False,
        )
        command.ensure_finalized()
        return command

    def assert_rejected(self, extension):
        command = self.command([extension])
        with mock.patch.object(cpp_extension, 'find_bisheng') as find, \
                mock.patch.object(cpp_extension, 'resolve_npu_arch') as arch, \
                mock.patch.object(TorchExtension, 'is_ninja_available') as ninja, \
                mock.patch.object(TorchExtension.BuildExtension, 'build_extensions') as parent:
            with self.assertRaisesRegex(ValueError, '.*contains .asc sources.*NpuExtension') as error:
                command.build_extensions()
            self.assertIn(extension.name, str(error.exception))
            self.assertIn('torch_npu.utils.cpp_extension.BuildExtension', str(error.exception))
            find.assert_not_called()
            arch.assert_not_called()
            ninja.assert_not_called()
            parent.assert_not_called()

    def test_plain_extensions_rejected_before_toolchain_checks(self):
        for factory in (Extension, TorchExtension.CppExtension):
            with self.subTest(factory=factory.__name__):
                self.assert_rejected(factory('plain_asc', ['missing.asc']))

    def test_incomplete_factory_metadata_is_rejected(self):
        for name in ('_npu_arch', '_npu_dlink', '_npu_dlink_flags'):
            with self.subTest(attribute=name):
                extension = cpp_extension.NpuExtension('incomplete', ['kernel.asc'])
                delattr(extension, name)
                self.assert_rejected(extension)

    def test_asc_added_after_factory_is_rejected(self):
        extension = cpp_extension.NpuExtension('late_asc', ['binding.cpp'])
        extension.sources.append('kernel.asc')
        self.assert_rejected(extension)

    def test_cpp_only_extensions_delegate_without_asc_checks(self):
        for factory in (Extension, TorchExtension.CppExtension, cpp_extension.NpuExtension):
            with self.subTest(factory=factory.__name__):
                command = self.command([factory('cpp_only', ['binding.cpp'])])
                with mock.patch.object(TorchExtension.BuildExtension, 'build_extensions',
                                       return_value='delegated') as parent, \
                        mock.patch.object(cpp_extension, 'find_bisheng') as find:
                    self.assertEqual(command.build_extensions(), 'delegated')
                    parent.assert_called_once_with()
                    find.assert_not_called()

    def test_valid_asc_metadata_survives_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'kernel.asc')
            with open(source, 'w') as stream:
                stream.write('// test\n')
            for dlink in (False, True):
                with self.subTest(dlink=dlink):
                    extension = cpp_extension.NpuExtension(
                        'valid_asc', [source], npu_arch='dav-3510', npu_dlink=dlink,
                        extra_compile_args={'bisheng_dlink': ['-O2']},
                    )
                    command = self.command([extension])
                    command.use_ninja = True
                    command.compiler = SimpleNamespace(src_extensions=['.cpp'])
                    with mock.patch.object(TorchExtension, 'IS_LINUX', True), \
                            mock.patch.object(TorchExtension, 'is_ninja_available', return_value=True), \
                            mock.patch.object(cpp_extension, 'find_bisheng', return_value='bisheng'), \
                            mock.patch.object(command, '_check_bisheng'), \
                            mock.patch.object(TorchExtension.BuildExtension, 'build_extensions') as parent:
                        command.build_extensions()
                    parent.assert_called_once_with()
                    self.assertIs(extension._npu_dlink, dlink)
                    self.assertEqual(extension._npu_dlink_flags, ['-O2'])
                    self.assertIn('.asc', command.compiler.src_extensions)


if __name__ == '__main__':
    unittest.main()
