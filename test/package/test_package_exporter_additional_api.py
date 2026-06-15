"""
Add validation cases for torch.package APIs in torch-npu CI:

1. PyTorch community lacks sufficient and direct API validations for some torch.package APIs, so this file is added.
2. This file validates torch.package.PackageExporter.add_dependency,
   torch.package.PackageExporter.all_paths,
   torch.package.PackageExporter.dependency_graph_string,
   torch.package.PackageExporter.get_unique_id,
   torch.package.PackageExporter.register_intern_hook,
   and torch.package.PackageExporter.close.
"""

# Owner(s): ["oncall: package/deploy"]

from io import BytesIO

from torch.package import PackageExporter, PackageImporter, PackagingError
from torch.testing._internal.common_utils import TestCase, run_tests


class TestPackageExporterAdditionalAPI(TestCase):
    """Direct tests for PackageExporter APIs not covered by existing package tests."""

    def test_add_dependency(self):
        buffer = BytesIO()
        exporter = PackageExporter(buffer)

        exporter.add_dependency("math")

        self.assertIn("math", exporter.dependency_graph.nodes)

        exporter.close()
        buffer.seek(0)
        importer = PackageImporter(buffer)

        import math

        self.assertIs(importer.import_module("math"), math)

    def test_add_dependency_nonexistent_module_raises(self):
        buffer = BytesIO()
        exporter = PackageExporter(buffer)

        exporter.add_dependency("nonexistent_module_for_package_exporter_test")

        with self.assertRaises(PackagingError):
            exporter.close()

    def test_all_paths(self):
        exporter = PackageExporter(BytesIO())
        exporter.dependency_graph.add_edge("a", "b")
        exporter.dependency_graph.add_edge("b", "c")
        exporter.dependency_graph.add_edge("a", "d")

        paths = exporter.all_paths("a", "c")

        self.assertIn('"a" -> "b"', paths)
        self.assertIn('"b" -> "c"', paths)
        self.assertNotIn('"a" -> "d"', paths)

    def test_dependency_graph_string(self):
        exporter = PackageExporter(BytesIO())
        exporter.dependency_graph.add_edge("a", "b")

        graph = exporter.dependency_graph_string()

        self.assertIn("digraph G", graph)
        self.assertIn('"a" -> "b"', graph)

    def test_get_unique_id(self):
        exporter = PackageExporter(BytesIO())

        self.assertEqual(exporter.get_unique_id(), "0")
        self.assertEqual(exporter.get_unique_id(), "1")
        self.assertEqual(exporter.get_unique_id(), "2")

    def test_register_intern_hook(self):
        buffer = BytesIO()
        interned_modules = []

        def intern_hook(package_exporter, module_name):
            interned_modules.append(module_name)

        with PackageExporter(buffer) as exporter:
            exporter.register_intern_hook(intern_hook)
            exporter.save_source_string("foo", "VALUE = 1", dependencies=False)

        self.assertEqual(interned_modules, ["foo"])

    def test_register_intern_hook_remove(self):
        buffer = BytesIO()
        interned_modules = []

        def intern_hook(package_exporter, module_name):
            interned_modules.append(module_name)

        with PackageExporter(buffer) as exporter:
            handle = exporter.register_intern_hook(intern_hook)
            handle.remove()
            exporter.save_source_string("foo", "VALUE = 1", dependencies=False)

        self.assertEqual(interned_modules, [])

    def test_close(self):
        buffer = BytesIO()
        exporter = PackageExporter(buffer)
        exporter.save_source_string("foo", "VALUE = 3", dependencies=False)
        exporter.close()

        buffer.seek(0)
        importer = PackageImporter(buffer)
        self.assertEqual(importer.import_module("foo").VALUE, 3)

    def test_close_twice_raises(self):
        exporter = PackageExporter(BytesIO())

        exporter.close()

        with self.assertRaises(Exception):
            exporter.close()


if __name__ == "__main__":
    run_tests()
