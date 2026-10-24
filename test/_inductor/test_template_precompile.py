import ast
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[2] / "torch_npu/_inductor/runtime/triton_heuristics.py"


def load_definition(name, namespace):
    # Exercise checkout code without importing TorchNPU or initializing a device.
    tree = ast.parse(SOURCE.read_text())
    node = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


class HeuristicType(Enum):
    TEMPLATE = auto()
    POINTWISE = auto()
    REDUCTION = auto()
    PERSISTENT_REDUCTION = auto()
    USER_AUTOTUNE = auto()


class TestTemplatePrecompile(unittest.TestCase):
    def route(self, heuristic, size_hints, threads=4, configs=1, **overrides):
        events = []
        obj = SimpleNamespace(
            heuristic_type=heuristic, size_hints=size_hints,
            configs=list(range(configs)), compile_results=[], skip_precompile=False,
            _prepare_precompile=lambda: events.append("prepare"),
            _precompile_worker_serial=lambda: events.append("serial"),
            _precompile_worker_parallel=lambda: events.append("parallel"),
        )
        obj.__dict__.update(overrides)
        worker = load_definition("_precompile_worker", dict(
            npu_config=SimpleNamespace(precompile_thread_num=threads),
            HeuristicType=HeuristicType))
        worker(obj)
        return events

    def test_template_compiles_on_calling_thread(self):
        for configs in (1, 3):
            with self.subTest(configs=configs):
                self.assertEqual(self.route(HeuristicType.TEMPLATE, None, configs=configs),
                                 ["prepare", "serial"])

    def test_vector_multiconfig_keeps_parallel_compile(self):
        for heuristic in HeuristicType:
            with self.subTest(heuristic=heuristic):
                self.assertEqual(self.route(heuristic, {"x": 1024}, configs=3),
                                 ["prepare", "parallel"])

    def test_missing_size_hints_alone_does_not_mark_template(self):
        self.assertEqual(self.route(HeuristicType.POINTWISE, None, configs=3),
                         ["prepare", "parallel"])

    def test_one_thread_keeps_serial_compile(self):
        self.assertEqual(self.route(HeuristicType.REDUCTION, {"x": 1024}, threads=1, configs=3),
                         ["prepare", "serial"])

    def test_cached_results_and_skip_keep_existing_paths(self):
        self.assertEqual(self.route(HeuristicType.TEMPLATE, None, compile_results=[object()]),
                         ["serial"])
        self.assertEqual(self.route(HeuristicType.TEMPLATE, None, skip_precompile=True), [])

    def test_template_and_foreach_decorator_metadata(self):
        hints = {"x": 1024}

        def cached_autotune(size_hints, configs, **kwargs):
            return dict(size_hints=size_hints, configs=configs, **kwargs)

        namespace = dict(
            cached_autotune=cached_autotune, HeuristicType=HeuristicType,
            triton=SimpleNamespace(Config=lambda *a, **kw: object()),
            triton_config_npu_index=lambda *a, **kw: [object(), object(), object()],
        )
        template = load_definition("template", namespace)
        foreach = load_definition("foreach", namespace)
        template_meta = template(num_stages=1, num_warps=4, triton_meta={})
        foreach_meta = foreach(size_hints=hints, num_warps=4, triton_meta={})
        self.assertIsNone(template_meta["size_hints"])
        self.assertIs(foreach_meta["size_hints"], hints)
        self.assertEqual(template_meta["heuristic_type"], HeuristicType.TEMPLATE)
        self.assertEqual(foreach_meta["heuristic_type"], HeuristicType.TEMPLATE)
        self.assertEqual(len(foreach_meta["configs"]), 3)


if __name__ == "__main__":
    unittest.main()
