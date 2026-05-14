import pkgutil
from collections import defaultdict
from collections.abc import Callable
from typing import List, Optional
from importlib import import_module


PatchFn = Callable[[], None]


class PatchManager:
    """
    Central patch registry for torch_npu.

    PatchManager discovers patch modules in two ways:
    1. Auto-import modules under torch_npu._init.patches whose names end with
       '_patches';
    2. Import external patch modules declared by DEFAULT_EXTRA_PATCH_MODULES
       or registered through register_patch_module().

    Patch modules should register patch functions through:

        @PatchManager.register_patch("group_name")
        def apply_xxx_patch():
            ...
    """

    DEFAULT_PATCH_ORDER = [
        "monkey",
        "api",
        "distributed",
        "dynamo",
        "profiler",
        "npu",
        "warning",
        "asd",
    ]

    DEFAULT_EXTRA_PATCH_MODULES = [
        # Component-owned patch modules outside torch_npu._init.patches.
        # Example:
        # "torch_npu.some_component.foo",
    ]

    PATCH_MODULE_SUFFIX = "_patches"

    _applied_patch_count = defaultdict(int)
    _builtin_patches_registered = False
    _custom_full_patch_order: Optional[List[str]] = None
    _patch_groups = defaultdict(list)
    _patch_modules: list[str] = []

    @classmethod
    def _add_patch(cls, group: str, fn: PatchFn):
        if fn not in cls._patch_groups[group]:
            cls._patch_groups[group].append(fn)

    @classmethod
    def register_patch(cls, group: str, fn: PatchFn | None = None):
        """
        Register a patch function into a patch group.

        Supports:

            PatchManager.register_patch("graph", apply_graph_patch)

        and:

            @PatchManager.register_patch("graph")
            def apply_graph_patch():
                ...
        """
        if not isinstance(group, str) or not group:
            raise ValueError("patch group must be a non-empty string")

        def decorator(real_fn: PatchFn):
            cls._add_patch(group, real_fn)
            return real_fn

        if fn is not None:
            return decorator(fn)

        return decorator

    @classmethod
    def _resolve_patch_order(cls) -> list[str]:
        """
        Resolve final patch execution order.

        - Built-in groups follow DEFAULT_PATCH_ORDER.
        - Groups not listed in DEFAULT_PATCH_ORDER are appended after default groups.
        """
        if cls._custom_full_patch_order is not None:
            base_order = list(cls._custom_full_patch_order)
        else:
            base_order = list(cls.DEFAULT_PATCH_ORDER)

        extra_groups = [group for group in cls._patch_groups if group not in base_order]

        return base_order + extra_groups

    @classmethod
    def _register_builtin_patches(cls):
        """
        Discover and import patch modules.

        This method imports:
        1. built-in patch modules under torch_npu._init.patches whose names end
        with '_patches';
        2. default external patch modules declared in DEFAULT_EXTRA_PATCH_MODULES;
        3. external patch modules registered by register_patch_module().

        Importing a patch module triggers @PatchManager.register_patch(...)
        decorators inside that module.
        """
        if cls._builtin_patches_registered:
            return

        import torch_npu._init.patches as patches_pkg

        # 1. Auto import torch_npu._init.patches/*_patches.py
        prefix = patches_pkg.__name__ + "."
        for module_info in sorted(
            pkgutil.iter_modules(patches_pkg.__path__), key=lambda x: x.name
        ):
            module_name = module_info.name

            if module_name == "patch_manager":
                continue
            if not module_name.endswith(cls.PATCH_MODULE_SUFFIX):
                continue

            import_module(prefix + module_name)

        # 2. Register built-in external patch modules
        for module_name in cls.DEFAULT_EXTRA_PATCH_MODULES:
            cls.register_patch_module(module_name)

        # 3. Import registered external patch modules
        for module_name in cls._patch_modules:
            import_module(module_name)

        cls._builtin_patches_registered = True

    @classmethod
    def apply_registered_patches(cls, group: str):
        """
        Apply newly registered patches in one group.

        This supports delayed apply:
        if new patch functions are registered after this group was applied,
        calling this method again only applies newly added patch functions.
        """
        cls._register_builtin_patches()

        patches = cls._patch_groups.get(group, [])
        start = cls._applied_patch_count[group]

        for patch in patches[start:]:
            patch()

        cls._applied_patch_count[group] = len(patches)

    @staticmethod
    def _patch_excepthook():
        """
        Patch Python global exception hook.

        Kept separate from normal patches because it is paired with shutdown
        exception handling.
        """
        from torch_npu.utils._error_code import _except_handler

        _except_handler.patch_excepthook()

    @classmethod
    def register_patch_module(cls, module_name: str):
        """
        Register an extra module that contains @PatchManager.register_patch(...)
        decorators. The module will be imported before patches are applied.
        """
        if not isinstance(module_name, str) or not module_name:
            raise ValueError("patch module name must be a non-empty string")

        if module_name not in cls._patch_modules:
            cls._patch_modules.append(module_name)

    @classmethod
    def set_patch_order(cls, order: list[str]):
        """
        Override base patch group order.

        Must be called before PatchManager.run().
        Registered groups not listed in this order will still be appended after
        the base order by _resolve_patch_order().
        """
        cls._custom_full_patch_order = list(order)

    @classmethod
    def clear_for_test(cls):
        """
        Test-only helper. Do not use in normal runtime.
        """
        cls._patch_groups.clear()
        cls._patch_modules.clear()
        cls._custom_full_patch_order = None
        cls._builtin_patches_registered = False
        cls._applied_patch_count.clear()


def _apply_all_patches():
    PatchManager._register_builtin_patches()

    for group in PatchManager._resolve_patch_order():
        PatchManager.apply_registered_patches(group)

    PatchManager._patch_excepthook()
