# Owner(s): ["module: dynamo"]
import functools
import weakref

import torch
import torch_npu
import torch._dynamo
import torch._dynamo.test_case
from torch._C._dynamo import guards
from torch.testing._internal.common_utils import set_default_dtype

RootGuardManager = guards.RootGuardManager
DictGuardManager = guards.DictGuardManager
GetAttrGuardAccessor = guards.GetAttrGuardAccessor
GetItemGuardAccessor = guards.GetItemGuardAccessor
TypeGuardAccessor = guards.TypeGuardAccessor
TENSOR_ALIASING = guards.TENSOR_ALIASING
install_tensor_aliasing_guard = guards.install_tensor_aliasing_guard
NO_TENSOR_ALIASING = guards.NO_TENSOR_ALIASING
install_no_tensor_aliasing_guard = guards.install_no_tensor_aliasing_guard


x1 = torch.tensor(4)
weakref_x = weakref.ref(x1)


class Pair:
    def __init__(self, x, y):
        self.x = x
        self.y = y


global_pair = Pair(torch.randn(4), 1)


def id_type(x):
    return id(type(x))


def equals_match(x, expected):
    return x == expected


def equals_match_verbose_code_parts(expected):
    return [f"x == {expected}"]


def ge_match(x, expected):
    return x >= expected


def ge_match_verbose_code_parts(expected):
    return f"expected >= {expected}"


def less_match(x, expected):
    return x < expected


def less_match_verbose_code_parts(expected):
    return [f"expected < {expected}"]


class GuardManagerTests(torch._dynamo.test_case.TestCase):
    def test_global_state_guard(self):
        guard = guards.GLOBAL_STATE(["global_state_check"])
        self.assertTrue(guard(None))
        with set_default_dtype(torch.double):
            self.assertFalse(guard(None))
        self.assertTrue(guard(None))
        _orig = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(not _orig)
            self.assertFalse(guard(None))
        finally:
            torch.use_deterministic_algorithms(_orig)
        self.assertTrue(guard(None))

    def test_python_lambda_leaf_guard(self):
        const_guard = guards.LAMBDA_GUARD(
            functools.partial(equals_match, expected=5),
            equals_match_verbose_code_parts(5),
        )
        self.assertTrue(const_guard(5))
        self.assertFalse(const_guard(4))
        self.assertFalse(const_guard("foo"))

    def test_type_guard(self):
        foo = 4
        guard = guards.TYPE_MATCH(id_type(foo), ["type(x) == int"])

        self.assertTrue(guard(5))
        self.assertTrue(guard(4))
        self.assertFalse(guard("foo"))

        foo = {"a": 1}
        guard = guards.TYPE_MATCH(id_type(foo), ["type(x) == dict"])
        self.assertTrue(guard(foo))
        self.assertTrue(guard({}))
        self.assertFalse(guard(5))
        self.assertFalse(guard("foo"))

        class Foo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        foo = Foo(1, 2)

        guard = guards.TYPE_MATCH(id_type(foo), ["type(x) == Foo"])
        self.assertTrue(guard(foo))
        self.assertFalse(guard({}))
        self.assertFalse(guard(5))
        self.assertFalse(guard("foo"))

    def test_id_guard(self):
        foo = 4
        guard = guards.ID_MATCH(id(foo), ["id(x) == id(foo)"])

        self.assertTrue(guard(foo))
        self.assertFalse(guard(5))
        self.assertFalse(guard("foo"))

        foo = {"a": 1}
        guard = guards.ID_MATCH(id(foo), ["id(x) == id(foo)"])
        self.assertTrue(guard(foo))
        self.assertFalse(guard({"a": 1}))
        self.assertFalse(guard({}))
        self.assertFalse(guard(5))

    def test_equals_guard(self):
        foo = 4
        guard = guards.EQUALS_MATCH(foo, ["x == 4"])

        self.assertTrue(guard(4))
        self.assertFalse(guard(5))
        self.assertFalse(guard("foo"))

        # tuple
        foo = (1, 2, 3)
        guard = guards.EQUALS_MATCH(foo, ["x == foo"])
        self.assertTrue(guard(foo))
        self.assertTrue(guard((1, 2, 3)))
        self.assertFalse(guard((1, 2, 3, 4)))
        self.assertFalse(guard({}))

        # list
        foo = [1, 2, 3]
        guard = guards.EQUALS_MATCH(foo, ["x == foo"])
        self.assertTrue(guard(foo))
        self.assertTrue(guard([1, 2, 3]))
        self.assertFalse(guard([1, 2, 3, 4]))

        # type
        foo = int
        guard = guards.EQUALS_MATCH(foo, ["x == foo"])
        self.assertTrue(guard(foo))
        self.assertTrue(guard(int))
        self.assertFalse(guard(float))

    def test_default_device_guard(self):
        foo = 1
        guard = guards.DEFAULT_DEVICE(["cpu device"])
        self.assertTrue(guard(foo))

        try:
            torch.set_default_device("npu:0")
            self.assertFalse(guard(foo))
        finally:
            torch.set_default_device(None)

    def test_data_ptr_match_guard(self):
        foo = torch.tensor([1, 2, 3])
        guard = guards.DATA_PTR_MATCH(foo, ["x.data_ptr() == foo.data_ptr()"])
        self.assertTrue(guard(foo))
        self.assertFalse(guard(torch.tensor([1, 2, 3])))

    def test_length_check_guard(self):
        foo = [1, 2, 3]
        guard = guards.LENGTH_CHECK(len(foo), ["len(x) == len(foo)"])
        self.assertTrue(guard(foo))
        self.assertFalse(guard([]))

    def test_no_hasattr_guard(self):
        class Bar:
            def __init__(self):
                self.bar = 2

        bar = Bar()

        class Foo:
            def __init__(self):
                self.foo = 2

        foo = Foo()

        guard = guards.NO_HASATTR("foo", ["hasattr(x, 'foo') == False"])
        self.assertTrue(guard(bar))
        self.assertFalse(guard(foo))

    def test_tensor_aliasing_guard(self):
        guard_manager = RootGuardManager()

        a = torch.randn(3, 4)

        class Foo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        f_locals = Foo(a, a)

        x_guard_mgr = guard_manager.getattr_manager("x", "", a)
        y_guard_mgr = guard_manager.getattr_manager("y", "", a)
        install_tensor_aliasing_guard(x_guard_mgr, y_guard_mgr, ["x is y"])

        # Check structure
        x_guards = x_guard_mgr.get_leaf_guards()
        y_guards = y_guard_mgr.get_leaf_guards()
        self.assertEqual(len(x_guards), 1)
        self.assertEqual(len(y_guards), 1)
        self.assertTrue(isinstance(x_guards[0], TENSOR_ALIASING))
        self.assertTrue(isinstance(y_guards[0], TENSOR_ALIASING))
        # Check that the two guards are the same object
        self.assertTrue(x_guards[0] is y_guards[0])

        f_locals_unaliased = Foo(torch.randn(3, 4), torch.randn(3, 4))
        self.assertEqual(len(x_guard_mgr.get_leaf_guards()), 1)
        self.assertEqual(len(y_guard_mgr.get_leaf_guards()), 1)
        self.assertTrue(guard_manager.check(f_locals))

        self.assertFalse(guard_manager.check(f_locals_unaliased))

    def test_dict_version_guard(self):
        foo = {"a": 1, "b": 2}
        guard = guards.DICT_VERSION(foo, ["x.version == foo.version"])

        self.assertTrue(guard(foo))
        self.assertFalse(guard(dict(foo)))
        foo["a"] = 2
        self.assertFalse(guard(foo))
        self.assertFalse(guard({"a": 1, "b": 2}))
        self.assertFalse(guard({}))

    def test_dynamic_indices_guard(self):
        guard1 = guards.DYNAMIC_INDICES(False, set(), ["x.size(0) == y.size(0)"])
        guard2 = guards.DYNAMIC_INDICES(True, set({0, 1}), ["x.size(0) == y.size(0)"])

        x = torch.randn(4)
        self.assertTrue(guard1(x))
        self.assertTrue(guard2(x))

        x._dynamo_dynamic_indices = set({0})
        self.assertFalse(guard1(x))
        self.assertTrue(guard2(x))

        x._dynamo_dynamic_indices = set({2})
        self.assertFalse(guard1(x))
        self.assertFalse(guard2(x))

    def test_tensor_match_guard(self):
        guard_manager = RootGuardManager()
        x = torch.randn(4, 4)
        size = list(x.size())
        stride = list(x.stride())
        guard_manager.add_tensor_match_guard(x, size, stride, "x", ["check_tensor(x)"])
        self.assertTrue(guard_manager.check(x))
        self.assertTrue(guard_manager.check_verbose(x).result)
        self.assertTrue(guard_manager.check(torch.randn(4, 4)))
        self.assertTrue(guard_manager.check_verbose(torch.randn(4, 4)).result)
        self.assertFalse(guard_manager.check(x.t_()))

        x = torch.randn(4, 4)
        x.t_()
        debug_info = guard_manager.check_verbose(x)
        print(debug_info.verbose_code_parts[0])
        self.assertTrue(
            "tensor 'x' stride mismatch" in debug_info.verbose_code_parts[0]
        )

    def test_no_tensor_aliasing_guard(self):
        guard_manager = RootGuardManager()

        a = torch.randn(3, 4)

        class Foo:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        f_locals = Foo(a, a, a)

        x_guard_mgr = guard_manager.getattr_manager("x", "", a)
        y_guard_mgr = guard_manager.getattr_manager("y", "", a)
        z_guard_mgr = guard_manager.getattr_manager("z", "", a)
        install_no_tensor_aliasing_guard(
            [x_guard_mgr, y_guard_mgr, z_guard_mgr],
            ["x", "y", "z"],
            ["no_aliasing(x, y, z)"],
        )

        # Check structure
        x_guards = x_guard_mgr.get_leaf_guards()
        y_guards = y_guard_mgr.get_leaf_guards()
        z_guards = z_guard_mgr.get_leaf_guards()
        self.assertEqual(len(x_guards), 1)
        self.assertEqual(len(y_guards), 1)
        self.assertEqual(len(z_guards), 1)
        self.assertTrue(isinstance(x_guards[0], NO_TENSOR_ALIASING))
        self.assertTrue(isinstance(y_guards[0], NO_TENSOR_ALIASING))
        self.assertTrue(isinstance(z_guards[0], NO_TENSOR_ALIASING))
        # Check that the two guards are the same object
        self.assertTrue(x_guards[0] is y_guards[0] is z_guards[0])
        self.assertFalse(guard_manager.check(f_locals))
        self.assertFalse(guard_manager.check_verbose(f_locals).result)

        f_locals_unaliased = Foo(
            torch.randn(3, 4),
            torch.randn(3, 4),
            torch.randn(3, 4),
        )
        self.assertTrue(guard_manager.check(f_locals_unaliased))
        self.assertTrue(guard_manager.check_verbose(f_locals_unaliased).result)
        # Check that hash map is cleared.
        self.assertTrue(guard_manager.check(f_locals_unaliased))

        f_locals_unaliased = Foo(
            a,
            torch.randn(3, 4),
            a,
        )
        self.assertFalse(guard_manager.check(f_locals_unaliased))
        self.assertFalse(guard_manager.check_verbose(f_locals_unaliased).result)

    def test_weakref_alive_guard(self):
        x = torch.rand(3, 4)
        weakref_x_copy = weakref.ref(x)

        guard = guards.WEAKREF_ALIVE(["weakref_x_copy is not None"])
        self.assertTrue(guard(weakref_x_copy()))
        del x
        self.assertFalse(guard(weakref_x_copy()))

    def test_guard_manager_leaf_guard(self):
        guard_manager = RootGuardManager()
        guard_manager.add_type_match_guard(id_type(5), ["type(x) == int"])
        guard_manager.add_lambda_guard(
            functools.partial(ge_match, expected=5),
            ge_match_verbose_code_parts(expected=5),
        )
        guard_manager.add_lambda_guard(
            functools.partial(less_match, expected=10),
            less_match_verbose_code_parts(expected=10),
        )
        self.assertEqual(len(guard_manager.get_leaf_guards()), 3)
        self.assertEqual(len(guard_manager.get_accessors()), 0)
        self.assertTrue(guard_manager.check(6))
        self.assertFalse(guard_manager.check(4))
        self.assertFalse(guard_manager.check("foo"))

    def test_attr_guard_manager(self):
        class Foo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        foo = Foo(1, 2)
        guard_manager = RootGuardManager()
        guard_manager.add_type_match_guard(id_type(foo), ["type(x) == Foo"])
        guard_manager.getattr_manager("x", "x", 1).add_lambda_guard(
            functools.partial(equals_match, expected=foo.x),
            equals_match_verbose_code_parts(foo.x),
        )
        guard_manager.getattr_manager("y", "y", 2).add_lambda_guard(
            functools.partial(equals_match, expected=foo.y),
            equals_match_verbose_code_parts(foo.y),
        )
        self.assertEqual(len(guard_manager.get_leaf_guards()), 1)
        # 2 child managers, one for x and one for y
        self.assertEqual(len(guard_manager.get_accessors()), 2)
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[0], GetAttrGuardAccessor)
        )
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[1], GetAttrGuardAccessor)
        )
        # Check leaf guards on child managers
        self.assertEqual(
            len(
                guard_manager.getattr_manager(
                    attr="x", source="x", example_value=None
                ).get_leaf_guards()
            ),
            1,
        )
        self.assertEqual(
            len(guard_manager.getattr_manager("y", "y", None).get_leaf_guards()), 1
        )

        self.assertTrue(guard_manager.check(foo))
        self.assertFalse(guard_manager.check(Foo(3, 4)))
        self.assertFalse(guard_manager.check("foo"))

    def test_item_guard_manager(self):
        foo = [1, 2]
        guard_manager = RootGuardManager()
        guard_manager.add_type_match_guard(id_type(foo), ["type(x) == Foo"])
        guard_manager.getitem_manager(0, "", 1).add_lambda_guard(
            functools.partial(equals_match, expected=foo[0]),
            equals_match_verbose_code_parts(foo[0]),
        )
        guard_manager.getitem_manager(1, "", 2).add_lambda_guard(
            functools.partial(equals_match, expected=foo[1]),
            equals_match_verbose_code_parts(foo[1]),
        )
        self.assertEqual(len(guard_manager.get_leaf_guards()), 1)
        # 2 child managers, one for x and one for y
        self.assertEqual(len(guard_manager.get_accessors()), 2)
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[0], GetItemGuardAccessor)
        )
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[1], GetItemGuardAccessor)
        )
        # Check leaf guards on child managers
        self.assertEqual(
            len(guard_manager.getitem_manager(0, "", None).get_leaf_guards()), 1
        )
        self.assertEqual(
            len(guard_manager.getitem_manager(1, "", None).get_leaf_guards()), 1
        )

        self.assertTrue(guard_manager.check(foo))
        self.assertFalse(guard_manager.check([3, 4]))
        self.assertFalse(guard_manager.check("foo"))

    def test_dict_getitem_accessor(self):
        foo = {
            "a": 1,
            "b": 2,
        }

        guards_manager = RootGuardManager()
        guards_manager.add_type_match_guard(id_type(foo), ["type(x) == Foo"])
        guards_manager.dict_getitem_manager("a", "", 1).add_equals_match_guard(
            1, ["a == 1"]
        )
        guards_manager.dict_getitem_manager("b", "", 2).add_equals_match_guard(
            2, ["b == 2"]
        )

        self.assertTrue(guards_manager.check(foo))
        self.assertFalse(guards_manager.check({"a": 1, "b": 3}))

    def test_globals(self):
        global global_pair, Pair
        guard_manager = RootGuardManager()
        gpair_mgr = guard_manager.globals_dict_manager(
            globals(), "", None
        ).getitem_manager("global_pair", "", global_pair)

        gpair_mgr.add_lambda_guard(
            lambda x: isinstance(x, Pair)
            and isinstance(x.x, torch.Tensor)
            and isinstance(x.y, int),
            "global guard fail",
        )

        self.assertTrue(guard_manager.check(global_pair))
        global_pair.y = "foo"
        self.assertFalse(guard_manager.check(global_pair))

    def test_type_manager(self):
        guard_manager = RootGuardManager()

        class A:
            a = 4

        class B(A):
            def mul(self, x):
                super().mul(x)

        foo = B()
        f_locals = {"foo": foo}

        # len(type(foo).__mro__) == 2
        foo_mgr = guard_manager.getitem_manager("foo", "", foo)
        type_manager = foo_mgr.type_manager("", type(foo))
        self.assertTrue(isinstance(foo_mgr.get_accessors()[0], TypeGuardAccessor))
        mro_manager = type_manager.getattr_manager("__mro__", "", type(foo).__mro__)
        self.assertTrue(
            isinstance(type_manager.get_accessors()[0], GetAttrGuardAccessor)
        )
        mro_manager.add_length_check_guard(
            3,
            "Expected len(type(foo).__mro__) == 3",
        )

        # type(foo).__mro__[0].a = 4
        item_manager = mro_manager.getitem_manager(1, "", type(foo).__mro__[1])
        self.assertTrue(
            isinstance(mro_manager.get_accessors()[0], GetItemGuardAccessor)
        )
        attr_manager = item_manager.getattr_manager("a", "", type(foo).__mro__[0].a)
        self.assertTrue(
            isinstance(item_manager.get_accessors()[0], GetAttrGuardAccessor)
        )
        attr_manager.add_lambda_guard(
            lambda x: x == 4,
            "Expected value 4",
        )

        self.assertTrue(guard_manager.check(f_locals))

    def test_tuple_iterator_getitem(self):
        a = (1, 2, 3, 4, 5, 6)
        foo = iter(a)
        next(foo)  # foo points at index=1

        guard_manager = RootGuardManager()
        # Check a[3] which is tuple_iterator_getitem(foo, 2)
        guard_manager.add_tuple_iterator_length_guard(
            5, id_type(iter(tuple())), ["len == 5"]
        )
        guard_manager.tuple_iterator_getitem_manager(2, "", foo).add_equals_match_guard(
            a[3], ["x==4"]
        )

        # Check that type match works
        self.assertFalse(guard_manager.check(False))

        self.assertTrue(guard_manager.check(foo))

        # Check that index error fails gracefully
        b = (1, 2)
        b_foo = iter(b)
        self.assertFalse(guard_manager.check(b_foo))

    def test_global_weakref(self):
        guard_manager = RootGuardManager()
        globals_manager = guard_manager.globals_dict_manager(globals(), "", None)
        weakref_manager = globals_manager.global_weakref_manager("weakref_x", "", None)

        weakref_manager.add_lambda_guard(
            lambda x: isinstance(x, torch.Tensor),
            "global weakref fail",
        )

        self.assertTrue(guard_manager.check(None))
        global x1
        del x1
        self.assertFalse(guard_manager.check(None))

    def test_lambda_manager(self):
        a = (1, 1, 3, 4, 5, 6)

        guard_manager = RootGuardManager()

        # Check that we can use the same accessor
        foo_mgr = guard_manager.lambda_manager(lambda x: x[2], "", None)
        foo_mgr.add_lambda_guard(
            lambda x: x == 3,
            "Expected value 3",
        )
        self.assertTrue(guard_manager.check(a))

        # test that exception works
        guard_manager = RootGuardManager()

        def fn(x):
            raise AssertionError("Test")

        foo_mgr = guard_manager.lambda_manager(fn, "", None)

        self.assertFalse(guard_manager.check(None))
        debug_info = guard_manager.check_verbose(None)
        self.assertFalse(debug_info.result)
        self.assertTrue("Test" in debug_info.verbose_code_parts[0])

    def test_dict_contains_guard(self):
        foo = {"a": 1, "b": 2}
        guard = guards.DICT_CONTAINS(True, "a", ["has a"])

        self.assertTrue(guard(foo))
        self.assertTrue(guard({"a": 1, "b": 2}))
        self.assertFalse(guard({"b": 2, "c": 3}))
        self.assertFalse(guard({}))

        guard = guards.DICT_CONTAINS(False, "c", ["not has c"])
        self.assertTrue(guard(foo))
        self.assertTrue(guard({"a": 1, "b": 2}))
        self.assertFalse(guard({"b": 2, "c": 3}))
        self.assertTrue(guard({}))

    def test_dict_guard_manager(self):
        root = RootGuardManager()

        def nothing():
            pass

        f_locals = {
            "d": {"a": 1, nothing: {"z": 3}, 100: torch.randn(4)},
        }

        # its a getitem_manager just for f_locals. But the child guard manager
        # should be a DictGuardManager.
        dict_mgr = root.getitem_manager("d", "", f_locals["d"])
        self.assertTrue(isinstance(dict_mgr, DictGuardManager))

        self.assertTrue(root.check(f_locals))

        # Check that no one can add a leaf guard
        with self.assertRaises(RuntimeError):
            dict_mgr.add_id_match_guard(id_type(f_locals), "id match")

        # Check that no one can add an arbitrary accessor
        with self.assertRaises(RuntimeError):
            dict_mgr.getitem_manager("a", "", f_locals.get("d").get("a"))

        # Check that it fails with different length dict
        f_locals_prime = {
            "d": {"a": 1, "b": 2},
        }
        self.assertFalse(root.check(f_locals_prime))

        # Add key-value manager ("a" : 1)
        self.assertTrue(root.check(f_locals))
        dict_mgr.get_key_manager(0, "", "a").add_equals_match_guard(
            "a", ["dict.keys()[0] == a"]
        )
        self.assertTrue(root.check(f_locals))
        dict_mgr.get_value_manager(0, "", 1).add_equals_match_guard(1, ["d[0] == 1"])
        self.assertTrue(root.check(f_locals))

        # Add key-value manager (nothing : {"z" : 3})
        self.assertTrue(root.check(f_locals))
        dict_mgr.get_key_manager(1, "", nothing).add_lambda_guard(
            lambda x: x is nothing, ["x is nothing"]
        )
        self.assertTrue(root.check(f_locals))
        value_mgr = dict_mgr.get_value_manager(1, "", f_locals.get("d")[nothing])
        self.assertTrue(isinstance(value_mgr, DictGuardManager))
        self.assertTrue(root.check(f_locals))

        # Check structure
        # Check that we are only guarding on two keys. This is common in
        # LazyVariableTracker.
        self.assertEqual(len(dict_mgr.get_key_value_managers()), 2)

        f_locals["d"]["a"] = 2
        self.assertFalse(root.check(f_locals))
        self.assertFalse(root.check_verbose(f_locals).result)

        f_locals["d"]["a"] = 1
        self.assertTrue(root.check(f_locals))

        f_locals["d"].pop(100)
        # fails because of len check
        self.assertFalse(root.check(f_locals))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
