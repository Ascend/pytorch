"""
为 ascend_custom_passes 中各个 fx pass 提供共享测试工具。

这些 pass 大多在 POST 阶段运行（即 AOT autograd 之后），其输入是一个含有
``meta['val']`` (FakeTensor) 的 aten-level FX 图。本模块提供一个轻量的
``GraphBuilder``：每调用一次 ``call`` 既追加一个 ``call_function`` 节点，
又在 ``FakeTensorMode`` 下重放该算子，把结果 FakeTensor 写入节点的
``meta['val']``，从而拼出一个与真实 pass 输入等价的 FX 图。

测试只检查图结构变换是否符合预期，不依赖 NPU 硬件，可在 CPU 上运行。
"""
import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensorMode


def new_fake_mode():
    """构造一个新的 FakeTensorMode；与图节点共享同一 fake_mode 才能复用 FakeTensor。"""
    return FakeTensorMode()


def fake_from_real(fake_mode, real_tensor):
    """将真实 CPU tensor 转换为 fake_mode 下的 FakeTensor。"""
    return fake_mode.from_tensor(real_tensor)


def make_empty_fake(fake_mode, shape, dtype=torch.float32, device="cpu"):
    """直接在 fake_mode 中创建一个指定 shape/dtype 的 FakeTensor。"""
    with fake_mode:
        return torch.empty(shape, dtype=dtype, device=device)


class GraphBuilder:
    """轻量构图工具：自动把每个 call_function 节点的 ``meta['val']`` 设置为
    在 ``fake_mode`` 下执行该算子的输出 FakeTensor。"""

    def __init__(self, fake_mode):
        self.graph = fx.Graph()
        self.fake_mode = fake_mode
        self._placeholders = {}

    def placeholder(self, name, fake_tensor):
        node = self.graph.placeholder(name)
        node.meta["val"] = fake_tensor
        self._placeholders[name] = node
        return node

    def call(self, target, args=(), kwargs=None):
        kwargs = kwargs or {}
        node = self.graph.call_function(target, args=args, kwargs=kwargs)

        def resolve(a):
            if isinstance(a, fx.Node):
                return a.meta.get("val", a)
            if isinstance(a, (list, tuple)):
                return type(a)(resolve(x) for x in a)
            return a

        try:
            with self.fake_mode:
                node.meta["val"] = target(
                    *[resolve(a) for a in args],
                    **{k: resolve(v) for k, v in kwargs.items()},
                )
        except Exception:
            # 某些算子（如 prims.iota）需要走显式 kwargs，调用方可手动补 meta。
            pass
        return node

    def output(self, value):
        self.graph.output(value)

    def to_module(self):
        gm = fx.GraphModule(torch.nn.Module(), self.graph)
        return gm


def count_target(graph, target):
    """统计图中 target == ``target`` 的 call_function 节点数。"""
    return sum(
        1 for n in graph.nodes
        if n.op == "call_function" and n.target is target
    )


def count_any_of(graph, targets):
    """统计图中 target 在 ``targets`` 集合内的 call_function 节点数。"""
    targets = tuple(targets)
    return sum(
        1 for n in graph.nodes
        if n.op == "call_function" and n.target in targets
    )
