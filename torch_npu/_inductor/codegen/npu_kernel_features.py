import functools
from typing import Iterable
from typing import Iterable
from typing import Tuple, List
import sympy
import torch
from torch._inductor.codegen.simd import SIMDScheduling
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures, NodeScheduleEntry
from torch._inductor.utils import cache_on_self
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


class NumelList(Tuple):

    def numels(self):
        numel = functools.reduce(lambda a, b: a * b, self)
        return numel

    def __eq__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel == numel2

    def __le__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel <= numel2

    def __lt__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel < numel2

    def __ge__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel >= numel2

    def __gt__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel > numel2

    def __mod__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel % numel2

    def __truediv__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel / numel2

    def __floordiv__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel // numel2

    def __mul__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel * numel2

    def __rmul__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel * numel2

    def __add__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel + numel2

    def __radd__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel + numel2

    def __hash__(self):
        return super(NumelList, self).__hash__()


class NPUKernelFeatures(SIMDKernelFeatures):
    def __init__(
            self,
            node_schedule: List[NodeScheduleEntry],
            numel: sympy.Expr,
            reduction_numel: sympy.Expr = sympy.S.One,
    ):
        super().__init__(node_schedule, numel, reduction_numel)
        self.numel = NumelList(self.numel) if isinstance(self.numel, Iterable) else self.numel
        self.reduction_numel = NumelList(self.reduction_numel) if isinstance(self.reduction_numel,
                                                                             Iterable) else self.reduction_numel
