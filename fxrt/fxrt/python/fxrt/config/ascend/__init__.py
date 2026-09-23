"""Fxrt Ascend configs"""

from .op_precision import OpPrecisionConf
from .aclgraph import AclGraphConf

op_precision = OpPrecisionConf.Instance()
acl_graph = AclGraphConf.Instance()

__all__ = ['op_precision', 'acl_graph']
