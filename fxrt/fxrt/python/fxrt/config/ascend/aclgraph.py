"""Ascend aclgraph config"""

from fxrt._fxrt_aclgraph_config import AclGraphConf
import torch


__all__ = ['AclGraphConf']
acl_graph = AclGraphConf.Instance()
def begin_capture():
    _get_or_create_pool_id()
    acl_graph.begin_capture()

def end_capture():
    acl_graph.end_capture()

def _get_or_create_pool_id():
    if acl_graph.pool_id() == (-1, -1):
        cur_pool_id = torch.npu.graph_pool_handle()
        acl_graph.set_pool_id(cur_pool_id)

def set_op_capture_skip(op_capture_skip: list):
    acl_graph.set_op_capture_skip(op_capture_skip)
