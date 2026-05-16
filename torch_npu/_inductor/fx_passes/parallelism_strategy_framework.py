import os
from typing import List, Dict, Type
import torch
import dataclasses
from torch._inductor.scheduler import BaseSchedulerNode
from .parallelism_strategy_base import ParallelStrategyBase
from .parallelism_strategy_cv import CVParallelismStrategy


@dataclasses.dataclass
class ParallelStrategy:
    strategy: ParallelStrategyBase


parallel_strategys: Dict[str, Type[ParallelStrategy]] = {}


def register_custom_parallel_strategy(strategy_name: str, strategy: ParallelStrategyBase) -> None:
    parallel_strategys[strategy_name] = ParallelStrategy(strategy)


class ParallelGroupingStrategy:
    
    def __init__(self):
        register_custom_parallel_strategy("default", CVParallelismStrategy)
    
    
    def execute_strategy(self, nodes: List[BaseSchedulerNode]) -> Dict[str, List[BaseSchedulerNode]]:
        parallel_scheduler_nodes_min = os.environ.get("PARALLEL_SCHEDULER_NODES_MIN", 20)
        if len(nodes) <= parallel_scheduler_nodes_min:
            return {}
        if "custom" in parallel_strategys:
            parallel_strategy = parallel_strategys["custom"]
        else:
            parallel_strategy = parallel_strategys["default"]
        strategy_cls = parallel_strategy.strategy
        strategy = strategy_cls()
        return strategy.assign_parallel_groups(nodes)