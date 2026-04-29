
from typing import List, Dict
from abc import ABC, abstractmethod
import torch
from torch._inductor.scheduler import BaseSchedulerNode


class ParallelStrategyBase(ABC):

    def __init__(self):
        self.name: str
        
    @abstractmethod
    def assign_parallel_groups(self, nodes: List[BaseSchedulerNode]) -> Dict[str, List[BaseSchedulerNode]]:
        pass
