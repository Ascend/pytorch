from enum import Enum
from functools import total_ordering


@total_ordering
class FxPassLevel(Enum):
    LEVEL1 = 1
    LEVEL2 = 2
    LEVEL3 = 3

    def __lt__(self, other):
        if isinstance(other, FxPassLevel):
            return self.value < other.value
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, FxPassLevel):
            return self.value == other.value
        return NotImplemented
    
    def __hash__(self):
        return hash(self.value)


@total_ordering
class PassType(Enum):
    PRE = 1
    POST = 2

    def __lt__(self, other):
        if isinstance(other, PassType):
            return self.value < other.value
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, PassType):
            return self.value == other.value
        return NotImplemented
    
    def __hash__(self):
        return hash(self.value)


@total_ordering
class ComputeType(Enum):
    VECTOR = 1
    CUBE = 2
    SCALAR = 3
    MEMORY = 4
    MIX = 5
    MLP_VECTOR = 6
    UNKNOWN = 100

    def __lt__(self, other):
        if isinstance(other, ComputeType):
            return self.value < other.value
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, ComputeType):
            return self.value == other.value
        return NotImplemented
    
    def __hash__(self):
        return hash(self.value)


@total_ordering
class GroupType(Enum):
    VECTOR = 1
    CUBE = 2
    MIX_01 = 3
    MIX_02 = 4
    MAIN = 5

    def __lt__(self, other):
        if isinstance(other, ComputeType):
            return self.value < other.value
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, ComputeType):
            return self.value == other.value
        return NotImplemented
    
    def __hash__(self):
        return hash(self.value)