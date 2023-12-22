from enum import Enum


class SubModuleID(Enum):
    PTA = 0
    OPS = 1
    DIST = 2
    GRAPH = 3
    PROF = 4


class ErrCode(Enum):
    SUCCESS = 0
    PARAM = 1
    TYPE = 2
    VALUE = 3
    PTR = 4
    INTERNAL = 5
    MEMORY = 6
    NOT_SUPPORT = 7
    NOT_FOUND = 8
    UNAVAIL = 9
    SYSCALL = 10
    TIMEOUT = 11
    PERMISSION = 12
    ACL = 100
    HCCL = 200
    GE = 300


def pta_error(error: ErrCode) -> str:
    return "\nERR{:0>2d}{:0>3d}".format(SubModuleID.PTA.value, error.value)


def ops_error(error: ErrCode) -> str:
    return "\nERR{:0>2d}{:0>3d}".format(SubModuleID.OPS.value, error.value)


def dist_error(error: ErrCode) -> str:
    return "\nERR{:0>2d}{:0>3d}".format(SubModuleID.DIST.value, error.value)


def graph_error(error: ErrCode) -> str:
    return "\nERR{:0>2d}{:0>3d}".format(SubModuleID.GRAPH.value, error.value)


def prof_error(error: ErrCode) -> str:
    return "\nERR{:0>2d}{:0>3d}".format(SubModuleID.PROF.value, error.value)
