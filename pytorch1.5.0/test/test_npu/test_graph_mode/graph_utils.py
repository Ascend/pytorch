import torch

def RunFuncInGraphMode(func):
    print("graph mode on")
    def wrapper(*args, **kw):
        print("runing: ", func.__name__)
        torch.npu.enable_graph_mode()
        func(*args, **kw)
        print("graph mode off")
        torch.npu.disable_graph_mode()
    return wrapper
