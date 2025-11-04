import enum
import torch

from torch_npu._C import _npu_from_dlpack
from torch_npu._C import _npu_to_dlpack


def _to_dlpack(tensor):
    return _npu_to_dlpack(tensor)


def _from_dlpack(ext_tensor) -> 'torch.Tensor':
    if hasattr(ext_tensor, '__dlpack__'):
        dlpack = ext_tensor.__dlpack__()
    else:
        # Old versions just call the converter
        dlpack = ext_tensor
    return _npu_from_dlpack(dlpack)


def _apply_dlpack_patch():
    """Patch torch.utils.dlpack and torch.utils to use torch_npu implementation for NPU tensors"""
    import torch.utils.dlpack as torch_dlpack
    
    # Store original functions
    _original_to_dlpack = torch_dlpack.to_dlpack
    _original_from_dlpack = torch_dlpack.from_dlpack
    
    def create_patched_to_dlpack(module_name):
        """Create a patched to_dlpack function with proper __module__ attribute"""
        def patched_to_dlpack(tensor):
            """Patched to_dlpack that uses torch_npu implementation for NPU tensors"""
            if hasattr(tensor, 'device') and tensor.device.type == 'npu':
                return _to_dlpack(tensor)
            return _original_to_dlpack(tensor)
        patched_to_dlpack.__module__ = module_name
        return patched_to_dlpack
    
    def create_patched_from_dlpack(module_name):
        """Create a patched from_dlpack function with proper __module__ attribute"""
        def patched_from_dlpack(ext_tensor):
            """Patched from_dlpack that uses torch_npu implementation when appropriate"""
            # For NPU tensors or when torch_npu is available, use our implementation
            try:
                return _from_dlpack(ext_tensor)
            except Exception:
                # Fallback to original implementation
                return _original_from_dlpack(ext_tensor)
        patched_from_dlpack.__module__ = module_name
        return patched_from_dlpack
    
    # Apply patches to torch.utils.dlpack
    torch_dlpack.to_dlpack = create_patched_to_dlpack('torch.utils.dlpack')
    torch_dlpack.from_dlpack = create_patched_from_dlpack('torch.utils.dlpack')
    
    # Also patch torch.utils.to_dlpack and torch.utils.from_dlpack if they exist
    if hasattr(torch.utils, 'to_dlpack'):
        _original_torch_utils_to_dlpack = torch.utils.to_dlpack
        torch.utils.to_dlpack = create_patched_to_dlpack('torch.utils')
    
    if hasattr(torch.utils, 'from_dlpack'):
        _original_torch_utils_from_dlpack = torch.utils.from_dlpack
        torch.utils.from_dlpack = create_patched_from_dlpack('torch.utils')
    
    # Also patch torch.from_dlpack and torch.to_dlpack if they exist
    if hasattr(torch, 'from_dlpack'):
        _original_torch_from_dlpack = torch.from_dlpack
        torch.from_dlpack = create_patched_from_dlpack('torch')
        
    if hasattr(torch, 'to_dlpack'):
        _original_torch_to_dlpack = torch.to_dlpack
        torch.to_dlpack = create_patched_to_dlpack('torch')
        
        # Add to_dlpack to torch.__all__ if it exists, otherwise create it
        if not hasattr(torch, '__all__'):
            torch.__all__ = []
        if 'to_dlpack' not in torch.__all__:
            torch.__all__.append('to_dlpack')
    
    # Also ensure from_dlpack is in torch.__all__ if it exists
    if hasattr(torch, 'from_dlpack'):
        if not hasattr(torch, '__all__'):
            torch.__all__ = []
        if 'from_dlpack' not in torch.__all__:
            torch.__all__.append('from_dlpack')
