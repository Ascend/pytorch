import torch
from torch._inductor.pattern_matcher import CallFunction, KeywordArg, LoweringPatternEntry
from torch._inductor.fx_passes.post_grad import pass_patterns, is_valid_mm_plus_mm

aten = torch.ops.aten


def patch_pattern_mm_plus_mm():

    def is_mm_plus_mm(entry) -> bool:
        if isinstance(entry, LoweringPatternEntry):
            handler_name = getattr(entry.handler, '__name__', '')
            return handler_name == 'mm_plus_mm'
        return False

    pattern = CallFunction(
        aten.add, 
        CallFunction(aten.mm, KeywordArg("mat1"), KeywordArg("mat2")),
        CallFunction(aten.mm, KeywordArg("mat3"), KeywordArg("mat4")), 
        extra_check=is_valid_mm_plus_mm
    )

    # currently, torch_npu does not support mm_plus_mm fusion
    for fn in pattern.fns:
        index = None
        for i, pattern_entry in enumerate(pass_patterns[1].patterns[(pattern.op, fn)]):
            if is_mm_plus_mm(pattern_entry):
                pass_patterns[1].patterns[(pattern.op, fn)].pop(i)
                break