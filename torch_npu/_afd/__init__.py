__all__ = ["create_schedule_context_holder"]

from ._schedule_context import _create_schedule_context_holder, ScheduleContextHolder


def create_schedule_context_holder(
        schedule_mode: int,
        session_num: int,
        micro_batch_num: int,
        micro_batch_size: int,
        selected_expert_num: int,
        expert_num: int,
        attn_to_ffn_token_size: int,
        ffn_to_attn_token_size: int,
        ffn_window: int = 0,
        ffn_window_size: int = 0,
        attention_window: int = 0,
        attention_window_size: int = 0,
) -> ScheduleContextHolder:
    """
    A holder class for managing scheduling context in distributed inference.

    Args:
        schedule_mode: Scheduling mode identifier, 0:schedule ffn, 1:shcedule attention
        session_num: Number of sessions
        micro_batch_num: Number of micro batches
        micro_batch_size: micro batch size
        selected_expert_num: selected experts num
        expert_num: Total number of experts
        attn_to_ffn_token_size: Token size from attention to FFN
        ffn_to_attn_token_size: Token size from FFN to attention
        ffn_window: FFN window addr (default: 0), must assign value when schedule_mode=0
        ffn_window_size: FFN window size (default: 0), must assign value when schedule_mode=0
        attention_window: Attention window addr (default: 0), must assign value when schedule_mode=1
        attention_window_size: Attention window size (default: 0), must assign value when schedule_mode=1

    Returns:
        ScheduleContextHolder: the schedule context holder
    """
    return _create_schedule_context_holder(
        schedule_mode,
        session_num,
        micro_batch_num,
        micro_batch_size,
        selected_expert_num,
        expert_num,
        attn_to_ffn_token_size,
        ffn_to_attn_token_size,
        ffn_window,
        ffn_window_size,
        attention_window,
        attention_window_size,
    )
