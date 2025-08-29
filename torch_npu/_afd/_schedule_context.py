__all__ = ["ScheduleContextHolder"]

import torch
import torch_npu


class ScheduleContextHolder:
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
    """
    # 记录afd模块是否已初始化
    _afd_initialized = False

    def __init__(
            self,
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
    ) -> None:
        if not ScheduleContextHolder._afd_initialized:
            ScheduleContextHolder._init_afd_module()

        self._impl = torch_npu._C._afd.ScheduleContextHolder(schedule_mode,
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
                                                            attention_window_size)

    @classmethod
    def _init_afd_module(cls):
        if not hasattr(torch_npu._C, "_afd_init"):
            raise RuntimeError("Failed to init _afd module as _afd_init is not found in torch_npu._C")

        if not torch_npu._C._afd_init():
            raise RuntimeError("Failed to init _afd module")

        cls._afd_initialized = True

    def init(self) -> None:
        ret = self._impl.init()
        if ret != 0:
            raise RuntimeError(f'ScheduleContextHolder init return {ret}')

    def get_schedule_context_tensor(self) -> torch.Tensor:
        """
        Get the scheduling context tensor.

        Returns:
            torch.Tensor: The context tensor

        Raises:
            RuntimeError: If tensor retrieval fails
        """
        ret, tensor = self._impl.get_context_tensor()
        if ret != 0:
            raise RuntimeError(f'get_context_tensor returned {ret}')
        return tensor

    def stop_schedule(self) -> None:
        """
        Stop scheduling.

        Raises:
            RuntimeError: If set stop flag to context fails
        """
        ret = self._impl.stop_schedule()
        if ret != 0:
            raise RuntimeError(f'stop returned {ret}')

    def get_schedule_context_info(self) -> str:
        """
        get schedule context info.

        Returns:
            str: the schedule context string info
        """
        return self._impl.get_schedule_context_info()


def _create_schedule_context_holder(
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
    holder = ScheduleContextHolder(
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
    holder.init()
    return holder
