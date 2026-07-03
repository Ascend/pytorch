# torch.optim

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T02:21:01.198Z pushedAt=2026-06-15T03:25:49.207Z -->

> [!NOTE]  
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|[torch.optim.Optimizer](https://pytorch.org/docs/2.9/optim.html#torch.optim.Optimizer)|Yes|-|
|[Optimizer.add_param_group](https://pytorch.org/docs/2.9/generated/torch.optim.Optimizer.add_param_group.html)|Yes|-|
|[Optimizer.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.Optimizer.load_state_dict.html)|Yes|-|
|[Optimizer.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.Optimizer.state_dict.html)|Yes|-|
|[Optimizer.step](https://pytorch.org/docs/2.9/generated/torch.optim.Optimizer.step.html)|Yes|-|
|[Optimizer.zero_grad](https://pytorch.org/docs/2.9/generated/torch.optim.Optimizer.zero_grad.html)|Yes|-|
|[torch.optim.Adadelta](https://pytorch.org/docs/2.9/generated/torch.optim.Adadelta.html)|Yes|Supports bf16, fp16, fp32<br>When the optimizer has foreach enabled (foreach=None or foreach=True) and there are too many parameter groups being optimized, performance degradation may occur due to the characteristics of the foreach operator. In this case, it is recommended to set foreach=False|
|[torch.optim.Adadelta.add_param_group](https://pytorch.org/docs/2.9/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.add_param_group)|Yes|-|
|[torch.optim.Adadelta.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.load_state_dict)|Yes|-|
|[torch.optim.Adadelta.register_load_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.register_load_state_dict_post_hook)|Yes|-|
|[torch.optim.Adadelta.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.register_load_state_dict_pre_hook)|Yes|-|
|[torch.optim.Adadelta.register_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.register_state_dict_post_hook)|Yes|-|
|[torch.optim.Adadelta.register_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.register_state_dict_pre_hook)|Yes|-|
|[torch.optim.Adadelta.register_step_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.register_step_post_hook)|Yes|-|
|[torch.optim.Adadelta.register_step_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.register_step_pre_hook)|Yes|-|
|[torch.optim.Adadelta.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.state_dict)|Yes|-|
|[torch.optim.Adadelta.step](https://pytorch.org/docs/2.9/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.step)|No|-|
|[torch.optim.Adadelta.zero_grad](https://pytorch.org/docs/2.9/generated/torch.optim.Adadelta.html#torch.optim.Adadelta.zero_grad)|Yes|-|
|[torch.optim.Adagrad](https://pytorch.org/docs/2.9/generated/torch.optim.Adagrad.html)|Yes|Supports bf16, fp16, fp32<br>When the optimizer has foreach enabled (foreach=None or foreach=True) and there are too many parameter groups being optimized, performance degradation may occur due to the characteristics of the foreach operator. In this case, it is recommended to set foreach=False|
|[torch.optim.Adagrad.add_param_group](https://pytorch.org/docs/2.9/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.add_param_group)|Yes|-|
|[torch.optim.Adagrad.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.load_state_dict)|Yes|-|
|[torch.optim.Adagrad.register_load_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.register_load_state_dict_post_hook)|Yes|-|
|[torch.optim.Adagrad.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.register_load_state_dict_pre_hook)|Yes|-|
|[torch.optim.Adagrad.register_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.register_state_dict_post_hook)|Yes|-|
|[torch.optim.Adagrad.register_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.register_state_dict_pre_hook)|Yes|-|
|[torch.optim.Adagrad.register_step_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.register_step_post_hook)|Yes|-|
|[torch.optim.Adagrad.register_step_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.register_step_pre_hook)|Yes|-|
|[torch.optim.Adagrad.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.state_dict)|Yes|-|
|[torch.optim.Adagrad.step](https://pytorch.org/docs/2.9/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.step)|Yes|-|
|[torch.optim.Adagrad.zero_grad](https://pytorch.org/docs/2.9/generated/torch.optim.Adagrad.html#torch.optim.Adagrad.zero_grad)|Yes|-|
|[torch.optim.Adam](https://pytorch.org/docs/2.9/generated/torch.optim.Adam.html)|Yes|Supports bf16, fp16, fp32<br>When the optimizer has foreach enabled (foreach=None or foreach=True) and there are too many parameter groups being optimized, performance degradation may occur due to the characteristics of the foreach operator. In this case, it is recommended to set foreach=False<br>May fall back to CPU execution in some cases|
|[torch.optim.Adam.add_param_group](https://pytorch.org/docs/2.9/generated/torch.optim.Adam.html#torch.optim.Adam.add_param_group)|Yes|-|
|[torch.optim.Adam.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.Adam.html#torch.optim.Adam.load_state_dict)|Yes|-|
|[torch.optim.Adam.register_load_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adam.html#torch.optim.Adam.register_load_state_dict_post_hook)|Yes|-|
|[torch.optim.Adam.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adam.html#torch.optim.Adam.register_load_state_dict_pre_hook)|Yes|-|
|[torch.optim.Adam.register_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adam.html#torch.optim.Adam.register_state_dict_post_hook)|Yes|-|
|[torch.optim.Adam.register_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adam.html#torch.optim.Adam.register_state_dict_pre_hook)|Yes|-|
|[torch.optim.Adam.register_step_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adam.html#torch.optim.Adam.register_step_post_hook)|Yes|-|
|[torch.optim.Adam.register_step_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adam.html#torch.optim.Adam.register_step_pre_hook)|Yes|-|
|[torch.optim.Adam.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.Adam.html#torch.optim.Adam.state_dict)|Yes|-|
|[torch.optim.Adam.step](https://pytorch.org/docs/2.9/generated/torch.optim.Adam.html#torch.optim.Adam.step)|No|-|
|[torch.optim.Adam.zero_grad](https://pytorch.org/docs/2.9/generated/torch.optim.Adam.html#torch.optim.Adam.zero_grad)|Yes|-|
|[torch.optim.AdamW](https://pytorch.org/docs/2.9/generated/torch.optim.AdamW.html)|Yes|Supports bf16, fp16, fp32, complex64<br>When the optimizer has foreach enabled (foreach=None or foreach=True) and there are too many parameter groups being optimized, performance degradation may occur due to the characteristics of the foreach operator. In this case, it is recommended to set foreach=False<br>When the optimizer has fused enabled (fused=True), the grad_scale and found_inf parameters are not yet supported. Aligned with the _single_tensor_adamw implementation, fp32 is consistent with CPU/CUDA, while fp16 and bf16 use higher precision implementation and are inconsistent with CPU/CUDA|
|[torch.optim.AdamW.add_param_group](https://pytorch.org/docs/2.9/generated/torch.optim.AdamW.html#torch.optim.AdamW.add_param_group)|Yes|-|
|[torch.optim.AdamW.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.AdamW.html#torch.optim.AdamW.load_state_dict)|Yes|-|
|[torch.optim.AdamW.register_load_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.AdamW.html#torch.optim.AdamW.register_load_state_dict_post_hook)|Yes|-|
|[torch.optim.AdamW.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.AdamW.html#torch.optim.AdamW.register_load_state_dict_pre_hook)|Yes|-|
|[torch.optim.AdamW.register_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.AdamW.html#torch.optim.AdamW.register_state_dict_post_hook)|Yes|-|
|[torch.optim.AdamW.register_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.AdamW.html#torch.optim.AdamW.register_state_dict_pre_hook)|Yes|-|
|[torch.optim.AdamW.register_step_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.AdamW.html#torch.optim.AdamW.register_step_post_hook)|Yes|-|
|[torch.optim.AdamW.register_step_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.AdamW.html#torch.optim.AdamW.register_step_pre_hook)|Yes|-|
|[torch.optim.AdamW.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.AdamW.html#torch.optim.AdamW.state_dict)|Yes|-|
|[torch.optim.AdamW.step](https://pytorch.org/docs/2.9/generated/torch.optim.AdamW.html#torch.optim.AdamW.step)|Yes|Supports fp16, fp32|
|[torch.optim.AdamW.zero_grad](https://pytorch.org/docs/2.9/generated/torch.optim.AdamW.html#torch.optim.AdamW.zero_grad)|Yes|Supports fp16, fp32|
|[torch.optim.SparseAdam.add_param_group](https://pytorch.org/docs/2.9/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.add_param_group)|Yes|-|
|[torch.optim.SparseAdam.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.load_state_dict)|Yes|-|
|[torch.optim.SparseAdam.register_load_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.register_load_state_dict_post_hook)|Yes|-|
|[torch.optim.SparseAdam.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.register_load_state_dict_pre_hook)|Yes|-|
|[torch.optim.SparseAdam.register_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.register_state_dict_post_hook)|Yes|-|
|[torch.optim.SparseAdam.register_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.register_state_dict_pre_hook)|Yes|-|
|[torch.optim.SparseAdam.register_step_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.register_step_post_hook)|Yes|-|
|[torch.optim.SparseAdam.register_step_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.register_step_pre_hook)|Yes|-|
|[torch.optim.SparseAdam.step](https://pytorch.org/docs/2.9/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.step)|-|-|
|[torch.optim.SparseAdam.zero_grad](https://pytorch.org/docs/2.9/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam.zero_grad)|-|-|
|[torch.optim.Adamax](https://pytorch.org/docs/2.9/generated/torch.optim.Adamax.html)|Yes|Supports bf16, fp16, fp32<br>When the optimizer has foreach enabled (foreach=None or foreach=True) and there are too many parameter groups being optimized, performance degradation may occur due to the characteristics of the foreach operator. In this case, it is recommended to set foreach=False|
|[torch.optim.Adamax.add_param_group](https://pytorch.org/docs/2.9/generated/torch.optim.Adamax.html#torch.optim.Adamax.add_param_group)|Yes|Supports fp16, fp32|
|[torch.optim.Adamax.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.Adamax.html#torch.optim.Adamax.load_state_dict)|Yes|Supports fp16, fp32|
|[torch.optim.Adamax.register_load_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adamax.html#torch.optim.Adamax.register_load_state_dict_post_hook)|Yes|-|
|[torch.optim.Adamax.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adamax.html#torch.optim.Adamax.register_load_state_dict_pre_hook)|Yes|-|
|[torch.optim.Adamax.register_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adamax.html#torch.optim.Adamax.register_state_dict_post_hook)|Yes|-|
|[torch.optim.Adamax.register_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adamax.html#torch.optim.Adamax.register_state_dict_pre_hook)|Yes|-|
|[torch.optim.Adamax.register_step_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adamax.html#torch.optim.Adamax.register_step_post_hook)|Yes|-|
|[torch.optim.Adamax.register_step_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Adamax.html#torch.optim.Adamax.register_step_pre_hook)|Yes|-|
|[torch.optim.Adamax.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.Adamax.html#torch.optim.Adamax.state_dict)|Yes|Supports fp16, fp32|
|[torch.optim.Adamax.step](https://pytorch.org/docs/2.9/generated/torch.optim.Adamax.html#torch.optim.Adamax.step)|Yes|Supports fp16, fp32|
|[torch.optim.Adamax.zero_grad](https://pytorch.org/docs/2.9/generated/torch.optim.Adamax.html#torch.optim.Adamax.zero_grad)|Yes|Supports fp16, fp32|
|[torch.optim.ASGD](https://pytorch.org/docs/2.9/generated/torch.optim.ASGD.html)|Yes|Supports fp16, fp32|
|[torch.optim.ASGD.add_param_group](https://pytorch.org/docs/2.9/generated/torch.optim.ASGD.html#torch.optim.ASGD.add_param_group)|Yes|Supports fp16, fp32|
|[torch.optim.ASGD.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.ASGD.html#torch.optim.ASGD.load_state_dict)|Yes|Supports fp16, fp32|
|[torch.optim.ASGD.register_load_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.ASGD.html#torch.optim.ASGD.register_load_state_dict_post_hook)|Yes|-|
|[torch.optim.ASGD.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.ASGD.html#torch.optim.ASGD.register_load_state_dict_pre_hook)|Yes|-|
|[torch.optim.ASGD.register_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.ASGD.html#torch.optim.ASGD.register_state_dict_post_hook)|Yes|-|
|[torch.optim.ASGD.register_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.ASGD.html#torch.optim.ASGD.register_state_dict_pre_hook)|Yes|-|
|[torch.optim.ASGD.register_step_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.ASGD.html#torch.optim.ASGD.register_step_post_hook)|Yes|-|
|[torch.optim.ASGD.register_step_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.ASGD.html#torch.optim.ASGD.register_step_pre_hook)|Yes|-|
|[torch.optim.ASGD.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.ASGD.html#torch.optim.ASGD.state_dict)|Yes|Supports fp16, fp32|
|[torch.optim.ASGD.step](https://pytorch.org/docs/2.9/generated/torch.optim.ASGD.html#torch.optim.ASGD.step)|Yes|Supports fp16, fp32|
|[torch.optim.ASGD.zero_grad](https://pytorch.org/docs/2.9/generated/torch.optim.ASGD.html#torch.optim.ASGD.zero_grad)|Yes|Supports fp16, fp32|
|[torch.optim.LBFGS](https://pytorch.org/docs/2.9/generated/torch.optim.LBFGS.html)|Yes|-|
|[torch.optim.LBFGS.add_param_group](https://pytorch.org/docs/2.9/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.add_param_group)|Yes|-|
|[torch.optim.LBFGS.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.load_state_dict)|Yes|-|
|[torch.optim.LBFGS.register_load_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.register_load_state_dict_post_hook)|Yes|-|
|[torch.optim.LBFGS.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.register_load_state_dict_pre_hook)|Yes|-|
|[torch.optim.LBFGS.register_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.register_state_dict_post_hook)|Yes|-|
|[torch.optim.LBFGS.register_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.register_state_dict_pre_hook)|Yes|-|
|[torch.optim.LBFGS.register_step_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.register_step_post_hook)|Yes|-|
|[torch.optim.LBFGS.register_step_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.register_step_pre_hook)|Yes|-|
|[torch.optim.LBFGS.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.state_dict)|Yes|-|
|[torch.optim.LBFGS.step](https://pytorch.org/docs/2.9/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.step)|No|-|
|[torch.optim.LBFGS.zero_grad](https://pytorch.org/docs/2.9/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.zero_grad)|Yes|-|
|[torch.optim.NAdam](https://pytorch.org/docs/2.9/generated/torch.optim.NAdam.html)|Yes|Supports bf16, fp16, fp32<br>When the optimizer has foreach enabled (foreach=None or foreach=True), performance degradation may occur if there are too many parameter groups being optimized due to the characteristics of the foreach operator. In this case, it is recommended to set foreach=False|
|[torch.optim.NAdam.add_param_group](https://pytorch.org/docs/2.9/generated/torch.optim.NAdam.html#torch.optim.NAdam.add_param_group)|Yes|Supports fp16, fp32|
|[torch.optim.NAdam.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.NAdam.html#torch.optim.NAdam.load_state_dict)|Yes|Supports fp16, fp32|
|[torch.optim.NAdam.register_load_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.NAdam.html#torch.optim.NAdam.register_load_state_dict_post_hook)|Yes|-|
|[torch.optim.NAdam.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.NAdam.html#torch.optim.NAdam.register_load_state_dict_pre_hook)|Yes|-|
|[torch.optim.NAdam.register_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.NAdam.html#torch.optim.NAdam.register_state_dict_post_hook)|Yes|-|
|[torch.optim.NAdam.register_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.NAdam.html#torch.optim.NAdam.register_state_dict_pre_hook)|Yes|-|
|[torch.optim.NAdam.register_step_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.NAdam.html#torch.optim.NAdam.register_step_post_hook)|Yes|-|
|[torch.optim.NAdam.register_step_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.NAdam.html#torch.optim.NAdam.register_step_pre_hook)|Yes|-|
|[torch.optim.NAdam.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.NAdam.html#torch.optim.NAdam.state_dict)|Yes|Supports fp16, fp32|
|[torch.optim.NAdam.step](https://pytorch.org/docs/2.9/generated/torch.optim.NAdam.html#torch.optim.NAdam.step)|Yes|Supports fp16, fp32|
|[torch.optim.NAdam.zero_grad](https://pytorch.org/docs/2.9/generated/torch.optim.NAdam.html#torch.optim.NAdam.zero_grad)|Yes|Supports fp16, fp32|
|[torch.optim.RAdam](https://pytorch.org/docs/2.9/generated/torch.optim.RAdam.html)|Yes|Supports bf16, fp16, fp32<br>When the optimizer has foreach enabled (foreach=None or foreach=True), performance degradation may occur if there are too many parameter groups being optimized due to the characteristics of the foreach operator. In this case, it is recommended to set foreach=False|
|[torch.optim.RAdam.add_param_group](https://pytorch.org/docs/2.9/generated/torch.optim.RAdam.html#torch.optim.RAdam.add_param_group)|Yes|Supports fp16, fp32|
|[torch.optim.RAdam.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.RAdam.html#torch.optim.RAdam.load_state_dict)|Yes|Supports fp16, fp32|
|[torch.optim.RAdam.register_load_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.RAdam.html#torch.optim.RAdam.register_load_state_dict_post_hook)|Yes|-|
|[torch.optim.RAdam.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.RAdam.html#torch.optim.RAdam.register_load_state_dict_pre_hook)|Yes|-|
|[torch.optim.RAdam.register_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.RAdam.html#torch.optim.RAdam.register_state_dict_post_hook)|Yes|-|
|[torch.optim.RAdam.register_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.RAdam.html#torch.optim.RAdam.register_state_dict_pre_hook)|Yes|-|
|[torch.optim.RAdam.register_step_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.RAdam.html#torch.optim.RAdam.register_step_post_hook)|Yes|-|
|[torch.optim.RAdam.register_step_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.RAdam.html#torch.optim.RAdam.register_step_pre_hook)|Yes|-|
|[torch.optim.RAdam.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.RAdam.html#torch.optim.RAdam.state_dict)|Yes|Supports fp16, fp32|
|[torch.optim.RAdam.step](https://pytorch.org/docs/2.9/generated/torch.optim.RAdam.html#torch.optim.RAdam.step)|Yes|Supports fp16, fp32|
|[torch.optim.RAdam.zero_grad](https://pytorch.org/docs/2.9/generated/torch.optim.RAdam.html#torch.optim.RAdam.zero_grad)|Yes|Supports fp16, fp32|
|[torch.optim.RMSprop](https://pytorch.org/docs/2.9/generated/torch.optim.RMSprop.html)|Yes|Supports bf16, fp16, fp32<br>When the optimizer has foreach enabled (foreach=None or foreach=True), performance degradation may occur if there are too many parameter groups being optimized due to the characteristics of the foreach operator. In this case, it is recommended to set foreach=False|
|[torch.optim.RMSprop.add_param_group](https://pytorch.org/docs/2.9/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.add_param_group)|Yes|-|
|[torch.optim.RMSprop.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.load_state_dict)|Yes|-|
|[torch.optim.RMSprop.register_load_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.register_load_state_dict_post_hook)|Yes|-|
|[torch.optim.RMSprop.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.register_load_state_dict_pre_hook)|Yes|-|
|[torch.optim.RMSprop.register_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.register_state_dict_post_hook)|Yes|-|
|[torch.optim.RMSprop.register_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.register_state_dict_pre_hook)|Yes|-|
|[torch.optim.RMSprop.register_step_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.register_step_post_hook)|Yes|-|
|[torch.optim.RMSprop.register_step_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.register_step_pre_hook)|Yes|-|
|[torch.optim.RMSprop.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.state_dict)|Yes|-|
|[torch.optim.RMSprop.step](https://pytorch.org/docs/2.9/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.step)|Yes|-|
|[torch.optim.RMSprop.zero_grad](https://pytorch.org/docs/2.9/generated/torch.optim.RMSprop.html#torch.optim.RMSprop.zero_grad)|Yes|-|
|[torch.optim.Rprop](https://pytorch.org/docs/2.9/generated/torch.optim.Rprop.html)|Yes|-|
|[torch.optim.Rprop.add_param_group](https://pytorch.org/docs/2.9/generated/torch.optim.Rprop.html#torch.optim.Rprop.add_param_group)|Yes|Supports fp16, fp32|
|[torch.optim.Rprop.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.Rprop.html#torch.optim.Rprop.load_state_dict)|Yes|Supports fp16, fp32|
|[torch.optim.Rprop.register_load_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Rprop.html#torch.optim.Rprop.register_load_state_dict_post_hook)|Yes|-|
|[torch.optim.Rprop.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Rprop.html#torch.optim.Rprop.register_load_state_dict_pre_hook)|Yes|-|
|[torch.optim.Rprop.register_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Rprop.html#torch.optim.Rprop.register_state_dict_post_hook)|Yes|-|
|[torch.optim.Rprop.register_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Rprop.html#torch.optim.Rprop.register_state_dict_pre_hook)|Yes|-|
|[torch.optim.Rprop.register_step_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Rprop.html#torch.optim.Rprop.register_step_post_hook)|Yes|-|
|[torch.optim.Rprop.register_step_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.Rprop.html#torch.optim.Rprop.register_step_pre_hook)|Yes|-|
|[torch.optim.Rprop.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.Rprop.html#torch.optim.Rprop.state_dict)|Yes|Supports fp16, fp32|
|[torch.optim.Rprop.step](https://pytorch.org/docs/2.9/generated/torch.optim.Rprop.html#torch.optim.Rprop.step)|Yes|Supports fp16, fp32|
|[torch.optim.Rprop.zero_grad](https://pytorch.org/docs/2.9/generated/torch.optim.Rprop.html#torch.optim.Rprop.zero_grad)|Yes|Supports fp16, fp32|
|[torch.optim.SGD](https://pytorch.org/docs/2.9/generated/torch.optim.SGD.html)|Yes|Supports bf16, fp16, fp32<br>When the optimizer has foreach enabled (foreach=None or foreach=True), performance degradation may occur if there are too many parameter groups being optimized due to the characteristics of the foreach operator. In this case, it is recommended to set foreach=False|
|[torch.optim.SGD.add_param_group](https://pytorch.org/docs/2.9/generated/torch.optim.SGD.html#torch.optim.SGD.add_param_group)|Yes|Supports fp16, fp32|
|[torch.optim.SGD.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.SGD.html#torch.optim.SGD.load_state_dict)|Yes|Supports fp16, fp32|
|[torch.optim.SGD.register_load_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.SGD.html#torch.optim.SGD.register_load_state_dict_post_hook)|Yes|-|
|[torch.optim.SGD.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.SGD.html#torch.optim.SGD.register_load_state_dict_pre_hook)|Yes|-|
|[torch.optim.SGD.register_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.SGD.html#torch.optim.SGD.register_state_dict_post_hook)|Yes|-|
|[torch.optim.SGD.register_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.SGD.html#torch.optim.SGD.register_state_dict_pre_hook)|Yes|-|
|[torch.optim.SGD.register_step_post_hook](https://pytorch.org/docs/2.9/generated/torch.optim.SGD.html#torch.optim.SGD.register_step_post_hook)|Yes|-|
|[torch.optim.SGD.register_step_pre_hook](https://pytorch.org/docs/2.9/generated/torch.optim.SGD.html#torch.optim.SGD.register_step_pre_hook)|Yes|-|
|[torch.optim.SGD.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.SGD.html#torch.optim.SGD.state_dict)|Yes|Supports fp16, fp32|
|[torch.optim.SGD.step](https://pytorch.org/docs/2.9/generated/torch.optim.SGD.html#torch.optim.SGD.step)|Yes|Supports fp16, fp32|
|[torch.optim.SGD.zero_grad](https://pytorch.org/docs/2.9/generated/torch.optim.SGD.html#torch.optim.SGD.zero_grad)|Yes|Supports fp16, fp32|
|[torch.optim.lr_scheduler.LambdaLR](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.LambdaLR.html)|Yes|-|
|[torch.optim.lr_scheduler.LambdaLR.get_last_lr](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR.get_last_lr)|Yes|-|
|[torch.optim.lr_scheduler.LambdaLR.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR.load_state_dict)|Yes|-|
|[torch.optim.lr_scheduler.LambdaLR.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR.state_dict)|Yes|-|
|[torch.optim.lr_scheduler.MultiplicativeLR](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.MultiplicativeLR.html)|Yes|-|
|[torch.optim.lr_scheduler.MultiplicativeLR.get_last_lr](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.MultiplicativeLR.html#torch.optim.lr_scheduler.MultiplicativeLR.get_last_lr)|Yes|Supports fp32|
|[torch.optim.lr_scheduler.MultiplicativeLR.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.MultiplicativeLR.html#torch.optim.lr_scheduler.MultiplicativeLR.load_state_dict)|Yes|Supports fp32|
|[torch.optim.lr_scheduler.MultiplicativeLR.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.MultiplicativeLR.html#torch.optim.lr_scheduler.MultiplicativeLR.state_dict)|Yes|Supports fp32|
|[torch.optim.lr_scheduler.StepLR](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.StepLR.html)|Yes|-|
|[torch.optim.lr_scheduler.StepLR.get_last_lr](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR.get_last_lr)|Yes|Supports fp16, fp32|
|[torch.optim.lr_scheduler.StepLR.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR.load_state_dict)|Yes|Supports fp16, fp32|
|[torch.optim.lr_scheduler.StepLR.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR.state_dict)|Yes|Supports fp16, fp32|
|[torch.optim.lr_scheduler.MultiStepLR](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.MultiStepLR.html)|Yes|-|
|[torch.optim.lr_scheduler.MultiStepLR.get_last_lr](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR.get_last_lr)|Yes|Supports fp16, fp32|
|[torch.optim.lr_scheduler.MultiStepLR.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR.load_state_dict)|Yes|Supports fp16, fp32|
|[torch.optim.lr_scheduler.MultiStepLR.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR.state_dict)|Yes|Supports fp16, fp32|
|[torch.optim.lr_scheduler.ConstantLR](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.ConstantLR.html)|Yes|-|
|[torch.optim.lr_scheduler.ConstantLR.get_last_lr](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.ConstantLR.html#torch.optim.lr_scheduler.ConstantLR.get_last_lr)|Yes|Supports fp32|
|[torch.optim.lr_scheduler.ConstantLR.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.ConstantLR.html#torch.optim.lr_scheduler.ConstantLR.load_state_dict)|Yes|Supports fp32|
|[torch.optim.lr_scheduler.ConstantLR.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.ConstantLR.html#torch.optim.lr_scheduler.ConstantLR.state_dict)|Yes|Supports fp32|
|[torch.optim.lr_scheduler.LinearLR](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.LinearLR.html)|Yes|-|
|[torch.optim.lr_scheduler.LinearLR.get_last_lr](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR.get_last_lr)|Yes|-|
|[torch.optim.lr_scheduler.LinearLR.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR.load_state_dict)|Yes|-|
|[torch.optim.lr_scheduler.LinearLR.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR.state_dict)|Yes|-|
|[torch.optim.lr_scheduler.ExponentialLR](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.ExponentialLR.html)|Yes|-|
|[torch.optim.lr_scheduler.ExponentialLR.get_last_lr](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR.get_last_lr)|Yes|-|
|[torch.optim.lr_scheduler.ExponentialLR.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR.load_state_dict)|Yes|-|
|[torch.optim.lr_scheduler.ExponentialLR.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR.state_dict)|Yes|-|
|[torch.optim.lr_scheduler.PolynomialLR](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.PolynomialLR.html)|Yes|-|
|[torch.optim.lr_scheduler.PolynomialLR.get_last_lr](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.PolynomialLR.html#torch.optim.lr_scheduler.PolynomialLR.get_last_lr)|Yes|-|
|[torch.optim.lr_scheduler.PolynomialLR.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.PolynomialLR.html#torch.optim.lr_scheduler.PolynomialLR.load_state_dict)|Yes|-|
|[torch.optim.lr_scheduler.PolynomialLR.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.PolynomialLR.html#torch.optim.lr_scheduler.PolynomialLR.state_dict)|Yes|-|
|[torch.optim.lr_scheduler.CosineAnnealingLR](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)|Yes|-|
|[torch.optim.lr_scheduler.CosineAnnealingLR.get_last_lr](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR.get_last_lr)|Yes|-|
|[torch.optim.lr_scheduler.CosineAnnealingLR.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR.load_state_dict)|Yes|-|
|[torch.optim.lr_scheduler.CosineAnnealingLR.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR.state_dict)|Yes|-|
|[torch.optim.lr_scheduler.ChainedScheduler](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.ChainedScheduler.html)|Yes|-|
|[torch.optim.lr_scheduler.ChainedScheduler.get_last_lr](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.ChainedScheduler.html#torch.optim.lr_scheduler.ChainedScheduler.get_last_lr)|Yes|-|
|[torch.optim.lr_scheduler.ChainedScheduler.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.ChainedScheduler.html#torch.optim.lr_scheduler.ChainedScheduler.load_state_dict)|Yes|-|
|[torch.optim.lr_scheduler.ChainedScheduler.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.ChainedScheduler.html#torch.optim.lr_scheduler.ChainedScheduler.state_dict)|Yes|-|
|[torch.optim.lr_scheduler.SequentialLR](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.SequentialLR.html)|Yes|-|
|[torch.optim.lr_scheduler.SequentialLR.get_last_lr](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.SequentialLR.html#torch.optim.lr_scheduler.SequentialLR.get_last_lr)|Yes|-|
|[torch.optim.lr_scheduler.SequentialLR.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.SequentialLR.html#torch.optim.lr_scheduler.SequentialLR.load_state_dict)|Yes|-|
|[torch.optim.lr_scheduler.SequentialLR.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.SequentialLR.html#torch.optim.lr_scheduler.SequentialLR.state_dict)|Yes|-|
|[torch.optim.lr_scheduler.ReduceLROnPlateau](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)|Yes|-|
|[torch.optim.lr_scheduler.CyclicLR](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.CyclicLR.html)|Yes|-|
|[torch.optim.lr_scheduler.CyclicLR.get_last_lr](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR.get_last_lr)|Yes|-|
|[torch.optim.lr_scheduler.CyclicLR.get_lr](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR.get_lr)|Yes|-|
|[torch.optim.lr_scheduler.OneCycleLR](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.OneCycleLR.html)|Yes|-|
|[torch.optim.lr_scheduler.OneCycleLR.get_last_lr](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR.get_last_lr)|Yes|-|
|[torch.optim.lr_scheduler.OneCycleLR.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR.load_state_dict)|Yes|-|
|[torch.optim.lr_scheduler.OneCycleLR.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR.state_dict)|Yes|-|
|[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html)|Yes|-|
|[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.get_last_lr](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.get_last_lr)|Yes|-|
|[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.load_state_dict)|Yes|-|
|[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.state_dict](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.state_dict)|Yes|-|
|[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.step](https://pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.step)|Yes|-|
