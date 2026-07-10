# torch.optim

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-09T08:33:03.526Z pushedAt=2026-07-09T08:44:08.304Z -->

> [!NOTE]
> If an API's "Supported" column is "Yes" and "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.optim.Optimizer|Yes|-|
|Optimizer.add_param_group|Yes|-|
|Optimizer.load_state_dict|Yes|-|
|Optimizer.state_dict|Yes|-|
|Optimizer.step|Yes|-|
|Optimizer.zero_grad|Yes|-|
|torch.optim.Adadelta|Yes|Supports bf16, fp16, fp32<br>When the optimizer is launched with foreach (foreach=None or foreach=True), performance degradation may occur if there are too many parameter groups being optimized due to the characteristics of the foreach operator. In such cases, it is recommended to set foreach=False|
|torch.optim.Adadelta.add_param_group|Yes|-|
|torch.optim.Adadelta.load_state_dict|Yes|-|
|torch.optim.Adadelta.register_load_state_dict_post_hook|Yes|-|
|torch.optim.Adadelta.register_load_state_dict_pre_hook|Yes|-|
|torch.optim.Adadelta.register_state_dict_post_hook|Yes|-|
|torch.optim.Adadelta.register_state_dict_pre_hook|Yes|-|
|torch.optim.Adadelta.register_step_post_hook|Yes|-|
|torch.optim.Adadelta.register_step_pre_hook|Yes|-|
|torch.optim.Adadelta.state_dict|Yes|-|
|torch.optim.Adadelta.step|No|-|
|torch.optim.Adadelta.zero_grad|Yes|-|
|torch.optim.Adagrad|Yes|Supports bf16, fp16, fp32<br>When the optimizer is launched with foreach (foreach=None or foreach=True), performance degradation may occur if there are too many parameter groups being optimized due to the characteristics of the foreach operator. In such cases, it is recommended to set foreach=False|
|torch.optim.Adagrad.add_param_group|Yes|-|
|torch.optim.Adagrad.load_state_dict|Yes|-|
|torch.optim.Adagrad.register_load_state_dict_post_hook|Yes|-|
|torch.optim.Adagrad.register_load_state_dict_pre_hook|Yes|-|
|torch.optim.Adagrad.register_state_dict_post_hook|Yes|-|
|torch.optim.Adagrad.register_state_dict_pre_hook|Yes|-|
|torch.optim.Adagrad.register_step_post_hook|Yes|-|
|torch.optim.Adagrad.register_step_pre_hook|Yes|-|
|torch.optim.Adagrad.state_dict|Yes|-|
|torch.optim.Adagrad.step|Yes|-|
|torch.optim.Adagrad.zero_grad|Yes|-|
|torch.optim.Adam|Yes|Supports bf16, fp16, fp32<br>When the optimizer is launched with foreach (foreach=None or foreach=True), performance degradation may occur if there are too many parameter groups being optimized due to the characteristics of the foreach operator. In such cases, it is recommended to set foreach=False<br>May fall back to CPU execution in some cases|
|torch.optim.Adam.add_param_group|Yes|-|
|torch.optim.Adam.load_state_dict|Yes|-|
|torch.optim.Adam.register_load_state_dict_post_hook|Yes|-|
|torch.optim.Adam.register_load_state_dict_pre_hook|Yes|-|
|torch.optim.Adam.register_state_dict_post_hook|Yes|-|
|torch.optim.Adam.register_state_dict_pre_hook|Yes|-|
|torch.optim.Adam.register_step_post_hook|Yes|-|
|torch.optim.Adam.register_step_pre_hook|Yes|-|
|torch.optim.Adam.state_dict|Yes|-|
|torch.optim.Adam.step|No|-|
|torch.optim.Adam.zero_grad|Yes|-|
|torch.optim.AdamW|Yes|Supports bf16, fp16, fp32, complex64<br>When the optimizer is launched with foreach (foreach=None or foreach=True), performance degradation may occur if there are too many parameter groups being optimized due to the characteristics of the foreach operator. In such cases, it is recommended to set foreach=False<br>When the optimizer is launched with fused (fused=True), the grad_scale and found_inf parameters are not yet supported. Aligned with the _single_tensor_adamw implementation, fp32 is consistent with cpu/cuda, while fp16 and bf16 use upcast implementation and are inconsistent with cpu/cuda|
|torch.optim.AdamW.add_param_group|Yes|-|
|torch.optim.AdamW.load_state_dict|Yes|-|
|torch.optim.AdamW.register_load_state_dict_post_hook|Yes|-|
|torch.optim.AdamW.register_load_state_dict_pre_hook|Yes|-|
|torch.optim.AdamW.register_state_dict_post_hook|Yes|-|
|torch.optim.AdamW.register_state_dict_pre_hook|Yes|-|
|torch.optim.AdamW.register_step_post_hook|Yes|-|
|torch.optim.AdamW.register_step_pre_hook|Yes|-|
|torch.optim.AdamW.state_dict|Yes|-|
|torch.optim.AdamW.step|Yes|Supports fp16, fp32|
|torch.optim.AdamW.zero_grad|Yes|Supports fp16, fp32|
|torch.optim.SparseAdam.add_param_group|Yes|-|
|torch.optim.SparseAdam.load_state_dict|Yes|-|
|torch.optim.SparseAdam.register_load_state_dict_post_hook|Yes|-|
|torch.optim.SparseAdam.register_load_state_dict_pre_hook|Yes|-|
|torch.optim.SparseAdam.register_state_dict_post_hook|Yes|-|
|torch.optim.SparseAdam.register_state_dict_pre_hook|Yes|-|
|torch.optim.SparseAdam.register_step_post_hook|Yes|-|
|torch.optim.SparseAdam.register_step_pre_hook|Yes|-|
|torch.optim.SparseAdam.step|-|-|
|torch.optim.SparseAdam.zero_grad|-|-|
|torch.optim.Adamax|Yes|Supports bf16, fp16, fp32<br>When the optimizer is launched with foreach (foreach=None or foreach=True), performance degradation may occur if there are too many parameter groups being optimized due to the characteristics of the foreach operator. In such cases, it is recommended to set foreach=False|
|torch.optim.Adamax.add_param_group|Yes|Supports fp16, fp32|
|torch.optim.Adamax.load_state_dict|Yes|Supports fp16, fp32|
|torch.optim.Adamax.register_load_state_dict_post_hook|Yes|-|
|torch.optim.Adamax.register_load_state_dict_pre_hook|Yes|-|
|torch.optim.Adamax.register_state_dict_post_hook|Yes|-|
|torch.optim.Adamax.register_state_dict_pre_hook|Yes|-|
|torch.optim.Adamax.register_step_post_hook|Yes|-|
|torch.optim.Adamax.register_step_pre_hook|Yes|-|
|torch.optim.Adamax.state_dict|Yes|Supports fp16, fp32|
|torch.optim.Adamax.step|Yes|Supports fp16, fp32|
|torch.optim.Adamax.zero_grad|Yes|Supports fp16, fp32|
|torch.optim.ASGD|Yes|Supports fp16, fp32|
|torch.optim.ASGD.add_param_group|Yes|Supports fp16, fp32|
|torch.optim.ASGD.load_state_dict|Yes|Supports fp16, fp32|
|torch.optim.ASGD.register_load_state_dict_post_hook|Yes|-|
|torch.optim.ASGD.register_load_state_dict_pre_hook|Yes|-|
|torch.optim.ASGD.register_state_dict_post_hook|Yes|-|
|torch.optim.ASGD.register_state_dict_pre_hook|Yes|-|
|torch.optim.ASGD.register_step_post_hook|Yes|-|
|torch.optim.ASGD.register_step_pre_hook|Yes|-|
|torch.optim.ASGD.state_dict|Yes|Supports fp16, fp32|
|torch.optim.ASGD.step|Yes|Supports fp16, fp32|
|torch.optim.ASGD.zero_grad|Yes|Supports fp16, fp32|
|torch.optim.LBFGS|Yes|-|
|torch.optim.LBFGS.add_param_group|Yes|-|
|torch.optim.LBFGS.load_state_dict|Yes|-|
|torch.optim.LBFGS.register_load_state_dict_post_hook|Yes|-|
|torch.optim.LBFGS.register_load_state_dict_pre_hook|Yes|-|
|torch.optim.LBFGS.register_state_dict_post_hook|Yes|-|
|torch.optim.LBFGS.register_state_dict_pre_hook|Yes|-|
|torch.optim.LBFGS.register_step_post_hook|Yes|-|
|torch.optim.LBFGS.register_step_pre_hook|Yes|-|
|torch.optim.LBFGS.state_dict|Yes|-|
|torch.optim.LBFGS.step|No|-|
|torch.optim.LBFGS.zero_grad|Yes|-|
|torch.optim.NAdam|Yes|Supports bf16, fp16, fp32<br>When the optimizer has foreach enabled (foreach=None or foreach=True), performance degradation may occur if there are too many parameter groups being optimized due to the characteristics of the foreach operator. In such cases, it is recommended to set foreach=False|
|torch.optim.NAdam.add_param_group|Yes|Supports fp16, fp32|
|torch.optim.NAdam.load_state_dict|Yes|Supports fp16, fp32|
|torch.optim.NAdam.register_load_state_dict_post_hook|Yes|-|
|torch.optim.NAdam.register_load_state_dict_pre_hook|Yes|-|
|torch.optim.NAdam.register_state_dict_post_hook|Yes|-|
|torch.optim.NAdam.register_state_dict_pre_hook|Yes|-|
|torch.optim.NAdam.register_step_post_hook|Yes|-|
|torch.optim.NAdam.register_step_pre_hook|Yes|-|
|torch.optim.NAdam.state_dict|Yes|Supports fp16, fp32|
|torch.optim.NAdam.step|Yes|Supports fp16, fp32|
|torch.optim.NAdam.zero_grad|Yes|Supports fp16, fp32|
|torch.optim.RAdam|Yes|Supports bf16, fp16, fp32<br>When the optimizer has foreach enabled (foreach=None or foreach=True), performance degradation may occur if there are too many parameter groups being optimized due to the characteristics of the foreach operator. In such cases, it is recommended to set foreach=False|
|torch.optim.RAdam.add_param_group|Yes|Supports fp16, fp32|
|torch.optim.RAdam.load_state_dict|Yes|Supports fp16, fp32|
|torch.optim.RAdam.register_load_state_dict_post_hook|Yes|-|
|torch.optim.RAdam.register_load_state_dict_pre_hook|Yes|-|
|torch.optim.RAdam.register_state_dict_post_hook|Yes|-|
|torch.optim.RAdam.register_state_dict_pre_hook|Yes|-|
|torch.optim.RAdam.register_step_post_hook|Yes|-|
|torch.optim.RAdam.register_step_pre_hook|Yes|-|
|torch.optim.RAdam.state_dict|Yes|Supports fp16, fp32|
|torch.optim.RAdam.step|Yes|Supports fp16, fp32|
|torch.optim.RAdam.zero_grad|Yes|Supports fp16, fp32|
|torch.optim.RMSprop|Yes|Supports bf16, fp16, fp32<br>When the optimizer has foreach enabled (foreach=None or foreach=True), performance degradation may occur if there are too many parameter groups being optimized due to the characteristics of the foreach operator. In such cases, it is recommended to set foreach=False|
|torch.optim.RMSprop.add_param_group|Yes|-|
|torch.optim.RMSprop.load_state_dict|Yes|-|
|torch.optim.RMSprop.register_load_state_dict_post_hook|Yes|-|
|torch.optim.RMSprop.register_load_state_dict_pre_hook|Yes|-|
|torch.optim.RMSprop.register_state_dict_post_hook|Yes|-|
|torch.optim.RMSprop.register_state_dict_pre_hook|Yes|-|
|torch.optim.RMSprop.register_step_post_hook|Yes|-|
|torch.optim.RMSprop.register_step_pre_hook|Yes|-|
|torch.optim.RMSprop.state_dict|Yes|-|
|torch.optim.RMSprop.step|Yes|-|
|torch.optim.RMSprop.zero_grad|Yes|-|
|torch.optim.Rprop|Yes|-|
|torch.optim.Rprop.add_param_group|Yes|Supports fp16, fp32|
|torch.optim.Rprop.load_state_dict|Yes|Supports fp16, fp32|
|torch.optim.Rprop.register_load_state_dict_post_hook|Yes|-|
|torch.optim.Rprop.register_load_state_dict_pre_hook|Yes|-|
|torch.optim.Rprop.register_state_dict_post_hook|Yes|-|
|torch.optim.Rprop.register_state_dict_pre_hook|Yes|-|
|torch.optim.Rprop.register_step_post_hook|Yes|-|
|torch.optim.Rprop.register_step_pre_hook|Yes|-|
|torch.optim.Rprop.state_dict|Yes|Supports fp16, fp32|
|torch.optim.Rprop.step|Yes|Supports fp16, fp32|
|torch.optim.Rprop.zero_grad|Yes|Supports fp16, fp32|
|torch.optim.SGD|Yes|Supports bf16, fp16, fp32<br>When the optimizer has foreach enabled (foreach=None or foreach=True), performance degradation may occur if there are too many parameter groups being optimized due to the characteristics of the foreach operator. In such cases, it is recommended to set foreach=False|
|torch.optim.SGD.add_param_group|Yes|Supports fp16, fp32|
|torch.optim.SGD.load_state_dict|Yes|Supports fp16, fp32|
|torch.optim.SGD.register_load_state_dict_post_hook|Yes|-|
|torch.optim.SGD.register_load_state_dict_pre_hook|Yes|-|
|torch.optim.SGD.register_state_dict_post_hook|Yes|-|
|torch.optim.SGD.register_state_dict_pre_hook|Yes|-|
|torch.optim.SGD.register_step_post_hook|Yes|-|
|torch.optim.SGD.register_step_pre_hook|Yes|-|
|torch.optim.SGD.state_dict|Yes|Supports fp16, fp32|
|torch.optim.SGD.step|Yes|Supports fp16, fp32|
|torch.optim.SGD.zero_grad|Yes|Supports fp16, fp32|
|torch.optim.lr_scheduler.LambdaLR|Yes|-|
|torch.optim.lr_scheduler.LambdaLR.get_last_lr|Yes|-|
|torch.optim.lr_scheduler.LambdaLR.load_state_dict|Yes|-|
|torch.optim.lr_scheduler.LambdaLR.state_dict|Yes|-|
|torch.optim.lr_scheduler.MultiplicativeLR|Yes|-|
|torch.optim.lr_scheduler.MultiplicativeLR.get_last_lr|Yes|Supports fp32|
|torch.optim.lr_scheduler.MultiplicativeLR.load_state_dict|Yes|Supports fp32|
|torch.optim.lr_scheduler.MultiplicativeLR.state_dict|Yes|Supports fp32|
|torch.optim.lr_scheduler.StepLR|Yes|-|
|torch.optim.lr_scheduler.StepLR.get_last_lr|Yes|Supports fp16, fp32|
|torch.optim.lr_scheduler.StepLR.load_state_dict|Yes|Supports fp16, fp32|
|torch.optim.lr_scheduler.StepLR.state_dict|Yes|Supports fp16, fp32|
|torch.optim.lr_scheduler.MultiStepLR|Yes|-|
|torch.optim.lr_scheduler.MultiStepLR.get_last_lr|Yes|Supports fp16, fp32|
|torch.optim.lr_scheduler.MultiStepLR.load_state_dict|Yes|Supports fp16, fp32|
|torch.optim.lr_scheduler.MultiStepLR.state_dict|Yes|Supports fp16, fp32|
|torch.optim.lr_scheduler.ConstantLR|Yes|-|
|torch.optim.lr_scheduler.ConstantLR.get_last_lr|Yes|Supports fp32|
|torch.optim.lr_scheduler.ConstantLR.load_state_dict|Yes|Supports fp32|
|torch.optim.lr_scheduler.ConstantLR.state_dict|Yes|Supports fp32|
|torch.optim.lr_scheduler.LinearLR|Yes|-|
|torch.optim.lr_scheduler.LinearLR.get_last_lr|Yes|-|
|torch.optim.lr_scheduler.LinearLR.load_state_dict|Yes|-|
|torch.optim.lr_scheduler.LinearLR.state_dict|Yes|-|
|torch.optim.lr_scheduler.ExponentialLR|Yes|-|
|torch.optim.lr_scheduler.ExponentialLR.get_last_lr|Yes|-|
|torch.optim.lr_scheduler.ExponentialLR.load_state_dict|Yes|-|
|torch.optim.lr_scheduler.ExponentialLR.state_dict|Yes|-|
|torch.optim.lr_scheduler.PolynomialLR|Yes|-|
|torch.optim.lr_scheduler.PolynomialLR.get_last_lr|Yes|-|
|torch.optim.lr_scheduler.PolynomialLR.load_state_dict|Yes|-|
|torch.optim.lr_scheduler.PolynomialLR.state_dict|Yes|-|
|torch.optim.lr_scheduler.CosineAnnealingLR|Yes|-|
|torch.optim.lr_scheduler.CosineAnnealingLR.get_last_lr|Yes|-|
|torch.optim.lr_scheduler.CosineAnnealingLR.load_state_dict|Yes|-|
|torch.optim.lr_scheduler.CosineAnnealingLR.state_dict|Yes|-|
|torch.optim.lr_scheduler.ChainedScheduler|Yes|-|
|torch.optim.lr_scheduler.ChainedScheduler.get_last_lr|Yes|-|
|torch.optim.lr_scheduler.ChainedScheduler.load_state_dict|Yes|-|
|torch.optim.lr_scheduler.ChainedScheduler.state_dict|Yes|-|
|torch.optim.lr_scheduler.SequentialLR|Yes|-|
|torch.optim.lr_scheduler.SequentialLR.get_last_lr|Yes|-|
|torch.optim.lr_scheduler.SequentialLR.load_state_dict|Yes|-|
|torch.optim.lr_scheduler.SequentialLR.state_dict|Yes|-|
|torch.optim.lr_scheduler.ReduceLROnPlateau|Yes|-|
|torch.optim.lr_scheduler.CyclicLR|Yes|-|
|torch.optim.lr_scheduler.CyclicLR.get_last_lr|Yes|-|
|torch.optim.lr_scheduler.CyclicLR.get_lr|Yes|-|
|torch.optim.lr_scheduler.OneCycleLR|Yes|-|
|torch.optim.lr_scheduler.OneCycleLR.get_last_lr|Yes|-|
|torch.optim.lr_scheduler.OneCycleLR.load_state_dict|Yes|-|
|torch.optim.lr_scheduler.OneCycleLR.state_dict|Yes|-|
|torch.optim.lr_scheduler.CosineAnnealingWarmRestarts|Yes|-|
|torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.get_last_lr|Yes|-|
|torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.load_state_dict|Yes|-|
|torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.state_dict|Yes|-|
|torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.step|Yes|-|
