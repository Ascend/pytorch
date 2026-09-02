import gc
import logging
import os
import sys

import torch
import torch.nn as nn

import torch_npu


log = logging.getLogger(__name__)
_patch_table = {}


def _patch_environment_variables():
    """Patch to allow configuring torchbenchmark paths from environment variables.
    This patch modifies the REPO_PATH and DATA_PATH variables in the torchbenchmark
    module to use values from environment variables if they are set.
    """
    import os
    from pathlib import Path

    try:
        import torchbenchmark  # noqa: F401

        torchbenchmark_module = sys.modules["torchbenchmark"]
        # Update DATA_PATH if environment variable is set
        if "TORCHBENCH_DATA_PATH" in os.environ:
            new_data_path = Path(os.environ["TORCHBENCH_DATA_PATH"])
            torchbenchmark_module.DATA_PATH = new_data_path
            log.info(f"Updated DATA_PATH to: {new_data_path}")  # noqa: G004

    except Exception as e:
        log.warning(f"Failed to patch torchbenchmark environment variables: {e}")  # noqa: G004


def register_patch(*model_names):
    def meta_decorator(fn):
        for model_name in model_names:
            _patch_table[model_name] = fn
        return fn

    return meta_decorator


def check_transformers_version(required_version):
    import transformers

    if transformers.__version__ != required_version:
        log.warning(
            f"transformers.__version__ is not equal to {required_version}, which may cause error patch."  # noqa: G004
        )


def use_aclnn():
    os.environ["USE_ACLOP"] = "0"


def _is_triton_experimental_backend():
    """Return True when the active NPU inductor backend is triton_experimental.

    configure_compile_options() sets TORCHINDUCTOR_NPU_BACKEND before
    patch_model() runs, so this reflects the backend selected for the run.

    The triton_experimental backend does not consume the ascend_npu_ir config
    (GENERATE_LIST / force_fallback_kernel_names / decomps_to_exclude_npu), and
    it relies on the stock inductor decomposition table. The compiler-tuning
    patches below are therefore either meaningless or actively harmful for it,
    so callers skip them when this returns True.
    """
    return os.environ.get("TORCHINDUCTOR_NPU_BACKEND") == "triton_experimental"


def _hf_t5_mt5_conditionalgeneration_forward_new(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_value=None,
    layer_head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
):
    batch_size, seq_length = hidden_states.shape[:2]

    real_seq_length = seq_length

    if past_key_value is not None:
        if len(past_key_value) != 2:
            raise ValueError(
                f"past_key_value should have 2 past states. Got {len(past_key_value)} past states"
            )
        real_seq_length += (
            past_key_value[0].shape[2] if query_length is None else query_length
        )

    key_length = (
        real_seq_length if key_value_states is None else key_value_states.shape[1]
    )

    def shape(states):
        return states.view(
            batch_size, -1, self.n_heads, self.key_value_proj_dim
        ).transpose(1, 2)

    def unshape(states):
        return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

    def project(hidden_states, proj_layer, key_value_states, past_key_value):
        if key_value_states is None:
            hidden_states = shape(proj_layer(hidden_states))
        elif past_key_value is None:
            hidden_states = shape(proj_layer(key_value_states))

        if past_key_value is not None:
            if key_value_states is None:
                hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
            elif past_key_value.shape[2] != key_value_states.shape[1]:
                hidden_states = shape(proj_layer(key_value_states))
            else:
                hidden_states = past_key_value
        return hidden_states

    query_states = shape(self.q(hidden_states))

    key_states = project(
        hidden_states,
        self.k,
        key_value_states,
        past_key_value[0] if past_key_value is not None else None,
    )
    value_states = project(
        hidden_states,
        self.v,
        key_value_states,
        past_key_value[1] if past_key_value is not None else None,
    )

    scores = torch.matmul(query_states, key_states.transpose(3, 2))

    def process_position_bias():
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, real_seq_length, key_length),
                device=scores.device,
                dtype=scores.dtype,
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(
                real_seq_length, key_length, device=scores.device
            )

        if past_key_value is not None:
            position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

        if mask is not None:
            position_bias = position_bias + mask
        return position_bias

    if position_bias is None:
        position_bias = process_position_bias()

    if self.pruned_heads:
        mask = torch.ones(position_bias.shape[1])
        mask[list(self.pruned_heads)] = 0
        position_bias_masked = position_bias[:, mask.bool()]
    else:
        position_bias_masked = position_bias

    # Only patch here, src code: [scores += position_bias_masked]
    # Prevent from two continuous _to_copy.
    scores = scores.float() + position_bias_masked

    attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)

    attn_weights = nn.functional.dropout(
        attn_weights, p=self.dropout, training=self.training
    )

    if layer_head_mask is not None:
        attn_weights = attn_weights * layer_head_mask

    attn_output = unshape(torch.matmul(attn_weights, value_states))
    attn_output = self.o(attn_output)

    present_key_value_state = (
        (key_states, value_states) if (self.is_decoder and use_cache) else None
    )
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    if output_attentions:
        outputs = outputs + (attn_weights,)
    return outputs


def _hf_distilbert_multiheadselfattention_forward_new(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    head_mask=None,
    output_attentions: bool = False,
):
    """
    Parameters:
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        mask: torch.tensor(bs, seq_length)

    Returns:
        weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
        seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
    """
    import math

    bs, q_length, dim = query.size()
    k_length = key.size(1)

    dim_per_head = self.dim // self.n_heads

    mask_reshp = (bs, 1, 1, k_length)

    def shape(x: torch.Tensor) -> torch.Tensor:
        return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

    def unshape(x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

    q = shape(self.q_lin(query))
    k = shape(self.k_lin(key))
    v = shape(self.v_lin(value))

    q = q / math.sqrt(dim_per_head)
    scores = torch.matmul(q, k.transpose(2, 3))
    mask = (mask == 0).view(mask_reshp).expand_as(scores)
    scores = scores.masked_fill(
        mask, torch.tensor(torch.finfo(scores.dtype).min, device="npu")
    )

    weights = nn.functional.softmax(scores, dim=-1)
    weights = self.dropout(weights)

    if head_mask is not None:
        weights = weights * head_mask

    context = torch.matmul(weights, v)
    context = unshape(context)
    context = self.out_lin(context)

    if output_attentions:
        return (context, weights)
    else:
        return (context,)


@register_patch("LearningToPaint")
def _patch_model_1():
    # For model LearningToPaint.
    from torchbenchmark.models import LearningToPaint

    USE_DEVICE = torch.cuda.is_available() or torch_npu.npu.is_available()
    LearningToPaint.baseline.utils.util.USE_CUDA = USE_DEVICE


@register_patch("hf_T5", "hf_T5_base")
def _patch_model_3():
    # For model hf_T5 and hf_T5_base.
    # In these models, accuracy check will fail because in the model's block [T5Attention],
    # two continuous _to_copy are invoked: the first _to_copy converts Tensor to half
    # and the second converts it to float. In eager, there will be a loss of precision.
    # But in graph, there will be a fusion pass to prevent it happens, causing acc check fail.
    try:
        from transformers.models.t5.modeling_t5 import T5Attention
    except ImportError:
        log.warning(
            "Import transformers failed or could not get T5Attention "
            "from module transformers.models.t5.modeling_t5"
        )
        return
    check_transformers_version("4.36.0")

    T5Attention.forward = _hf_t5_mt5_conditionalgeneration_forward_new


@register_patch("soft_actor_critic")
def _patch_model_7():
    """
    soft_actor_critic failed accurazy test because of discontiguous memory.
    Solving the problem by adding  .contiguous() in soft_actor_critic/net.py line:242 SquashedNormal.__init__
    This patch would be removed in the near future.
    """
    from torchbenchmark.models.soft_actor_critic import Model

    Model.DEFAULT_TRAIN_BSIZE = 32768
    Model.DEFAULT_EVAL_BSIZE = 32768

    import torch.nn.functional as F
    from torchbenchmark.models.soft_actor_critic.nets import (
        BetaDist,
        SquashedNormal,
        StochasticActor,
    )

    def new_forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        mu, log_std = out.chunk(2, dim=1)
        if self.dist_impl == "pyd":
            log_std = torch.tanh(log_std)
            log_std = self.log_std_low + 0.5 * (
                self.log_std_high - self.log_std_low
            ) * (log_std + 1)
            std = log_std.exp()
            dist = SquashedNormal(mu.contiguous(), std.contiguous())
        elif self.dist_impl == "beta":
            out = 1.0 + F.softplus(out)
            alpha, beta = out.chunk(2, dim=1)
            dist = BetaDist(alpha, beta)
        return dist

    StochasticActor.forward = new_forward


@register_patch("nvidia_deeprecommender")
def _patch_model_10():
    try:
        from torchbenchmark.models.nvidia_deeprecommender import (
            nvinfer,
            nvtrain,
        )
    except ImportError:
        log.warning(
            "Import nvidia_deeprecommender failed; the NPU compatibility patch "
            "was not applied"
        )
        return

    train_cls = nvtrain.DeepRecommenderTrainBenchmark
    inference_cls = nvinfer.DeepRecommenderInferenceBenchmark
    if getattr(train_cls, "_npu_patch_applied", False):
        return

    original_train_init = train_cls.__init__
    original_inference_init = inference_cls.__init__

    def reset_optimizer(self):
        if self.args.optimizer == "adam":
            self.optimizer = nvtrain.optim.Adam(
                self.rencoder.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "adagrad":
            self.optimizer = nvtrain.optim.Adagrad(
                self.rencoder.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "momentum":
            self.optimizer = nvtrain.optim.SGD(
                self.rencoder.parameters(),
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=self.args.weight_decay,
            )
            self.scheduler = nvtrain.MultiStepLR(
                self.optimizer,
                milestones=[24, 36, 48, 66, 72],
                gamma=0.5,
            )
        elif self.args.optimizer == "rmsprop":
            self.optimizer = nvtrain.optim.RMSprop(
                self.rencoder.parameters(),
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer kind: {self.args.optimizer}")

    def train_init(
        self, device="cpu", jit=False, batch_size=256, processCommandLine=False
    ):
        target_device = torch.device(device)
        if target_device.type != "npu":
            return original_train_init(
                self, device, jit, batch_size, processCommandLine
            )

        # The upstream benchmark constructor only accepts CPU and CUDA. Build
        # its state on CPU first, then move the model and inputs to the actual
        # NPU device without relying on transfer_to_npu or CUDA API rewriting.
        original_train_init(self, "cpu", jit, batch_size, processCommandLine)
        self.device = target_device

        # Keep the existing DVM-friendly shape workaround while making it
        # backend-independent. The outer TorchBench runner also uses this model
        # with triton_experimental.
        if hasattr(self, "scheduler"):
            del self.scheduler
        del self.optimizer
        del self.rencoder
        del self.toyinputs
        gc.collect()

        self.toyvocab = 197952
        self.toyinputs = torch.randn(self.toybatch, self.toyvocab)
        if self.toytest:
            self.rencoder = nvtrain.model.AutoEncoder(
                layer_sizes=[self.toyvocab]
                + [int(layer) for layer in self.args.hidden_layers.split(",")],
                nl_type=self.args.non_linearity_type,
                is_constrained=self.args.constrained,
                dp_drop_prob=self.args.drop_prob,
                last_layer_activations=not self.args.skip_last_layer_nl,
            )

        self.args.use_cuda = False
        self.rencoder = self.rencoder.to(target_device)
        self.toyinputs = self.toyinputs.to(target_device)
        reset_optimizer(self)

    def inference_init(
        self, device="cpu", jit=False, batch_size=256, usecommandlineargs=False
    ):
        target_device = torch.device(device)
        if target_device.type != "npu":
            return original_inference_init(
                self, device, jit, batch_size, usecommandlineargs
            )

        original_inference_init(self, "cpu", jit, batch_size, usecommandlineargs)
        self.device = target_device
        self.args.use_cuda = False
        self.rencoder = self.rencoder.to(target_device)
        self.toyinputs = self.toyinputs.to(target_device)

    train_cls.__init__ = train_init
    inference_cls.__init__ = inference_init
    train_cls._npu_patch_applied = True
    inference_cls._npu_patch_applied = True


@register_patch("resnet50", "resnet152", "resnext50_32x4d", "densenet121")
def _patch_model_18():
    patch_remove_decomposition(
        [
            "aten._native_batch_norm_legit_no_training",
            "aten.threshold_backward",
            "aten.native_batch_norm_backward",
        ]
    )


@register_patch("torch_multimodal_clip")
def _patch_model_19():
    try:
        # Import the model class and necessary modules
        from PIL import Image
        from torchbenchmark import DATA_PATH
        from torchbenchmark.models.torch_multimodal_clip import Model
        from torchmultimodal.models.clip.model import clip_vit_b32
        from torchmultimodal.modules.losses.contrastive_loss_with_temperature import (
            ContrastiveLossWithTemperature,
        )
        from torchmultimodal.transforms.clip_transform import (
            CLIPImageTransform,
            CLIPTextTransform,
        )
    except ImportError:
        log.warning(
            "from torchmultimodal import Model failed or could not get DATA_PATH from torchbenchmark"
        )
        return

    def new_init(self, test, device, batch_size=1, extra_args=None):
        if extra_args is None:
            extra_args = []
        super(Model, self).__init__(
            test=test, device=device, batch_size=batch_size, extra_args=extra_args
        )

        # use global DATA_PATH directly instead of local .data folder
        self.data_folder = str(DATA_PATH)
        self.image_name = "pizza.jpg"
        self.image = Image.open(os.path.join(self.data_folder, self.image_name))
        self.text = ["pizza", "dog"] * 16
        self.img_transform = CLIPImageTransform(is_train=False)
        self.text_transform = CLIPTextTransform()

        self.images = [self.image for _ in range(self.batch_size)]
        self.texts = [self.text for _ in range(self.batch_size)]

        self.image_tensor = self.img_transform(self.images).to(self.device)
        self.text_tensor = self.text_transform(self.text).to(self.device)
        self.model = clip_vit_b32()
        self.model.to(self.device)

        # Create optimizer
        self.loss_fn = ContrastiveLossWithTemperature()
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),
            lr=5.0e-4,
            weight_decay=1.0e-4,
            eps=1.0e-6,
        )

    Model.__init__ = new_init

    try:
        from torchmultimodal.models.clip.text_encoder import CLIPTextEncoder
    except ImportError:
        log.warning(
            "from torchmultimodal.models.clip.text_encoder import CLIPTextEncoder failed"
        )
        return

    def new_forward(self, text, return_hidden_state: bool = False):
        if text.size(1) != self.context_length:
            raise ValueError(
                f"length of input should be {self.context_length} but found {text.size(1)}"
            )
        embeddings = self.token_embedding(text)
        embeddings = embeddings + self.positional_embedding
        embeddings = embeddings.permute(1, 0, 2)
        embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)

        # [n_ctx, bs, transformer.width] -> [bs, n_ctx, transformer.width]
        embeddings = torch.permute(embeddings, (1, 0, 2))
        hidden_state = self.ln_final(embeddings)
        hidden_state = hidden_state * 1  # pass
        if return_hidden_state:
            return hidden_state

        projected_embeddings = self.projection(
            hidden_state[
                torch.arange(hidden_state.shape[0], device="npu"), text.argmax(dim=-1)
            ]
        )
        return projected_embeddings

    CLIPTextEncoder.forward = new_forward


def patch_remove_ops_from_generate_list(op_names=None):
    if _is_triton_experimental_backend():
        print(
            "[patch] triton_experimental backend: skip GENERATE_LIST tuning "
            "(ascend_npu_ir only)."
        )
        return
    try:
        import torch

        from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir import (
            config as anir_config,
        )

        if not op_names:
            print("[patch] No op names provided, nothing to do.")
            return

        for name in op_names:
            parts = name.split(".")
            op = torch.ops
            for p in parts:
                op = getattr(op, p)

            if op in anir_config.GENERATE_LIST:
                anir_config.GENERATE_LIST.remove(op)
                print(f"[patch] Successfully removed {name} from GENERATE_LIST.")
            else:
                print(
                    f"[patch] {name} not found in GENERATE_LIST (maybe already removed)."
                )

    except Exception as e:
        print(f"[patch] Failed to modify GENERATE_LIST: {e}")


def patch_remove_decomposition(op_names=None):
    if _is_triton_experimental_backend():
        print(
            "[patch] triton_experimental backend: keep stock inductor "
            "decompositions (skip removal)."
        )
        return
    try:
        from torch._decomp import remove_decompositions
        from torch._inductor import decomposition as inductor_decomp

        if not op_names:
            print("[patch] No op names provided, nothing to do.")
            return

        ops = []
        for name in op_names:
            op = torch.ops
            for p in name.split("."):
                op = getattr(op, p)
            ops.append(op)

        remove_decompositions(inductor_decomp.decompositions, ops)
        print(f"[patch] Successfully removed {len(ops)} decompositions from inductor.")

    except Exception as e:
        print(f"[patch] Failed to remove decompositions from inductor: {e}")


def patch_force_fallback_kernels(kernel_names=None):
    """Force ascend_npu_ir to fall back on the given fused kernels.

    Only meaningful for the mlir/dvm (ascend_npu_ir) backend; the
    triton_experimental backend does not consume this config, so skip it there.
    """
    if _is_triton_experimental_backend():
        print(
            "[patch] triton_experimental backend: skip force_fallback_kernel_names "
            "(ascend_npu_ir only)."
        )
        return
    if not kernel_names:
        print("[patch] No kernel names provided, nothing to do.")
        return
    try:
        from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir import (
            config as anir_config,
        )

        for name in kernel_names:
            anir_config.force_fallback_kernel_names[name] = True
        print(f"[patch] Forced fallback for {len(kernel_names)} kernel(s).")
    except ImportError:
        log.warning("import ascend_npu_ir config failed for force_fallback patch")


def patch_exclude_decomps_npu(op_names=None):
    """Exclude ops from ascend_npu_ir decomposition and the inductor table.

    The decomps_to_exclude_npu list is ascend_npu_ir-specific, and the
    accompanying remove_decompositions() call mutates the global inductor
    decomposition table that triton_experimental relies on, so skip both there.
    """
    if _is_triton_experimental_backend():
        print(
            "[patch] triton_experimental backend: keep stock decompositions "
            "(skip decomps_to_exclude_npu)."
        )
        return
    if not op_names:
        print("[patch] No op names provided, nothing to do.")
        return
    try:
        from torch._decomp import remove_decompositions
        from torch._inductor import decomposition as inductor_decomp

        from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir import (
            config as anir_config,
        )

        ops = []
        for name in op_names:
            op = torch.ops
            for p in name.split("."):
                op = getattr(op, p)
            ops.append(op)

        if hasattr(anir_config, "decomps_to_exclude_npu") and isinstance(
            anir_config.decomps_to_exclude_npu, list
        ):
            for op in ops:
                if op not in anir_config.decomps_to_exclude_npu:
                    anir_config.decomps_to_exclude_npu.append(op)
            remove_decompositions(inductor_decomp.decompositions, ops)
            print(f"[patch] Excluded {len(ops)} decomposition(s) for NPU.")
    except Exception:
        log.warning(
            "import config failed for decomps_to_exclude_npu patch", exc_info=True
        )


@register_patch("speech_transformer")
def _patch_model_20():
    import numpy as np

    try:
        from torchbenchmark.models.speech_transformer.speech_transformer.transformer.attention import (
            MultiHeadAttention,
            ScaledDotProductAttention,
        )
    except ImportError:
        log.warning(
            "import torchvision fail or could not get MultiHeadAttention or ScaledDotProductAttention from module "
            "torchbenchmark.models.speech_transformer.transformer.attention"
        )
        return

    def new_init(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        # fix two different devices npu, cpu
        self.temperature = d_k**0.5
        self.attention = ScaledDotProductAttention(
            temperature=self.temperature, attn_dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    MultiHeadAttention.__init__ = new_init

    patch_remove_ops_from_generate_list(["aten.cat", "aten.full"])


@register_patch("hf_T5_base")
def _patch_model_21():
    # The current operator suffers from severe performance degradation.
    # This patch will be removed after the issue is fixed in the future.
    patch_remove_decomposition(["aten._softmax"])
    patch_force_fallback_kernels(["mlir_fused_add_lt_neg_where_16"])


@register_patch("hf_T5_large")
def _patch_model_22():
    # The current operator suffers from severe performance degradation.
    # This patch will be removed after the issue is fixed in the future.
    patch_remove_decomposition(["aten._softmax"])
    patch_force_fallback_kernels(["mlir_fused_add_lt_neg_where_16"])


@register_patch("pytorch_unet")
def _patch_model_23():
    # Patch to address performance issues in the current network
    # 1. Prevent aten.constant_pad_nd from participating in fusion.
    #    This operator will be officially removed from the whitelist in future PTA versions,
    #    at which point this patch will be deleted.
    # 2. Disable decomposition for aten.upsample_bilinear2d.default
    patch_remove_ops_from_generate_list(["aten.constant_pad_nd"])


@register_patch("squeezenet1_1")
def _patch_squeezenet1_1():
    """
    fallbackdiv
    """
    patch_remove_ops_from_generate_list(["aten.div"])
    patch_force_fallback_kernels(["mlir_fused_relu_threshold_backward_2"])


@register_patch("T5ForConditionalGeneration")
def _patch_model_24():
    patch_force_fallback_kernels(["mlir_fused_add_lt_neg_where_13"])
    from torch._higher_order_ops.effects import (
        _EffectType,
        _register_effectful_op,
    )

    _register_effectful_op(
        torch.ops.aten.native_dropout.default,
        _EffectType.ORDERED,
    )


@register_patch("BartForCausalLM")
def _patch_model_25():
    patch_exclude_decomps_npu(
        [
            "aten.native_layer_norm",
            "aten.native_layer_norm_backward",
        ]
    )


@register_patch("DistilBertForMaskedLM")
def _patch_model_26():
    try:
        from transformers.models.distilbert.modeling_distilbert import MultiHeadSelfAttention
    except ImportError:
        log.warning(
            "Import transformers failed or could not get MultiHeadSelfAttention "
            "from module transformers.models.distilbert.modeling_distilbert"
        )
        return

    MultiHeadSelfAttention.forward = _hf_distilbert_multiheadselfattention_forward_new


@register_patch("timm_resnest")
def _patch_model_27():
    patch_remove_decomposition(
        [
            "aten.threshold_backward",
            "aten.native_batch_norm_backward",
        ]
    )


@register_patch("fastNLP_Bert")
def _patch_fastNLP_Bert():
    """Match upstream torchbenchmark fastNLP_Bert install patch.

    The model creates an empty chinese_wwm_pytorch.bin for structure-only
    benchmarking. Without this patch, BertModel.from_pretrained loads that
    empty file and raises EOFError.
    """
    try:
        from fastNLP.modules.encoder.bert import BertConfig, BertModel
    except ImportError:
        log.warning(
            "Import fastNLP failed or could not get BertModel/BertConfig; "
            "skip fastNLP_Bert patch"
        )
        return

    original_from_pretrained = BertModel.from_pretrained.__func__

    @classmethod
    def from_pretrained_no_weights(cls, pretrained_model_dir_or_name, *args, **kwargs):
        config_path = os.environ.get("TORCHBENCH_FASTNLP_CONFIG_PATH")
        if not config_path:
            candidate = os.path.join(pretrained_model_dir_or_name, "bert_config.json")
            if os.path.isfile(candidate):
                config_path = candidate
        if config_path and os.path.isfile(config_path):
            return cls(config=BertConfig.from_json_file(config_path))
        return original_from_pretrained(
            cls, pretrained_model_dir_or_name, *args, **kwargs
        )

    BertModel.from_pretrained = from_pretrained_no_weights


@register_patch("drq")
def _patch_drq():
    """Load obs.pkl from TORCHBENCH_DATA_PATH / DATA_PATH."""
    import pickle as pkl

    import numpy as np
    from torchbenchmark import DATA_PATH
    from torchbenchmark.models.drq import MockEnv
    from torchbenchmark.models.drq.drqutils import FrameStack
    import torchbenchmark.models.drq as drq_mod

    def make_env(cfg):
        obs_path = os.path.join(str(DATA_PATH), "obs.pkl")
        mockobs = pkl.load(open(obs_path, "rb"))
        mockobs = np.random.randint(low=11, high=228, size=mockobs.shape, dtype=np.uint8)
        env = MockEnv(mockobs)
        env = FrameStack(env, k=cfg.frame_stack)
        env.seed(cfg.seed)
        return env

    drq_mod.make_env = make_env



@register_patch("yolov3")
def _patch_yolov3():
    """Use DATA_PATH for coco128, and select_device for NPU."""
    from torchbenchmark import DATA_PATH

    data_dir = os.path.join(str(DATA_PATH), "coco128")
    # yolov3 asserts default data/.data/coco128 at import time; allow import then remap.
    _real_exists = os.path.exists

    def _exists_proxy(path):
        path_s = str(path).replace("\\", "/")
        if path_s.endswith("data/.data/coco128"):
            return _real_exists(data_dir) or _real_exists(path)
        return _real_exists(path)

    os.path.exists = _exists_proxy
    try:
        import torchbenchmark.models.yolov3 as yolov3_mod
        import torchbenchmark.models.yolov3.yolo_utils.torch_utils as torch_utils
    finally:
        os.path.exists = _real_exists

    yolov3_mod.DATA_DIR = data_dir

    def new_select_device(device="", apex=False, batch_size=None):
        device = "" if device is None else str(device)
        if device.lower() == "cpu" or not torch.npu.is_available():
            return torch.device("cpu")
        if device in ("", "npu"):
            idx = torch.npu.current_device()
        else:
            idx = int(device.split(",")[0].replace("npu:", ""))
            torch.npu.set_device(idx)
        return torch.device(f"npu:{idx}")

    torch_utils.select_device = new_select_device


def patch_model(model_name):
    if model_name not in _patch_table:
        return
    # do patch
    _patch_table[model_name]()
