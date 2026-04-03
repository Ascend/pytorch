import importlib
import json
import logging
import os
import sys
from typing import Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torchair
import common
import torch_npu
from torch_npu.dynamo.torchair._utils.path_manager import PathManager

log = logging.getLogger(__name__)
_patch_table = {}


def register_patch(*model_names):
    def meta_decorator(fn):
        for model_name in model_names:
            _patch_table[model_name] = fn
        return fn

    return meta_decorator


def check_transformers_version(required_version):
    import transformers
    if transformers.__version__ != required_version:
        log.warning(f"transformers.__version__ is not equal to {required_version}, which may cause error patch.")


def use_aclnn():
    os.environ["USE_ACLOP"] = "0"


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
            raise ValueError(f"past_key_value should have 2 past states. Got {len(past_key_value)} past states")
        real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

    key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

    def shape(states):
        return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

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
        hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
    )
    value_states = project(
        hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
    )

    scores = torch.matmul(query_states, key_states.transpose(3, 2))

    def process_position_bias():
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

        if past_key_value is not None:
            position_bias = position_bias[:, :, -hidden_states.size(1):, :]

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

    attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    if layer_head_mask is not None:
        attn_weights = attn_weights * layer_head_mask

    attn_output = unshape(torch.matmul(attn_weights, value_states))
    attn_output = self.o(attn_output)

    present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    if output_attentions:
        outputs = outputs + (attn_weights,)
    return outputs


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
        log.warning("Import transformers failed or could not get T5Attention "
                    "from module transformers.models.t5.modeling_t5")
        return
    check_transformers_version("4.36.0")

    T5Attention.forward = _hf_t5_mt5_conditionalgeneration_forward_new


@register_patch("fastNLP_Bert")
def _patch_model_5():
    os.environ['BREAK_GRAPH_OP_LIST'] = 'NN.LINEAR'
    # None-public interface, just for test.
    # This env is added after torchair's init,
    # so need to call break_graph patch again,
    torchair._utils.npu_patch_break_graph()


@register_patch("hf_Longformer")
def _patch_model_6():
    """
    Hf_Longformer failed accurazy test because of discontiguous memory.
    Solving the problem by adding  .contiguous() after .view() and .as_strided in LongformerSelfAttention._chunk.
    This patch would be removed in the near future.
    """
    # close AddLayerNormFusionPass
    close_view_optimise()
    module_spec = importlib.util.find_spec("transformers")
    if module_spec is None:
        return
    from transformers.models.longformer import LongformerSelfAttention
    src_chunk = LongformerSelfAttention._chunk

    def _chunk(cls, hidden_states, window_overlap, onnx_export: bool = False):
        if not onnx_export:
            hidden_states = hidden_states.view(
                hidden_states.size(0),
                torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
                window_overlap * 2,
                hidden_states.size(2),
            ).contiguous()
            chunk_size = list(hidden_states.size())
            chunk_size[1] = chunk_size[1] * 2 - 1

            chunk_stride = list(hidden_states.stride())
            chunk_stride[1] = chunk_stride[1] // 2
            return hidden_states.as_strided(size=chunk_size, stride=chunk_stride).contiguous()
        return src_chunk(hidden_states, window_overlap, True)

    LongformerSelfAttention._chunk = _chunk


@register_patch("soft_actor_critic")
def _patch_model_7():
    """
    soft_actor_critic failed accurazy test because of discontiguous memory.
    Solving the problem by adding  .contiguous() in soft_actor_critic/net.py line:242 SquashedNormal.__init__
    This patch would be removed in the near future.
    """
    from torchbenchmark.models.soft_actor_critic.nets import StochasticActor, SquashedNormal, BetaDist
    import torch.nn.functional as F

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


@register_patch("dcgan", "mobilenet_v2", "phlippe_resnet", "shufflenet_v2_x1_0", "squeezenet1_1", "vgg16",
                "alexnet", "densenet121", "maml_omniglot")
def _patch_model_8():
    """
    close conv amp for some model only in accuracy mode.
    This patch would be removed in the near future.
    """
    if {"--only", "--amp", "--accuracy"} <= set(sys.argv):
        from torch.nn.modules.conv import Conv2d

        def conv2d_amp_disabled(self, x):
            with torch.npu.amp.autocast(enabled=False):
                return self._conv_forward(x, self.weight, self.bias)

        Conv2d.forward = conv2d_amp_disabled


@register_patch("timm_nfnet")
def _patch_model_9():
    # close conv amp for timm_nfnet only in accuracy mode.
    # Increase the batch_size to a larger size 16,
    # to mitigate the impact of BatchNorm's tolerance on convolution
    if {"--only", "--amp", "--accuracy"} <= set(sys.argv):
        try:
            import timm
            import torch.nn.functional as F
            from timm.layers.std_conv import ScaledStdConv2dSame
            from timm.layers.padding import pad_same
        except ImportError:
            log.warning("Import timm failed or could not get ScaledStdConv2dSame"
                        "from module timm.layers.std_conv.ScaledStdConv2dSame")
            return
        if timm.__version__ != '0.9.16':
            log.warning("timm.__version__ is not equal to 0.9.16, which may cause error patch.")

        def new_forward(self, x):
            if self.same_pad:
                x = pad_same(x, self.kernel_size, self.stride, self.dilation)
            weight = F.batch_norm(
                self.weight.reshape(1, self.out_channels, -1), None, None,
                weight=(self.gain * self.scale).view(-1),
                training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
            with torch.npu.amp.autocast(enabled=False):
                return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        ScaledStdConv2dSame.forward = new_forward

    try:
        from torchbenchmark.models.timm_nfnet import Model
    except ImportError:
        log.warning("Import Model failed or could not find timm_nfnet"
                    "from module torchbenchmark.models.timm_nfnet.Model")
        return

    def new__init(self, test, device, jit=False, batch_size=None, extra_args=None):
        super(Model, self).__init__(test=test, model_name='dm_nfnet_f0',
                                    device=device, batch_size=16, extra_args=extra_args)

    Model.__init__ = new__init


@register_patch("nvidia_deeprecommender")
def _patch_model_10():
    try:
        from torch_npu.contrib import transfer_to_npu
        import torch_npu._inductor
    except ImportError:
        log.warning("NPU_FlAG is False!")
        return

    try:
        from torchbenchmark.models.nvidia_deeprecommender.nvtrain import DeepRecommenderTrainBenchmark
    except ImportError:
        log.warning("Import nvidia_deeprecommender failed or could not get DeepRecommenderTrainBenchmark"
                    "from module torchbenchmark.models.nvidia_deeprecommender.nvtrain.DeepRecommenderTrainBenchmark")
        return

    def new_init(self, device="cpu", jit=False, batch_size=256, process_command_line=False):
        self.TrainInit("cuda", jit, batch_size, process_command_line)

    DeepRecommenderTrainBenchmark.__init__ = new_init


@register_patch("functorch_dp_cifar10")
def _patch_model_11():
    if {"--only", "--amp", "--accuracy"} <= set(sys.argv):
        try:
            from torchbenchmark.models.functorch_dp_cifar10 import Model
            import torchvision.models as models
        except ImportError:
            log.warning("import torchvision fail or could not get Model from module "
                        "torchbenchmark.models.functorch_dp_cifar10")
            return

        def new_init(self, test, device, batch_size=None, extra_args=None):
            if extra_args is None:
                extra_args = []
            super(Model, self).__init__(test=test, device=device, batch_size=32, extra_args=extra_args)
            self.model = models.__dict__['resnet18'](
                pretrained=False, norm_layer=(lambda c: nn.GroupNorm(min(c, 32), c)))
            self.model = self.model.to(device)
            self.example_inputs = (
                torch.randn((self.batch_size, 3, 32, 32), device=self.device),
            )
            self.example_target = torch.randint(0, 10, (self.batch_size,), device=self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.CrossEntropyLoss()

        Model.__init__ = new_init


def create_fusion_switch_file():
    fusion_config = {}
    fusion_config.setdefault("Switch", {}).setdefault("GraphFusion", {})["AddLayerNormFusionPass"] = "off"
    fusion_config_file = os.path.join(os.getcwd(), "fusion_switch.cfg")
    PathManager.check_path_writeable_and_safety(fusion_config_file)
    with os.fdopen(os.open(fusion_config_file, os.O_WRONLY | os.O_CREAT, mode=600), 'w') as f:
        json.dump(fusion_config, f)
    config = torchair.CompilerConfig()
    config.fusion_config.fusion_switch_file = fusion_config_file

    def clean_fusion_config_file():
        PathManager.remove_file_safety(fusion_config_file)

    from common import register_callback
    register_callback(clean_fusion_config_file)
    return config


def close_view_optimise():
    config = create_fusion_switch_file()
    config.experimental_config.enable_view_optimize = False
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    def compile_with_view_switch(args):
        return torch._dynamo.optimize(npu_backend, nopython=args.nopython)
    common.compile_with_backend = compile_with_view_switch


def close_add_layer_norm_fusion_pass():
    npu_backend = torchair.get_npu_backend(compiler_config=create_fusion_switch_file())

    def compile_with_fusion_switch(args):
        return torch._dynamo.optimize(npu_backend, nopython=args.nopython)
    common.compile_with_backend = compile_with_fusion_switch


@register_patch("moco")
def _patch_model_13():
    from argparse import Namespace
    import torch.distributed as dist

    try:
        from torch_npu.contrib import transfer_to_npu
    except ImportError:
        log.warning("NPU_FlAG is False!")
        return

    try:
        import torchvision.models as models
        from torchbenchmark.models.moco import Model
        from torchbenchmark.models.moco.moco.builder import MoCo
    except ImportError:
        log.warning("import torchvision fail or could not get Model,MoCo from module torchbenchmark.models.moco ")
        return

    def new_init(self, test, device, batch_size=None, extra_args=None):
        if extra_args is None:
            extra_args = []
        super(Model, self).__init__(test=test, device=device, batch_size=batch_size, extra_args=extra_args)
        self.opt = Namespace(**{
            "arch": "resnet50", "epochs": 2, "start_epoch": 0, "lr": 0.03, "schedule": [120, 160], "momentum": 0.9,
            "weight_decay": 1e-4, "gpu": None, "moco_dim": 128, "moco_k": 32000, "moco_m": 0.999, "moco_t": 0.07,
            "mlp": False, "aug_plus": False, "cos": False, "fake_data": True, "distributed": True,
        })
        try:
            dist.init_process_group(backend="nccl", init_method="tcp://localhost:10001", world_size=1, rank=0)
        except RuntimeError:
            pass  # already initialized?

        if device == "cpu":
            raise NotImplementedError("DistributedDataParallel/allgather requires npu")

        self.model = MoCo(
            models.__dict__[self.opt.arch],
            self.opt.moco_dim,
            self.opt.moco_k,
            self.opt.moco_m,
            self.opt.moco_t,
            self.opt.mlp,
        )
        self.model.to(self.device)

        # Define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            self.opt.lr,
            momentum=self.opt.momentum,
            weight_decay=self.opt.weight_decay,
        )

        def collate_train_fn(data):
            ind = data[0]
            return [batches[2 * ind], batches[2 * ind + 1]], 0

        batches = []
        for _ in range(4):
            batches.append(torch.randn(self.batch_size, 3, 224, 224).to(self.device))
        self.example_inputs = torch.utils.data.DataLoader(range(2), collate_fn=collate_train_fn)
        if torch.cuda.is_available():
            for _, (images, _) in enumerate(self.example_inputs):
                images[0] = images[0].cuda(device=0, non_blocking=True)
                images[1] = images[1].cuda(device=0, non_blocking=True)
        else:
            for _, (images, _) in enumerate(self.example_inputs):
                images[0] = images[0].npu(device=0, non_blocking=True)
                images[1] = images[1].npu(device=0, non_blocking=True)

    Model.__init__ = new_init

    if {"--only", "--amp", "--accuracy"} <= set(sys.argv):
        try:
            import torchvision
            from torchvision.models.resnet import ResNet
        except ImportError:
            log.warning("Import torchvision failed or could not get ResNet "
                        "from module torchvision.models.resnet")
            return

        def _new_forward_impl(self, x):
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            @torch.compiler.disable(recursive=False)
            def avgpool(x):
                x = self.avgpool(x)
                return x

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

        ResNet._forward_impl = _new_forward_impl


@register_patch("timm_vovnet")
def _patch_model_14():
    if {"--only", "--amp", "--accuracy"} <= set(sys.argv):
        import torch.nn.functional as F
        from torch.nn.modules.conv import Conv2d
        from torch.nn.modules.pooling import AdaptiveAvgPool2d
        from torch.nn.modules.linear import Linear

        def conv2d_amp_disabled(self, x):
            with torch.npu.amp.autocast(enabled=False):
                return self._conv_forward(x, self.weight, self.bias)
        Conv2d.forward = conv2d_amp_disabled

        def adaptive_avgpool_amp_disabled(self, x):
            with torch.npu.amp.autocast(enabled=False):
                return F.adaptive_avg_pool2d(x.float(), self.output_size)
        AdaptiveAvgPool2d.forward = adaptive_avgpool_amp_disabled

        def linear_amp_disabled(self, x):
            with torch.npu.amp.autocast(enabled=False):
                return F.linear(x, self.weight, self.bias)
        Linear.forward = linear_amp_disabled


@register_patch("resnet50", "resnet152", "resnext50_32x4d")
def _patch_model_18():
    if {"--only", "--amp", "--accuracy"} <= set(sys.argv):
        try:
            import torchvision.models as models
        except ImportError:
            log.warning("Import torchvision failed or could not get models "
                    "from module torchvision.models")
            return

        if 'resnet50' in sys.argv:
            from torchbenchmark.models.resnet50 import Model
            model = 'resnet50'
            weight = models.ResNet50_Weights.IMAGENET1K_V1
        elif 'resnet152' in sys.argv:
            from torchbenchmark.models.resnet152 import Model
            model = 'resnet152'
            weight = models.ResNet152_Weights.IMAGENET1K_V1
        elif 'resnext50_32x4d' in sys.argv:
            from torchbenchmark.models.resnext50_32x4d import Model
            model = 'resnext50_32x4d'
            weight = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        else:
            raise RuntimeError("args.only expect model resnet50, resnet152 or resnext50_32x4d")

        def new_init(self, test, device, batch_size=None, extra_args=None):
            if extra_args is None:
                extra_args = []
            super(Model, self).__init__(model_name=model, test=test, device=device,
                                        batch_size=32, weights=weight,
                                        extra_args=extra_args)
        Model.__init__ = new_init


@register_patch("torch_multimodal_clip")
def _patch_model_19():
    try:
        from torchmultimodal.models.clip.text_encoder import CLIPTextEncoder
    except ImportError:
        log.warning("from torchmultimodal.models.clip.text_encoder import CLIPTextEncoder failed")
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
        hidden_state = hidden_state * 1 # pass
        if return_hidden_state:
            return hidden_state

        projected_embeddings = self.projection(
            hidden_state[torch.arange(hidden_state.shape[0], device="npu"), text.argmax(dim=-1)]
        )
        return projected_embeddings

    CLIPTextEncoder.forward = new_forward


def patch_remove_ops_from_generate_list(op_names=None):
    try:
        import torch
        from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir import config as anir_config

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
                print(f"[patch] {name} not found in GENERATE_LIST (maybe already removed).")

    except Exception as e:
        print(f"[patch] Failed to modify GENERATE_LIST: {e}")


@register_patch("speech_transformer")
def _patch_model_20():
    import numpy as np
    try:
        from torchbenchmark.models.speech_transformer.speech_transformer.transformer.attention import MultiHeadAttention, ScaledDotProductAttention
    except ImportError:
        log.warning("import torchvision fail or could not get MultiHeadAttention or ScaledDotProductAttention from module "
                    "torchbenchmark.models.speech_transformer.transformer.attention")
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
        self.temperature = d_k ** 0.5
        self.attention = ScaledDotProductAttention(temperature=self.temperature,
                                                   attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    MultiHeadAttention.__init__ = new_init

    patch_remove_ops_from_generate_list(["aten.cat", "aten.full"])



def patch_model(model_name):
    if model_name not in _patch_table.keys():
        return
    # do patch
    _patch_table[model_name]()
