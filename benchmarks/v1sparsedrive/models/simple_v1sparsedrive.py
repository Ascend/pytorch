import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModel
from mmdet.models.backbones import ResNet
from mmdet.models.necks import FPN


# =========================
# utils
# =========================

def build_mlp(channels, act_layer=nn.ReLU, last_norm=False):
    layers = []
    for i in range(len(channels) - 1):
        in_c = channels[i]
        out_c = channels[i + 1]
        layers.append(nn.Linear(in_c, out_c))
        is_last = (i == len(channels) - 2)
        if not is_last:
            layers.append(act_layer(inplace=True) if act_layer == nn.ReLU else act_layer())
            layers.append(nn.LayerNorm(out_c))
        elif last_norm:
            layers.append(nn.LayerNorm(out_c))
    return nn.Sequential(*layers)


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x):
        return x * self.scale


# =========================
# grid mask
# =========================

class GridMask(nn.Module):
    def __init__(self, use_h=True, use_w=True, rotate=1, ratio=0.5, prob=0.7):
        super().__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.ratio = ratio
        self.prob = prob

    def forward(self, x):
        if (not self.training) or torch.rand(1).item() > self.prob:
            return x

        _, _, h, w = x.shape
        device = x.device

        d = max(2, min(h, w) // 8)
        l = max(1, int(d * self.ratio))

        mask = torch.ones((h, w), device=device, dtype=x.dtype)

        if self.use_h:
            for i in range(0, h, d):
                mask[i:i + l, :] = 0

        if self.use_w:
            for j in range(0, w, d):
                mask[:, j:j + l] = 0

        mask = mask.unsqueeze(0).unsqueeze(0)
        x = x * mask
        return x


# =========================
# anchor / encoder like
# =========================

class SparseBox3DKeyPointsGenerator(nn.Module):
    def __init__(self, embed_dims=256, num_pts=9):
        super().__init__()
        self.learnable_fc = nn.Linear(embed_dims, num_pts * 2)

    def forward(self, query):
        return self.learnable_fc(query)


class SparsePoint3DKeyPointsGenerator(nn.Module):
    def __init__(self, embed_dims=256, out_dim=600):
        super().__init__()
        self.learnable_fc = nn.Linear(embed_dims, out_dim)

    def forward(self, query):
        return self.learnable_fc(query)


class TrajSparsePoint3DKeyPointsGenerator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class InstanceBank(nn.Module):
    def __init__(self, num_queries=300, embed_dims=256, anchor_handler=None):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dims = embed_dims
        self.anchor_handler = anchor_handler
        self.query_embed = nn.Parameter(torch.randn(num_queries, embed_dims) * 0.02)

    def forward(self, batch_size, device):
        query = self.query_embed.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        return query


class SparseBox3DEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_fc = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(inplace=True), nn.LayerNorm(128),
            nn.Linear(128, 128), nn.ReLU(inplace=True), nn.LayerNorm(128),
            nn.Linear(128, 128), nn.ReLU(inplace=True), nn.LayerNorm(128),
            nn.Linear(128, 128), nn.ReLU(inplace=True), nn.LayerNorm(128),
        )
        self.size_fc = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(inplace=True), nn.LayerNorm(32),
            nn.Linear(32, 32), nn.ReLU(inplace=True), nn.LayerNorm(32),
            nn.Linear(32, 32), nn.ReLU(inplace=True), nn.LayerNorm(32),
            nn.Linear(32, 32), nn.ReLU(inplace=True), nn.LayerNorm(32),
        )
        self.yaw_fc = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(inplace=True), nn.LayerNorm(32),
            nn.Linear(32, 32), nn.ReLU(inplace=True), nn.LayerNorm(32),
            nn.Linear(32, 32), nn.ReLU(inplace=True), nn.LayerNorm(32),
            nn.Linear(32, 32), nn.ReLU(inplace=True), nn.LayerNorm(32),
        )
        self.vel_fc = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(inplace=True), nn.LayerNorm(64),
            nn.Linear(64, 64), nn.ReLU(inplace=True), nn.LayerNorm(64),
            nn.Linear(64, 64), nn.ReLU(inplace=True), nn.LayerNorm(64),
            nn.Linear(64, 64), nn.ReLU(inplace=True), nn.LayerNorm(64),
        )
        self.out_proj = nn.Linear(128 + 32 + 32 + 64, 256)

    def forward(self, anchors):
        pos = anchors[..., 0:3]
        size = anchors[..., 3:6]
        yaw = anchors[..., 6:8]
        vel = anchors[..., 8:11]
        x = torch.cat([
            self.pos_fc(pos),
            self.size_fc(size),
            self.yaw_fc(yaw),
            self.vel_fc(vel),
        ], dim=-1)
        return self.out_proj(x)


class SparsePoint3DEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_fc = nn.Sequential(
            nn.Linear(40, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
        )

    def forward(self, anchors):
        return self.pos_fc(anchors)


# =========================
# attention / ffn like
# =========================

class MultiheadFlashAttention(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, batch_first=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=batch_first,
        )
        self.proj_drop = nn.Dropout(0.0)
        self.dropout_layer = nn.Dropout(0.1)

    def forward(self, x):
        out, _ = self.attn(x, x, x, need_weights=False)
        out = self.proj_drop(out)
        out = self.dropout_layer(out)
        return x + out


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, kdim=None, vdim=None, batch_first=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=batch_first,
            kdim=kdim,
            vdim=vdim,
        )
        self.proj_drop = nn.Dropout(0.0)
        self.dropout_layer = nn.Dropout(0.1)

    def forward(self, q, k=None, v=None):
        if k is None:
            k = q
        if v is None:
            v = k
        out, _ = self.attn(q, k, v, need_weights=False)
        out = self.proj_drop(out)
        out = self.dropout_layer(out)
        return q + out


class AsymmetricFFN(nn.Module):
    def __init__(self, in_dims=256, embed_dims=256, hidden_dims=512, pre_norm_dim=None):
        super().__init__()
        if pre_norm_dim is None:
            pre_norm_dim = in_dims

        self.in_dims = in_dims
        self.embed_dims = embed_dims
        self.pre_norm_dim = pre_norm_dim

        self.pre_norm = nn.LayerNorm(pre_norm_dim)
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(in_dims, hidden_dims),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            ),
            nn.Linear(hidden_dims, embed_dims),
            nn.Dropout(0.1),
        )
        self.dropout_layer = nn.Identity()
        self.identity_fc = nn.Linear(in_dims, embed_dims) if in_dims != embed_dims else nn.Identity()

    def forward(self, x):
        identity = self.identity_fc(x)
        out = self.pre_norm(x)
        out = self.layers(out)
        out = self.dropout_layer(out)
        return identity + out


# =========================
# feature aggregation like
# =========================

class DeformableFeatureAggregation(nn.Module):
    def __init__(self, embed_dims=256, num_levels=4, num_pts_out=18, weights_out=416):
        super().__init__()
        self.proj_drop = nn.Dropout(0.0)
        self.kps_generator = SparseBox3DKeyPointsGenerator(embed_dims, num_pts=num_pts_out // 2) \
            if num_pts_out == 18 else SparsePoint3DKeyPointsGenerator(embed_dims, out_dim=num_pts_out)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.camera_encoder = nn.Sequential(
            nn.Linear(12, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
        )
        self.weights_fc = nn.Linear(embed_dims, weights_out)

        self.level_proj = nn.ModuleList([
            nn.Linear(embed_dims, embed_dims) for _ in range(num_levels)
        ])

    def forward(self, query, fpn_feats, metas=None):
        pooled = []
        for i, feat in enumerate(fpn_feats):
            x = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
            x = self.level_proj[i](x).unsqueeze(1)
            pooled.append(x)

        pooled = torch.stack(pooled, dim=0).mean(dim=0)
        query = query + pooled

        out = self.output_proj(query)
        out = self.proj_drop(out)
        return out


# =========================
# refinement like
# =========================

class SparseBox3DRefinementModule(nn.Module):
    def __init__(self, embed_dims=256, num_classes=10, box_dim=11):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, box_dim),
            Scale(),
        )
        self.cls_layers = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, num_classes),
        )
        self.quality_layers = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, 2),
        )

    def forward(self, query):
        box = self.layers(query)
        cls = self.cls_layers(query)
        quality = self.quality_layers(query)
        return box, cls, quality


class SparsePoint3DRefinementModule(nn.Module):
    def __init__(self, embed_dims=256, num_classes=3, point_dim=40):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, point_dim),
            Scale(),
        )
        self.cls_layers = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, num_classes),
        )

    def forward(self, query):
        pts = self.layers(query)
        cls = self.cls_layers(query)
        return pts, cls


# =========================
# sparse4d head like
# =========================

class Sparse4DHeadLike(nn.Module):
    def __init__(
        self,
        mode="det",
        embed_dims=256,
        num_queries=300,
        num_classes=10,
        num_decoder=6,
    ):
        super().__init__()
        self.mode = mode

        if mode == "det":
            self.instance_bank = InstanceBank(
                num_queries=num_queries,
                embed_dims=embed_dims,
                anchor_handler=SparseBox3DKeyPointsGenerator(),
            )
            self.anchor_encoder = SparseBox3DEncoder()
            self.fc_before = nn.Linear(embed_dims, 512, bias=False)
            self.fc_after = nn.Linear(512, embed_dims, bias=False)

            layers = []
            for _ in range(num_decoder):
                layers.extend([
                    DeformableFeatureAggregation(embed_dims=256, num_pts_out=18, weights_out=416),
                    AsymmetricFFN(in_dims=512, embed_dims=256, hidden_dims=1024, pre_norm_dim=512),
                    nn.LayerNorm(256),
                    SparseBox3DRefinementModule(embed_dims=256, num_classes=num_classes, box_dim=11),
                    MultiheadFlashAttention(embed_dims=512, num_heads=8),
                    MultiheadFlashAttention(embed_dims=512, num_heads=8),
                    nn.LayerNorm(256),
                ])
            self.layers = nn.ModuleList(layers)

        else:
            self.instance_bank = InstanceBank(
                num_queries=num_queries,
                embed_dims=embed_dims,
                anchor_handler=SparsePoint3DKeyPointsGenerator(),
            )
            self.anchor_encoder = SparsePoint3DEncoder()
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

            layers = []
            layers.extend([
                MultiheadFlashAttention(embed_dims=256, num_heads=8),
                nn.LayerNorm(256),
            ])
            for _ in range(num_decoder):
                layers.extend([
                    DeformableFeatureAggregation(embed_dims=256, num_pts_out=600, weights_out=9600),
                    AsymmetricFFN(in_dims=256, embed_dims=256, hidden_dims=512, pre_norm_dim=256),
                    nn.LayerNorm(256),
                    SparsePoint3DRefinementModule(embed_dims=256, num_classes=num_classes, point_dim=40),
                    MultiheadFlashAttention(embed_dims=256, num_heads=8),
                    MultiheadFlashAttention(embed_dims=256, num_heads=8),
                    nn.LayerNorm(256),
                ])
            self.layers = nn.ModuleList(layers)

    def _make_dummy_anchors(self, batch_size, num_queries, device):
        if self.mode == "det":
            return torch.randn(batch_size, num_queries, 11, device=device)
        else:
            return torch.randn(batch_size, num_queries, 40, device=device)

    def forward(self, fpn_feats, metas=None):
        b = fpn_feats[0].shape[0]
        device = fpn_feats[0].device
        query = self.instance_bank(b, device)

        anchors = self._make_dummy_anchors(b, query.shape[1], device)
        anchor_embed = self.anchor_encoder(anchors)
        query = query + anchor_embed

        all_cls = []
        all_reg = []
        all_quality = []

        i = 0
        while i < len(self.layers):
            layer = self.layers[i]

            if isinstance(layer, DeformableFeatureAggregation):
                q_in = self.fc_before(query)
                q = self.fc_after(q_in)
                query = layer(q, fpn_feats, metas)
                i += 1

            elif isinstance(layer, AsymmetricFFN):
                q_in = self.fc_before(query)
                query = layer(q_in)
                i += 1

            elif isinstance(layer, nn.LayerNorm):
                query = layer(query)
                i += 1

            elif isinstance(layer, SparseBox3DRefinementModule):
                box, cls, quality = layer(query)
                all_reg.append(box)
                all_cls.append(cls)
                all_quality.append(quality)
                i += 1

            elif isinstance(layer, SparsePoint3DRefinementModule):
                pts, cls = layer(query)
                all_reg.append(pts)
                all_cls.append(cls)
                i += 1

            elif isinstance(layer, MultiheadFlashAttention):
                q_in = self.fc_before(query)
                q_out = layer(q_in)
                query = self.fc_after(q_out)
                i += 1

            else:
                query = layer(query)
                i += 1

        out = {
            "query": query,
            "all_cls_scores": all_cls,
            "all_reg_preds": all_reg,
        }
        if self.mode == "det":
            out["all_quality_scores"] = all_quality
        return out


# =========================
# motion plan head like
# =========================

class InstanceQueue(nn.Module):
    def __init__(self, embed_dims=256):
        super().__init__()
        self.ego_feature_encoder = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, feat):
        x = self.ego_feature_encoder(feat)
        x = x.flatten(1)
        return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class V1TrajPooler(nn.Module):
    def __init__(self, embed_dims=256):
        super().__init__()
        self.kps_generator = TrajSparsePoint3DKeyPointsGenerator()
        self.proj_drop = nn.Dropout(0.0)
        self.camera_encoder = nn.Sequential(
            nn.Linear(12, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
        )
        self.weights_fc = nn.Linear(256, 960)
        self.output_proj = nn.Linear(256, 256)

    def forward(self, query, fpn_feats):
        pooled = F.adaptive_avg_pool2d(fpn_feats[0], (1, 1)).flatten(1).unsqueeze(1)
        out = query + pooled
        out = self.output_proj(out)
        return out


class V1ModulationLayer(nn.Module):
    def __init__(self, embed_dims=256):
        super().__init__()
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dims, embed_dims * 2),
        )

    def forward(self, x, cond):
        scale_shift = self.scale_shift_mlp(cond)
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class V11MotionPlanningRefinementModule(nn.Module):
    def __init__(self, embed_dims=256, motion_dim=24, status_dim=10):
        super().__init__()
        self.motion_cls_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, 1),
        )
        self.motion_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, motion_dim),
        )
        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, status_dim),
        )

    def forward(self, x):
        pooled = x.mean(dim=1)
        motion_cls = self.motion_cls_branch(pooled)
        motion_reg = self.motion_reg_branch(pooled)
        status = self.plan_status_branch(pooled)
        return motion_cls, motion_reg, status


class V4DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(self, embed_dims=256, plan_dim=12):
        super().__init__()
        self.plan_cls_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, plan_dim),
        )

    def forward(self, x):
        pooled = x.mean(dim=1)
        plan_cls = self.plan_cls_branch(pooled)
        plan_reg = self.plan_reg_branch(pooled)
        return plan_cls, plan_reg


class V13MotionPlanningHeadLike(nn.Module):
    def __init__(self, embed_dims=256):
        super().__init__()
        self.instance_queue = InstanceQueue(embed_dims=embed_dims)

        self.interact_layers = nn.ModuleList([
            MultiheadAttention(embed_dims=512, num_heads=8),
            MultiheadFlashAttention(embed_dims=512, num_heads=8),
            nn.LayerNorm(256),
            MultiheadFlashAttention(embed_dims=256, num_heads=8),
            nn.LayerNorm(256),
            AsymmetricFFN(in_dims=256, embed_dims=256, hidden_dims=512, pre_norm_dim=256),
            nn.LayerNorm(256),

            MultiheadAttention(embed_dims=512, num_heads=8),
            MultiheadFlashAttention(embed_dims=512, num_heads=8),
            nn.LayerNorm(256),
            MultiheadFlashAttention(embed_dims=256, num_heads=8),
            nn.LayerNorm(256),
            AsymmetricFFN(in_dims=256, embed_dims=256, hidden_dims=512, pre_norm_dim=256),
            nn.LayerNorm(256),

            MultiheadAttention(embed_dims=512, num_heads=8),
            MultiheadFlashAttention(embed_dims=512, num_heads=8),
            nn.LayerNorm(256),
            MultiheadFlashAttention(embed_dims=256, num_heads=8),
            nn.LayerNorm(256),
            AsymmetricFFN(in_dims=256, embed_dims=256, hidden_dims=512, pre_norm_dim=256),
            nn.LayerNorm(256),

            V11MotionPlanningRefinementModule(embed_dims=256, motion_dim=24, status_dim=10),
        ])

        self.diff_layers = nn.ModuleList([
            V1TrajPooler(embed_dims=256),
            MultiheadFlashAttention(embed_dims=256, num_heads=8),
            nn.LayerNorm(256),
            MultiheadFlashAttention(embed_dims=512, num_heads=8),
            nn.LayerNorm(256),
            MultiheadFlashAttention(embed_dims=256, num_heads=8),
            nn.LayerNorm(256),
            AsymmetricFFN(in_dims=256, embed_dims=256, hidden_dims=512, pre_norm_dim=256),
            nn.LayerNorm(256),
            V1ModulationLayer(embed_dims=256),
            V4DiffMotionPlanningRefinementModule(embed_dims=256, plan_dim=12),

            V1TrajPooler(embed_dims=256),
            MultiheadFlashAttention(embed_dims=256, num_heads=8),
            nn.LayerNorm(256),
            MultiheadFlashAttention(embed_dims=512, num_heads=8),
            nn.LayerNorm(256),
            MultiheadFlashAttention(embed_dims=256, num_heads=8),
            nn.LayerNorm(256),
            AsymmetricFFN(in_dims=256, embed_dims=256, hidden_dims=512, pre_norm_dim=256),
            nn.LayerNorm(256),
            V1ModulationLayer(embed_dims=256),
            V4DiffMotionPlanningRefinementModule(embed_dims=256, plan_dim=12),
        ])

        self.fc_before = nn.Linear(256, 512, bias=False)
        self.fc_after = nn.Linear(512, 256, bias=False)

        self.motion_anchor_encoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
        )
        self.plan_anchor_encoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
        )
        self.plan_pos_encoder = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
        )
        self.kps_generator = TrajSparsePoint3DKeyPointsGenerator()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(256),
            nn.Linear(256, 1024),
            nn.Mish(),
            nn.Linear(1024, 256),
        )

    def forward(self, det_out, map_out, fpn_feats, metas=None):
        b = fpn_feats[0].shape[0]
        device = fpn_feats[0].device

        det_query = det_out["query"]
        map_query = map_out["query"]

        ego_feat = self.instance_queue(fpn_feats[0])
        ego_query = ego_feat.unsqueeze(1)

        motion_query = ego_query + det_query.mean(dim=1, keepdim=True)
        motion_query = motion_query + self.motion_anchor_encoder(motion_query)

        x = motion_query
        motion_cls = motion_reg = plan_status = None

        for layer in self.interact_layers:
            if isinstance(layer, MultiheadAttention):
                x = self.fc_after(layer(self.fc_before(x)))
            elif isinstance(layer, MultiheadFlashAttention):
                if x.shape[-1] == 256 and layer.attn.embed_dim == 512:
                    x = self.fc_after(layer(self.fc_before(x)))
                else:
                    x = layer(x)
            elif isinstance(layer, nn.LayerNorm):
                x = layer(x)
            elif isinstance(layer, AsymmetricFFN):
                x = layer(x)
            elif isinstance(layer, V11MotionPlanningRefinementModule):
                motion_cls, motion_reg, plan_status = layer(x)

        t = torch.arange(b, device=device).float()
        time_emb = self.time_mlp(t)

        plan_seed = torch.cat([
            det_query.mean(dim=1),
            map_query.mean(dim=1),
            ego_feat
        ], dim=-1)
        plan_query = self.plan_pos_encoder(plan_seed).unsqueeze(1)
        plan_query = plan_query + self.plan_anchor_encoder(plan_query)

        y = plan_query
        plan_cls = plan_reg = None
        for layer in self.diff_layers:
            if isinstance(layer, V1TrajPooler):
                y = layer(y, fpn_feats)
            elif isinstance(layer, MultiheadFlashAttention):
                if y.shape[-1] == 256 and layer.attn.embed_dim == 512:
                    y = self.fc_after(layer(self.fc_before(y)))
                else:
                    y = layer(y)
            elif isinstance(layer, nn.LayerNorm):
                y = layer(y)
            elif isinstance(layer, AsymmetricFFN):
                y = layer(y)
            elif isinstance(layer, V1ModulationLayer):
                y = layer(y, time_emb)
            elif isinstance(layer, V4DiffMotionPlanningRefinementModule):
                plan_cls, plan_reg = layer(y)

        return {
            "motion_query": x,
            "plan_query": y,
            "motion_cls": motion_cls,
            "motion_reg": motion_reg,
            "plan_status": plan_status,
            "plan_cls": plan_cls,
            "plan_reg": plan_reg,
        }


class V1SparseDriveHead(nn.Module):
    def __init__(
        self,
        num_det_classes=10,
        num_map_classes=3,
        det_num_queries=300,
        map_num_queries=100,
    ):
        super().__init__()

        self.det_head = Sparse4DHeadLike(
            mode="det",
            embed_dims=256,
            num_queries=det_num_queries,
            num_classes=num_det_classes,
            num_decoder=6,
        )
        self.map_head = Sparse4DHeadLike(
            mode="map",
            embed_dims=256,
            num_queries=map_num_queries,
            num_classes=num_map_classes,
            num_decoder=6,
        )
        self.motion_plan_head = V13MotionPlanningHeadLike(embed_dims=256)

    def forward(self, fpn_feats, metas=None):
        det_out = self.det_head(fpn_feats, metas)
        map_out = self.map_head(fpn_feats, metas)
        motion_plan_out = self.motion_plan_head(det_out, map_out, fpn_feats, metas)

        return {
            "det": det_out,
            "map": map_out,
            "motion_plan": motion_plan_out,
        }


class DenseDepthNetLike(nn.Module):
    def __init__(self, num_depth_layers=3, in_channels=256):
        super().__init__()
        self.depth_layers = nn.ModuleList([
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
            for _ in range(num_depth_layers)
        ])

    def forward(self, feats):
        outs = []
        for i, layer in enumerate(self.depth_layers):
            feat = feats[i]
            outs.append(layer(feat))
        return outs


class SimpleV1SparseDrive(BaseModel):
    def __init__(
        self,
        num_det_classes=10,
        num_map_classes=3,
        backbone_pretrained=None,
        use_grid_mask=True,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.num_det_classes = num_det_classes
        self.num_map_classes = num_map_classes

        self.img_backbone = ResNet(
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            style='pytorch',
            init_cfg=(
                dict(type='Pretrained', checkpoint=backbone_pretrained)
                if backbone_pretrained is not None else None
            ),
        )

        self.img_neck = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=4,
        )

        self.head = V1SparseDriveHead(
            num_det_classes=num_det_classes,
            num_map_classes=num_map_classes,
            det_num_queries=300,
            map_num_queries=100,
        )

        self.depth_branch = DenseDepthNetLike(num_depth_layers=3, in_channels=256)
        self.grid_mask = GridMask() if use_grid_mask else nn.Identity()

    def extract_feat(self, x):
        x = self.grid_mask(x)
        feats = self.img_backbone(x)
        feats = self.img_neck(feats)
        return feats

    def _forward_impl(self, inputs, data_samples=None):
        feats = self.extract_feat(inputs)
        head_out = self.head(feats, data_samples)
        depth_out = self.depth_branch(feats)

        outputs = {
            "head": head_out,
            "depth_branch": depth_out,
            "fpn_feats": feats,
        }
        return outputs

    def _parse_data_samples(self, data_samples, device):
        parsed = {
            "gt_det_boxes": [],
            "gt_det_labels": [],
            "gt_map_points": [],
            "gt_map_labels": [],
            "gt_motion": [],
            "gt_plan": [],
        }

        if data_samples is None:
            return parsed

        for sample in data_samples:
            parsed["gt_det_boxes"].append(sample.get("gt_det_boxes", None))
            parsed["gt_det_labels"].append(sample.get("gt_det_labels", None))
            parsed["gt_map_points"].append(sample.get("gt_map_points", None))
            parsed["gt_map_labels"].append(sample.get("gt_map_labels", None))
            parsed["gt_motion"].append(sample.get("gt_motion", None))
            parsed["gt_plan"].append(sample.get("gt_plan", None))

        return parsed

    def loss(self, inputs, data_samples=None):
        outputs = self._forward_impl(inputs, data_samples)

        det_out = outputs["head"]["det"]
        map_out = outputs["head"]["map"]
        motion_out = outputs["head"]["motion_plan"]
        depth_out = outputs["depth_branch"]

        device = inputs.device
        b = inputs.shape[0]
        parsed_samples = self._parse_data_samples(data_samples, device)

        loss_dict = {}

        det_last_cls = det_out["all_cls_scores"][-1]
        det_last_reg = det_out["all_reg_preds"][-1]
        det_last_quality = det_out["all_quality_scores"][-1]

        det_target_cls = torch.zeros(
            det_last_cls.shape[0], det_last_cls.shape[1],
            dtype=torch.long, device=device
        )
        det_target_reg = torch.zeros_like(det_last_reg)
        det_target_quality = torch.zeros_like(det_last_quality)

        loss_dict["loss_det_cls"] = F.cross_entropy(
            det_last_cls.reshape(-1, det_last_cls.shape[-1]),
            det_target_cls.reshape(-1)
        )
        loss_dict["loss_det_reg"] = F.l1_loss(det_last_reg, det_target_reg)
        loss_dict["loss_det_quality"] = F.l1_loss(det_last_quality, det_target_quality)

        map_last_cls = map_out["all_cls_scores"][-1]
        map_last_reg = map_out["all_reg_preds"][-1]

        map_target_cls = torch.zeros(
            map_last_cls.shape[0], map_last_cls.shape[1],
            dtype=torch.long, device=device
        )
        map_target_reg = torch.zeros_like(map_last_reg)

        loss_dict["loss_map_cls"] = F.cross_entropy(
            map_last_cls.reshape(-1, map_last_cls.shape[-1]),
            map_target_cls.reshape(-1)
        )
        loss_dict["loss_map_reg"] = F.l1_loss(map_last_reg, map_target_reg)

        motion_cls = motion_out["motion_cls"]
        motion_reg = motion_out["motion_reg"]
        plan_status = motion_out["plan_status"]
        plan_cls = motion_out["plan_cls"]
        plan_reg = motion_out["plan_reg"]

        if data_samples is not None and len(parsed_samples["gt_motion"]) == b and parsed_samples["gt_motion"][0] is not None:
            gt_motion = torch.stack([x.to(device) for x in parsed_samples["gt_motion"]], dim=0)
        else:
            gt_motion = torch.zeros_like(motion_reg)

        if data_samples is not None and len(parsed_samples["gt_plan"]) == b and parsed_samples["gt_plan"][0] is not None:
            gt_plan = torch.stack([x.to(device) for x in parsed_samples["gt_plan"]], dim=0)
        else:
            gt_plan = torch.zeros_like(plan_reg)

        loss_dict["loss_motion_cls"] = F.binary_cross_entropy_with_logits(
            motion_cls, torch.zeros_like(motion_cls)
        )
        loss_dict["loss_motion_reg"] = F.l1_loss(motion_reg, gt_motion)
        loss_dict["loss_plan_status"] = F.cross_entropy(
            plan_status, torch.zeros(b, dtype=torch.long, device=device)
        )
        loss_dict["loss_plan_cls"] = F.binary_cross_entropy_with_logits(
            plan_cls, torch.zeros_like(plan_cls)
        )
        loss_dict["loss_plan_reg"] = F.l1_loss(plan_reg, gt_plan)

        for i, d in enumerate(depth_out):
            loss_dict[f"loss_depth_{i}"] = d.abs().mean()

        return loss_dict

    def predict(self, inputs, data_samples=None):
        return self._forward_impl(inputs, data_samples)

    def _forward(self, inputs, data_samples=None):
        return self._forward_impl(inputs, data_samples)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise ValueError(f'Invalid mode: {mode}')