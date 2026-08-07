# ------------------------------------------------------------------------------
# Talk2DINO
# ------------------------------------------------------------------------------
import copy
import math
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange

from models.builder import MODELS
# from models.dinotext.gumbel import gumbel_sigmoid
from models.dinotext.modules import FeatureEncoder

import us
from src.e8_balanced_retrieval_adapter import compute_e8_scores
from src.retrieval_grounded_prototypes import fuse_prototype_scores


@MODELS.register_module()
class Sim2Mask(nn.Module):
    def __init__(self, init_w=1.0, init_b=0.0, gumbel_tau=1.0, learnable=True):
        super().__init__()
        self.init_w = init_w
        self.init_b = init_b
        self.gumbel_tau = gumbel_tau
        self.learnable = learnable

        assert not ((init_w is None) ^ (init_b is None))
        if learnable:
            self.w = nn.Parameter(torch.full([], float(init_w)))
            self.b = nn.Parameter(torch.full([], float(init_b)))
        else:
            self.w = init_w
            self.b = init_b

    def forward(self, x, deterministic=False):
        logits = x * self.w + self.b

        soft_mask = torch.sigmoid(logits)
        if deterministic:
            hard_mask = soft_mask.gt(0.5).type(logits.dtype)
        else:
            hard_mask = gumbel_sigmoid(logits, hard=True, tau=self.gumbel_tau)

        return hard_mask, soft_mask

    def extra_repr(self):
        return f'init_w={self.init_w}, init_b={self.init_b}, learnable={self.learnable}, gumbel_tau={self.gumbel_tau}'


class MaskerBackbone(nn.Module):
    """Masker image encoder backbone.
    """
    def __init__(self, clip_visual, freeze_idx):
        super().__init__()
        self.transformer = copy.deepcopy(clip_visual.transformer)
        self.transformer.resblocks = self.transformer.resblocks[freeze_idx:]

        for block in self.transformer.resblocks:
            if hasattr(block, "hook_handler"):
                block.hook_handler.remove()

        self.ln_post = copy.deepcopy(clip_visual.ln_post)
        self.proj = copy.deepcopy(clip_visual.proj)

        self.layers = len(self.transformer.resblocks)
        self.patch_size = clip_visual.patch_size

        self.output_dim = clip_visual.output_dim if self.proj is not None else clip_visual.width

    def forward(self, x, spatial=True, ignore_last_attn=True):
        if self.layers:
            x = self.transformer(x, ignore_last_attn=ignore_last_attn)

        x = x.permute(1, 0, 2)  # LND -> NLD

        if spatial:
            x = self.ln_post(x)
        else:
            x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

class MaskerImageFeatureEncoder(FeatureEncoder):
    def __init__(self, backbone: nn.Module, decoder: nn.Module, ignore_last_attn: bool = True):
        super().__init__()
        self.ignore_last_attn = ignore_last_attn
        self.patch_size = backbone.patch_size
        self.backbone = backbone
        self.decoder = decoder

        for resblock in self.backbone.transformer.resblocks:
            resblock.hook_handler = resblock.register_forward_hook(self.hook)

    def _encode(self, image, image_feat):
        H, W = image.shape[-2:]
        h = H // self.patch_size
        w = W // self.patch_size

        x = self.backbone(image_feat, spatial=True, ignore_last_attn=self.ignore_last_attn)  # BLC
        x = rearrange(x[:, 1:], "B (H W) C -> B C H W", H=h, W=w)
        x = self.decoder(x)

        return x

@MODELS.register_module()
class Masker(nn.Module):
    def __init__(self, backbone, decoder, image_proj, sim2mask, ignore_last_attn, **kwargs):
        super().__init__()
        self.ignore_last_attn = ignore_last_attn

        decoder["C"] = backbone.output_dim
        decoder = MODELS.build(decoder)
        decoder = nn.Sequential(OrderedDict([
            ("decoder", decoder),
            ("image_proj", image_proj)
        ]))

        self.image_encoder = MaskerImageFeatureEncoder(backbone, decoder, ignore_last_attn=ignore_last_attn)

        self.sim2mask = Sim2Mask(**sim2mask)

    def forward(self, image, image_feat, text_emb, deterministic=False):
        B = image.size(0)
        image_emb, feats = self.image_encoder(image, image_feat, ret_feats=True)  # [BCHW]

        image_emb_norm = us.normalize(image_emb, dim=1)
        text_emb_norm = us.normalize(text_emb, dim=-1)

        H, W = image_emb.shape[2:]
        D = dist.get_world_size()

        # simmap [B, B*D, H, W] where D is #devices
        all_text_emb_norm = us.gather_cat(text_emb_norm, grad=True, contiguous_grad=True)
        simmap = torch.einsum("bchw,nc->bnhw", image_emb_norm, all_text_emb_norm)
        mask, soft_mask = self.sim2mask(simmap, deterministic=deterministic)

        # mask [B, B*D, H, W] where D is #devices
        # positive global label
        pos_indices = torch.arange(B, dtype=torch.long, device=image_emb.device) + B * dist.get_rank()
        pos_mask = mask[torch.arange(B), pos_indices].unsqueeze(1)  # [B, 1, H, W]

        offdiag = torch.ones(B, B*D, dtype=torch.bool, device=mask.device)
        offdiag[torch.arange(B), pos_indices] = False

        soft_pos_mask = soft_mask[torch.arange(B), pos_indices].unsqueeze(1)
        soft_neg_mask = soft_mask.masked_select(offdiag[..., None, None]).view(B, B*D-1, H, W)

        masks = {
            "pos": pos_mask,  # [B, 1, H, W]

            "soft_pos": soft_pos_mask,
            "soft_neg": soft_neg_mask,
            "soft_all": soft_mask,  # [B, N, H, W]
        }

        return masks, image_emb, text_emb, feats

    @torch.no_grad()
    def forward_seg(self, image, image_feat, text_emb, deterministic=True, hard=False):
        """Make mask by 1:N matching

        Args:
            image [B, 3, H, W]
            image_feat [L, B, C]: CLIP features
            text_emb [N, C]
            deterministic (bool): deterministic inference flag for gumbel noise
            hard (bool): decide hard or soft returning segmentation mask.
                Note that soft mask is required for proper evaluation

        Return:
            mask [B, N, H', W'] (H' and W' are downsampled H/W)
        """
        image_emb = self.image_encoder(image, image_feat)  # [BCHW]

        image_emb = us.normalize(image_emb, dim=1)  # BCHW
        text_emb = us.normalize(text_emb, dim=-1)  # NC

        simmap = torch.einsum("b c h w, n c -> b n h w", image_emb, text_emb)

        hard_mask, soft_mask = self.sim2mask(simmap, deterministic=deterministic)
        mask = hard_mask if hard else soft_mask

        return mask, simmap

@MODELS.register_module()
class DINOTextMasker(nn.Module):
    def __init__(self, similarity_type="cosine"):
        super().__init__()
        self.sim2mask = DINOTextSim2Mask()
        self.sim2mask = self.sim2mask.eval()
        self.similarity_type = similarity_type

    @torch.no_grad()
    def forward_seg(self, image_feat, text_emb, deterministic=True, hard=False):
        """Make mask by 1:N matching

        Args:
            image [B, 3, H, W]
            image_feat [L, B, C]: CLIP features
            text_emb [N, K, C]
            deterministic (bool): deterministic inference flag for gumbel noise
            hard (bool): decide hard or soft returning segmentation mask.
                Note that soft mask is required for proper evaluation
            use_k_nn (bool): use kNN to segment
            k_nn (int): number of nearest neighbors for kNN segmentation

        Return:
            mask [B, N, H', W'] (H' and W' are downsampled H/W)
        """
        b, c, h, w = image_feat.shape
        n, c = text_emb.shape

        if self.similarity_type == "cosine":
            image_feat = us.normalize(image_feat, dim=1)  # BCHW
            # text_emb = us.normalize(text_emb, dim=-1)  # NKC
            simmap = torch.einsum("b c h w, n c -> b n h w", image_feat, text_emb)
        else:
            raise NotImplementedError("similarity type {} not implemented".format(self.similarity_type))

        hard_mask, soft_mask = self.sim2mask(simmap, deterministic=deterministic)
        mask = hard_mask if hard else soft_mask

        return mask, simmap

    @torch.no_grad()
    def forward_seg_with_prototypes(
        self,
        image_feat,
        text_emb,
        prototypes,
        valid_mask,
        confidence,
        *,
        prototype_temperature=0.10,
        prototype_fusion_weight=0.25,
        deterministic=True,
        hard=False,
    ):
        """Explicit RGTP path that fuses raw cosine scores before one sigmoid."""
        if prototype_fusion_weight == 0 or not torch.any(valid_mask):
            return self.forward_seg(
                image_feat,
                text_emb,
                deterministic=deterministic,
                hard=hard,
            )
        if (
            not math.isfinite(prototype_temperature)
            or prototype_temperature <= 0
        ):
            raise ValueError("prototype_temperature must be strictly positive")
        if (
            not math.isfinite(prototype_fusion_weight)
            or not 0 <= prototype_fusion_weight <= 1
        ):
            raise ValueError("prototype_fusion_weight must be in [0, 1]")

        image_feat = us.normalize(image_feat, dim=1)
        text_emb = text_emb.to(
            device=image_feat.device,
            dtype=image_feat.dtype,
        )
        prototypes = prototypes.to(
            device=image_feat.device,
            dtype=image_feat.dtype,
        )
        valid_mask = valid_mask.to(device=image_feat.device, dtype=torch.bool)
        confidence = confidence.to(
            device=image_feat.device,
            dtype=image_feat.dtype,
        )
        if prototypes.ndim != 3:
            raise ValueError("prototypes must have shape [C, K, D]")
        if valid_mask.shape != prototypes.shape[:2]:
            raise ValueError("valid_mask must have shape [C, K]")
        if confidence.shape != prototypes.shape[:1]:
            raise ValueError("confidence must have shape [C]")
        if text_emb.shape != (prototypes.shape[0], prototypes.shape[2]):
            raise ValueError(
                "text embeddings and prototypes must have matching [C, D]"
            )
        if not torch.isfinite(prototypes).all() or not torch.isfinite(
            confidence
        ).all():
            raise ValueError("prototype inputs must be finite")
        if not torch.isfinite(text_emb).all() or not torch.isfinite(
            image_feat
        ).all():
            raise ValueError("image and text embeddings must be finite")
        if torch.any((confidence < 0) | (confidence > 1)):
            raise ValueError("confidence must be in [0, 1]")

        base_score = torch.einsum(
            "b c h w, n c -> b n h w",
            image_feat,
            text_emb,
        )
        prototype_score = torch.einsum(
            "b d h w, n k d -> b n k h w",
            image_feat,
            prototypes,
        )
        final_score = fuse_prototype_scores(
            base_score.permute(0, 2, 3, 1),
            prototype_score.permute(0, 3, 4, 1, 2),
            valid_mask,
            confidence,
            prototype_fusion_weight=prototype_fusion_weight,
            prototype_temperature=prototype_temperature,
        ).permute(0, 3, 1, 2)

        hard_mask, soft_mask = self.sim2mask(
            final_score,
            deterministic=deterministic,
        )
        mask = hard_mask if hard else soft_mask
        return mask, final_score

    @torch.no_grad()
    def forward_seg_with_balanced_prototypes(
        self,
        image_feat,
        text_emb,
        prototypes,
        valid_mask,
        beta,
        *,
        prototype_temperature=0.10,
        responsibility_temperature=0.10,
        reliability_mode="entropy",
        deterministic=True,
        hard=False,
    ):
        """E8 target-conditioned fusion followed by exactly one sigmoid."""
        for name, value in (
            ("prototype_temperature", prototype_temperature),
            ("responsibility_temperature", responsibility_temperature),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and strictly positive")
        if (
            not isinstance(reliability_mode, str)
            or reliability_mode not in {"entropy", "constant_one"}
        ):
            raise ValueError(
                "reliability_mode must be 'entropy' or 'constant_one'"
            )
        if image_feat.ndim != 4:
            raise ValueError("image features must have shape [B, D, H, W]")
        if text_emb.ndim != 2:
            raise ValueError("text embeddings must have shape [C, D]")
        if prototypes.ndim != 3:
            raise ValueError("prototypes must have shape [C, K, D]")
        class_count, prototype_count, embedding_dim = prototypes.shape
        if text_emb.shape != (class_count, embedding_dim):
            raise ValueError(
                "text embeddings and prototypes must have matching [C, D]"
            )
        if image_feat.shape[1] != embedding_dim:
            raise ValueError(
                "image features and prototypes must have the same embedding "
                "dimension"
            )
        if valid_mask.shape != (class_count, prototype_count):
            raise ValueError("valid_mask must have shape [C, K]")
        if valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean")
        if beta.shape != (class_count,):
            raise ValueError("beta must have shape [C]")
        if not torch.is_tensor(beta) or not beta.is_floating_point():
            raise ValueError("beta must be a floating-point tensor")
        if not torch.isfinite(beta).all() or torch.any((beta < 0) | (beta > 1)):
            raise ValueError("beta must be finite and in [0, 1]")

        has_valid_prototype = valid_mask.any(dim=-1)
        if not torch.any(beta * has_valid_prototype.to(beta.dtype)):
            # Do not even normalize or recompute E3 scores in the exact-control
            # branch. This preserves bitwise-identical scores and masks.
            return self.forward_seg(
                image_feat,
                text_emb,
                deterministic=deterministic,
                hard=hard,
            )

        # This is the one and only image normalization in the E8 inference
        # path. It is the same operation used by ``forward_seg`` (E3).
        image_feat = us.normalize(image_feat, dim=1)
        text_emb = text_emb.to(
            device=image_feat.device,
            dtype=image_feat.dtype,
        )
        prototypes = prototypes.to(
            device=image_feat.device,
            dtype=image_feat.dtype,
        )
        valid_mask = valid_mask.to(device=image_feat.device)
        beta = beta.to(device=image_feat.device, dtype=image_feat.dtype)
        if not all(
            torch.isfinite(value).all()
            for value in (image_feat, text_emb, prototypes)
        ):
            raise ValueError("E8 inference embeddings must be finite")

        batch_size, _, height, width = image_feat.shape
        # Compute the E3 control exactly once and reuse this tensor in E8.
        base_score = torch.einsum(
            "b c h w, n c -> b n h w",
            image_feat,
            text_emb,
        )
        spatial_targets = image_feat.permute(0, 2, 3, 1).reshape(
            batch_size * height * width,
            embedding_dim,
        )
        flat_base_score = base_score.permute(1, 0, 2, 3).reshape(
            class_count,
            batch_size * height * width,
        )
        scores = compute_e8_scores(
            text_emb,
            spatial_targets,
            prototypes,
            beta,
            prototype_temperature=prototype_temperature,
            responsibility_temperature=responsibility_temperature,
            prototype_valid_mask=valid_mask,
            precomputed_base_score=flat_base_score,
            targets_are_normalized=True,
            reliability_mode=reliability_mode,
        )
        fused_score = (
            scores.final_score.reshape(
                class_count,
                batch_size,
                height,
                width,
            )
            .permute(1, 0, 2, 3)
        )
        effective_beta = scores.effective_beta.reshape(
            class_count,
            batch_size,
            height,
            width,
        ).permute(1, 0, 2, 3)
        # Preserve E3's score layout as well as its values. Some elementwise
        # CPU kernels can otherwise differ by a final bit across layouts.
        final_score = torch.empty_like(base_score)
        torch.where(
            effective_beta == 0,
            base_score,
            fused_score,
            out=final_score,
        )
        hard_mask, soft_mask = self.sim2mask(
            final_score,
            deterministic=deterministic,
        )
        mask = hard_mask if hard else soft_mask
        return mask, final_score


@MODELS.register_module()
class DINOTextSim2Mask(nn.Module):
    def __init__(self, gumbel_tau=1.0):
        super().__init__()
        self.gumbel_tau = gumbel_tau

    def forward(self, x, deterministic=False):
        soft_mask = torch.sigmoid(x)
        if deterministic:
            hard_mask = soft_mask.gt(0.5).type(x.dtype)
        else:
            hard_mask = gumbel_sigmoid(x, hard=True, tau=self.gumbel_tau)

        return hard_mask, soft_mask
