import os
from math import sqrt
import yaml

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import BertModel, AutoTokenizer
import torchvision.transforms as T
import clip
import importlib

from models.builder import MODELS
from models.dinotext.pamr import PAMR
from models.dinotext.masker import DINOTextMasker
import us
from datasets import get_template

from src.model import ProjectionLayer, VisualProjectionLayer, CLIPLastLayer, DoubleMLP
from src.e6_prototype_bank import sha256_file
from src.hooks import average_text_tokens, get_vit_out, feats
from src.local_weights import DEFAULT_WEIGHT_DIR, load_local_clip, load_local_vision_backbone, load_state_dict_from_local_file, resolve_weight_path
from src.retrieval_grounded_prototypes import (
    PrototypeBank,
    RGTPSettings,
    RetrievalGroundedPrototypes,
)
from src.e7_prototype_adapter import (
    LearnedRetrievalSettings,
    load_learned_retrieval_prototypes,
)
from src.e8_balanced_retrieval_adapter import (
    BalancedRetrievalSettings,
    load_balanced_retrieval_prototypes,
)
from src.e9_sparse_region_alignment import load_e9_adapter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _square_grid_side(patch_count):
    side = int(sqrt(patch_count))
    if side * side != patch_count:
        raise ValueError(
            f"projected patch count {patch_count} does not form a square "
            "inference grid"
        )
    return side


@MODELS.register_module()
class DINOText(nn.Module):
    
    def get_self_attention(self, module, input, output):
        self.feats['self_attn'] = output
        
    def get_clip_second_last_dense_out(self, model: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        self.feats['clip_second_last_out'] = output
        self.feats['clip_second_last_out'].to(dtype=torch.float32)
    
    def get_all_out_tokens(self, model: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        self.feats['clip_txt_out_tokens'] = output
        
    def __init__(
            self, model_name, resize_dim, clip_model_name, proj_class, proj_name, proj_model, avg_self_attn_token=False, disentangled_self_attn_token=True, pre_trained=True,
            unfreeze_last_text_layer=False, is_eval=True, use_avg_text_token=False, keep_cls=False, keep_end_seq=False, with_bg_clean=False,
            weight_dir=DEFAULT_WEIGHT_DIR, backbone_weights=None, clip_model_path=None,
            retrieval_grounded_prototypes=None,
            learned_retrieval_prototypes=None,
            balanced_retrieval_prototypes=None,
            sparse_region_alignment=None,
            **kwargs
    ):
        super().__init__()
        if sparse_region_alignment is not None:
            if not isinstance(sparse_region_alignment, dict) or set(
                sparse_region_alignment
            ) != {"adapter_path", "adapter_sha256", "source_git_commit"}:
                raise ValueError(
                    "sparse_region_alignment must contain exactly adapter_path, "
                    "adapter_sha256, and source_git_commit"
                )
        retrieval_mode_count = sum(
            option is not None
            for option in (
                retrieval_grounded_prototypes,
                learned_retrieval_prototypes,
                balanced_retrieval_prototypes,
                sparse_region_alignment,
            )
        )
        if retrieval_mode_count > 1:
            raise ValueError(
                "E6, E7, E8, and E9 inference modes are "
                "mutually exclusive"
            )
        projection_config_path = os.path.join('configs', f"{proj_class}.yaml")
        projection_checkpoint_path = os.path.join("weights", f"{proj_name}.pth")
        self.sparse_region_adapter = None
        if sparse_region_alignment is not None:
            if not pre_trained:
                raise ValueError("E9 requires the frozen E3 projection")
            self.sparse_region_adapter = load_e9_adapter(
                sparse_region_alignment["adapter_path"],
                device=device,
                require_production=True,
                expected_e3_identity={
                    "config_sha256": sha256_file(projection_config_path),
                    "checkpoint_sha256": sha256_file(
                        projection_checkpoint_path
                    ),
                },
                expected_checkpoint_sha256=sparse_region_alignment[
                    "adapter_sha256"
                ],
                expected_source_git_commit=sparse_region_alignment[
                    "source_git_commit"
                ],
            )
        self.feats = {}
        self.model_name = model_name
        # loading the model
        
        if 'dinov2' in model_name or 'dinov3' in model_name:
            self.model = load_local_vision_backbone(model_name, resize_dim, backbone_weights, weight_dir)
            
        elif 'mae' in model_name or 'sam' in model_name or 'clip' in model_name or 'dino' in model_name:
            self.model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,  # remove classifier nn.Linear
                img_size=resize_dim
            )
            if backbone_weights is None:
                raise FileNotFoundError(f"Missing local backbone weights. Configuration field: backbone_weights (model={model_name}).")
            load_state_dict_from_local_file(self.model, resolve_weight_path(backbone_weights, weight_dir), f"backbone weights for {model_name}")
            
            if 'sam' in model_name:
                self.model.blocks[-1].register_forward_hook(get_vit_out)
        else:
            raise Exception("Unknown ViT model")
        # self.model.eval()
        mean = (0.485, 0.456, 0.406) if not 'clip' in model_name else (0.4815, 0.4578, 0.4082)
        std = (0.229, 0.224, 0.225) if not 'clip' in model_name else (0.2686, 0.2613, 0.2758)
        self.image_transforms = T.Compose([
            T.Resize((resize_dim, resize_dim)),
            lambda x: T.ToTensor()(x) if not isinstance(x, torch.Tensor) else x / 255.0,  # ensure tensor
            T.Normalize(mean, std),
        ])
        
        self.model.to(device)
        self.model.requires_grad_(False)
        
        self.clip_model_name = clip_model_name
        if 'bert' in self.clip_model_name:
            self.clip_model = BertModel.from_pretrained(self.clip_model_name, output_hidden_states = False)
            # load the corresponding wordtokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.clip_model_name)
        else:
            self.clip_model, _ = load_local_clip(clip_model_name, device=device, model_path=clip_model_path, weight_dir=weight_dir)
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)
        if unfreeze_last_text_layer:
            for param in self.clip_model.transformer.resblocks[-1].parameters():
                param.requires_grad = True
            for param in self.clip_model.ln_final.parameters():
                param.requires_grad = True
            self.clip_model.text_projection.requires_grad = True
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        with open(projection_config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)['model']
            
        ProjClass = getattr(importlib.import_module('src.model'), proj_model)
        self.proj = ProjClass.from_config(config)
        if type(self.proj) == CLIPLastLayer:
            self.clip_model.transformer.resblocks[-2].register_forward_hook(self.get_clip_second_last_dense_out)
        
            
        if pre_trained:
            self.proj.load_state_dict(torch.load(projection_checkpoint_path, 'cpu'))
        self.proj.to(device)

        self.rgtp = None
        self.rgtp_context = None
        if retrieval_grounded_prototypes is not None:
            if not pre_trained:
                raise ValueError(
                    "retrieval-grounded prototypes require the frozen E3 "
                    "projection checkpoint"
                )
            if not isinstance(retrieval_grounded_prototypes, dict):
                raise TypeError(
                    "retrieval_grounded_prototypes must be a configuration mapping"
                )
            rgtp_config = dict(retrieval_grounded_prototypes)
            try:
                bank_path = rgtp_config.pop("bank_path")
            except KeyError as error:
                raise ValueError(
                    "retrieval_grounded_prototypes.bank_path is required"
                ) from error
            settings = RGTPSettings.from_mapping(rgtp_config)
            bank = PrototypeBank.load(
                bank_path,
                expected_config_sha256=sha256_file(
                    projection_config_path
                ),
                expected_checkpoint_sha256=sha256_file(
                    projection_checkpoint_path
                ),
            )
            if bank.metadata["e3_checkpoint_name"] != os.path.basename(
                projection_checkpoint_path
            ):
                raise ValueError(
                    "prototype bank E3 checkpoint name does not match the "
                    "evaluation projection checkpoint"
                )
            self.rgtp = RetrievalGroundedPrototypes(bank, settings)

        self.learned_rpa = None
        self.learned_rpa_context = None
        self.learned_rpa_has_effect = False
        if learned_retrieval_prototypes is not None:
            if self.rgtp is not None:
                raise ValueError("E6 RGTP and E7 learned RPA are mutually exclusive")
            if not pre_trained:
                raise ValueError(
                    "learned retrieval prototypes require the frozen E3 projection"
                )
            if not isinstance(learned_retrieval_prototypes, dict):
                raise TypeError(
                    "learned_retrieval_prototypes must be a configuration mapping"
                )
            learned_config = dict(learned_retrieval_prototypes)
            try:
                bank_path = learned_config.pop("bank_path")
                adapter_path = learned_config.pop("adapter_path")
            except KeyError as error:
                raise ValueError(
                    "learned_retrieval_prototypes requires bank_path and adapter_path"
                ) from error
            settings = LearnedRetrievalSettings.from_mapping(learned_config)
            self.learned_rpa = load_learned_retrieval_prototypes(
                bank_path,
                adapter_path,
                settings,
                e3_config_sha256=sha256_file(projection_config_path),
                e3_checkpoint_sha256=sha256_file(projection_checkpoint_path),
                device=device,
            )

        self.balanced_rpa = None
        self.balanced_rpa_context = None
        self.balanced_rpa_has_effect = False
        if balanced_retrieval_prototypes is not None:
            if self.rgtp is not None or self.learned_rpa is not None:
                raise ValueError(
                    "E8 balanced retrieval, E7 learned RPA, and E6 RGTP "
                    "are mutually exclusive"
                )
            if not pre_trained:
                raise ValueError(
                    "balanced retrieval prototypes require the frozen E3 "
                    "projection"
                )
            if not isinstance(balanced_retrieval_prototypes, dict):
                raise TypeError(
                    "balanced_retrieval_prototypes must be a configuration "
                    "mapping"
                )
            balanced_config = dict(balanced_retrieval_prototypes)
            try:
                bank_path = balanced_config.pop("bank_path")
                adapter_path = balanced_config.pop("adapter_path")
            except KeyError as error:
                raise ValueError(
                    "balanced_retrieval_prototypes requires bank_path and "
                    "adapter_path"
                ) from error
            settings = BalancedRetrievalSettings.from_mapping(balanced_config)
            self.balanced_rpa = load_balanced_retrieval_prototypes(
                bank_path,
                adapter_path,
                settings,
                e3_config_sha256=sha256_file(projection_config_path),
                e3_checkpoint_sha256=sha256_file(
                    projection_checkpoint_path
                ),
                device=device,
            )
        
        self.masker = DINOTextMasker(similarity_type="cosine")
        self.masker = self.masker.eval()
        
        self.pamr = None
                
        self.avg_self_attn_token = avg_self_attn_token
        self.disentangled_self_attn_token = disentangled_self_attn_token
        
        if self.avg_self_attn_token or self.disentangled_self_attn_token or is_eval:
            self.model.blocks[-1].attn.qkv.register_forward_hook(self.get_self_attention)
            self.num_global_tokens = 5 if 'reg' in model_name or 'dinov3' in model_name else 1
            if 'sam' in self.model_name:
                self.num_global_tokens = 0
            if 'dinov3' in self.model_name:
                if 'vit_base' in self.model_name:
                    self.num_attn_heads = 12
                elif 'vit_large' in self.model_name:
                    self.num_attn_heads = 16
                else:
                    raise Exception("Unknown dinov3 model")
            else:
                self.num_attn_heads = self.model.num_heads
            self.scale = 0.125
        
        self.use_avg_text_token = use_avg_text_token
        if self.use_avg_text_token:
            self.feats = {}
            # in this case we register a forward hook with the aim of getting all the tokens and not only the cls
            self.clip_model.ln_final.register_forward_hook(self.get_all_out_tokens)
            self.keep_cls = keep_cls
            self.keep_end_seq = keep_end_seq        
            
        self.with_bg_clean = with_bg_clean    

    
    def process_self_attention(self, output, batch_size, num_tokens, num_attn_heads, embed_dim, scale, num_global_tokens, ret_self_attn_maps=False):
        qkv = output.reshape(batch_size, num_tokens, 3, num_attn_heads, embed_dim // num_attn_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0] * scale, qkv[1]
        attn = q @ k.transpose(-2, -1)
        self_attn_maps = attn[:, : , 0, num_global_tokens:]
        self_attn = self_attn_maps.mean(dim=1)
        self_attn = self_attn.softmax(dim=-1)
        if ret_self_attn_maps:
            return self_attn, self_attn_maps
        else:
            return self_attn
    
    def encode_text(self, tokenized_texts):
        if type(self.proj) == CLIPLastLayer:
            self.clip_model.encode_text(tokenized_texts)
            x = self.feats['clip_second_last_out']
            x = x.to(dtype=torch.float32)
        else:
            x = self.clip_model.encode_text(tokenized_texts)
        return x
    
    def encode_image(self, images):
        batch_size, _, _, _ = images.shape
        self_attn_maps = None
        x = self.model(images, is_training=(self.avg_self_attn_token or self.disentangled_self_attn_token))
        batch_size, num_tokens, embed_dim = x['x_norm_patchtokens'].shape
        num_tokens = num_tokens + self.num_global_tokens
        if self.avg_self_attn_token or self.disentangled_self_attn_token:
            self_attn, self_attn_maps = self.process_self_attention(self.feats['self_attn'], batch_size, num_tokens, self.num_attn_heads, embed_dim, self.scale, self.num_global_tokens, ret_self_attn_maps=True)
        if self.avg_self_attn_token:
            x = (self_attn.unsqueeze(-1) * x['x_norm_patchtokens']).mean(dim=1)
        elif self.disentangled_self_attn_token:
            self_attn_maps = self_attn_maps.softmax(dim=-1)
            x = (x['x_norm_patchtokens'].unsqueeze(1) * self_attn_maps.unsqueeze(-1)).mean(dim=2)

        return x, self_attn_maps

    def forward(self, image, text, return_logit_scale=False):
        with torch.no_grad():
            txt_embed = self.encode_text(text)
            
        img_embed, self_attn_maps = self.encode_image(image)
        
        if type(self.proj) == CLIPLastLayer:
            img_embed, txt_embed = self.proj(img_embed, txt_embed, ret_embeds=True, self_attn_maps=self_attn_maps, text_argmax=text.argmax(dim=-1))
        else:
            img_embed, txt_embed = self.proj(img_embed, txt_embed, ret_embeds=True, self_attn_maps=self_attn_maps)
        
        if return_logit_scale:
            return txt_embed, img_embed, self.logit_scale

        return txt_embed, img_embed
        
    @torch.no_grad()
    def build_dataset_class_tokens(self, template_set, classnames):
        tokens = []
        templates = get_template(template_set)
        for classname in classnames:
            if 'bert' not in self.clip_model_name:
                tokens.append(
                    clip.tokenize([template.format(classname) for template in templates])
                )
            else:
                tokens.append(self.tokenizer([template.format(classname) for template in templates], return_tensors='pt', padding='max_length')['input_ids'])
        # [N, T, L], N: number of instance, T: number of captions (including ensembled), L: sequence length
        tokens = torch.stack(tokens)

        return tokens

    @torch.no_grad()
    def build_text_embedding(self, text, return_raw=False):
        """
        Args:
            text (torch.Tensor): [NUM_CLASSES, NUM_TEMPLATES, CONTEXT_LENGTH] text tokens

        Returns:
            text_embs
        """
        text = text.to(next(self.parameters()).device)
        num_classes, num_templates = text.shape[:2]
        text_argmax = text.argmax(dim=-1)
        text_argmax = rearrange(text_argmax, 'n t -> (n t)', n=num_classes, t=num_templates)
        text = rearrange(text, 'n t l -> (n t) l', n=num_classes, t=num_templates)
        # chunked inference for memory limitation
        chunk_size = 32
        N = text.size(0)
        if type(self.proj) == CLIPLastLayer:
            text_embs = torch.cat([
            self.proj.project_clip_txt(self.encode_text(text[i:i + chunk_size]).permute(1, 0, 2), text_argmax=text_argmax[i:i + chunk_size])
            for i in range(0, N, chunk_size)
        ])
        else:
            if not self.use_avg_text_token:
                # performing classification using CLS textual token
                if 'bert' not in self.clip_model_name:
                    text_embs = torch.cat([
                        self.clip_model.encode_text(text[i:i + chunk_size])
                        for i in range(0, N, chunk_size)
                    ])
                else:
                    # encoding with BERT
                    text_embs = []
                    for i in range(0, N, chunk_size):
                        outputs = self.clip_model(text[i:i + chunk_size])
                        text_embs.append(outputs['pooler_output'])
                    text_embs = torch.cat(text_embs)
            else:
                # using text token average
                text_embs = []
                for i in range(0, N, chunk_size):
                    self.clip_model.encode_text(text[i:i + chunk_size])
                    text_embs.append(average_text_tokens(self.feats['clip_txt_out_tokens'] @ self.clip_model.text_projection, text[i:i + chunk_size] > 0, self.keep_cls, self.keep_end_seq))
                text_embs = torch.cat(text_embs)
        # [N, T, C]
        text_embs = rearrange(text_embs, '(n t) c -> n t c', n=num_classes, t=num_templates)
        # [N, C]
        text_embs = text_embs.mean(dim=1).float()
        raw_text_embs = text_embs
        if type(self.proj) == ProjectionLayer or type(self.proj) == DoubleMLP:
            text_embs = self.proj.project_clip_txt(text_embs)
        text_embs = us.normalize(text_embs, dim=-1)

        if return_raw:
            if type(self.proj) != ProjectionLayer:
                raise TypeError(
                    "retrieval-grounded prototypes require ProjectionLayer"
                )
            return us.normalize(raw_text_embs, dim=-1), text_embs
        return text_embs

    @torch.no_grad()
    def build_retrieval_grounded_prototypes(
        self,
        raw_text_embeddings,
        mapped_text_embeddings,
    ):
        if self.rgtp is None:
            raise RuntimeError("retrieval-grounded prototypes are not enabled")
        self.rgtp_context = self.rgtp.generate(
            raw_text_embeddings,
            mapped_text_embeddings,
        )
        return self.rgtp_context

    @torch.no_grad()
    def build_learned_retrieval_prototypes(
        self,
        raw_text_embeddings,
        mapped_text_embeddings,
    ):
        if self.learned_rpa is None:
            raise RuntimeError("E7 learned retrieval prototypes are not enabled")
        self.learned_rpa_context = self.learned_rpa.generate(
            raw_text_embeddings,
            mapped_text_embeddings,
        )
        self.learned_rpa_has_effect = bool(
            torch.any(self.learned_rpa_context.beta).detach().cpu()
        )
        return self.learned_rpa_context

    @torch.no_grad()
    def build_balanced_retrieval_prototypes(
        self,
        raw_text_embeddings,
        mapped_text_embeddings,
    ):
        if self.balanced_rpa is None:
            raise RuntimeError(
                "E8 balanced retrieval prototypes are not enabled"
            )
        self.balanced_rpa_context = self.balanced_rpa.generate(
            raw_text_embeddings,
            mapped_text_embeddings,
        )
        self.balanced_rpa_has_effect = bool(
            torch.any(
                self.balanced_rpa_context.beta
                * self.balanced_rpa_context.valid_mask.any(dim=-1).to(
                    self.balanced_rpa_context.beta.dtype
                )
            )
            .detach()
            .cpu()
        )
        return self.balanced_rpa_context

    def apply_pamr(self, image, mask):
        image = F.interpolate(image, mask.shape[-2:], mode="bilinear", align_corners=True)
        if self.pamr is None:
            pamr_iter = 10
            pamr_kernel = [1, 2, 4, 8, 12, 24]
            self.pamr = PAMR(pamr_iter, pamr_kernel)
            self.pamr.eval()
            self.pamr.to(next(self.parameters()).device)

        mask = self.pamr(image, mask)
        return mask

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b
    
    @torch.no_grad()
    def generate_masks(
            self, image, text_emb, apply_pamr=False, lambda_bg=0.2,
            # kp_w=0.3,
    ):
        """Generate masks for each text embeddings

        Args:
            image [B, 3, H, W]

        Returns:
            softmask [B, N, H, W]: softmasks for each text embeddings
        """

        H, W = image.shape[2:]  # original image shape

        # padded image size
        pH, pW = image.shape[2:]
        batch_size = image.shape[0]

        image = image[:, [2, 1, 0], :, :]  # BGR to RGB
        ori_image = image.clone()
        
        img_preprocessed = self.image_transforms(image).to(next(self.parameters()).device)
        if 'dinov2' in self.model_name:
            image_feat = self.model.forward_features(img_preprocessed)['x_norm_patchtokens']
        elif 'dinov3' in self.model_name:
            image_feat = self.model.forward_features(img_preprocessed)[:, 5:, :]
        elif 'mae' in self.model_name or 'clip' in self.model_name or 'dino' in self.model_name:
            image_feat = self.model.forward_features(img_preprocessed)[:, 1:, :]
        elif 'sam' in self.model_name:
            self.model.forward_features(img_preprocessed)
            image_feat = feats['vit_out'].reshape(feats['vit_out'].shape[0], feats['vit_out'].shape[1]**2, feats['vit_out'].shape[-1]) # BS x N_PATCHES x EMBED_DIM
              
        batch_size, num_tokens, embed_dim = image_feat.shape
        if type(self.proj) == VisualProjectionLayer:
            image_feat = self.proj.project_dino(image_feat.float())
        if type(self.proj) == DoubleMLP:
            image_feat = self.proj.project_visual(image_feat.float())
        projected_patch_tokens = image_feat
        b, np, c = image_feat.shape
        np_h = np_w = _square_grid_side(np)
        image_feat = image_feat.reshape(b, np_h, np_w, c).permute(0, 3, 1, 2)
        
        self_attn, self_attn_maps = self.process_self_attention(self.feats['self_attn'], batch_size, num_tokens + self.num_global_tokens, self.num_attn_heads, embed_dim, self.scale, self.num_global_tokens, ret_self_attn_maps=True)
        if (
            self.rgtp is None
            and self.learned_rpa is None
            and self.balanced_rpa is None
            and self.sparse_region_adapter is None
        ):
            mask, simmap = self.masker.forward_seg(
                image_feat,
                text_emb,
                hard=False,
            )
        elif self.sparse_region_adapter is not None:
            attention_prior = self_attn_maps.softmax(dim=-1).mean(dim=1)
            attention_prior = attention_prior / attention_prior.sum(
                dim=-1, keepdim=True
            ).clamp_min(torch.finfo(attention_prior.dtype).tiny)
            patch_grid = projected_patch_tokens.reshape(
                b, np_h, np_w, c
            ).permute(0, 3, 1, 2)
            mask, simmap = self.masker.forward_seg_with_sparse_region_alignment(
                patch_grid,
                text_emb,
                attention_prior,
                self.sparse_region_adapter,
                hard=False,
            )
        elif self.rgtp is not None:
            if self.rgtp_context is None:
                raise RuntimeError(
                    "RGTP class prototypes must be built before mask generation"
                )
            mask, simmap = self.masker.forward_seg_with_prototypes(
                image_feat,
                text_emb,
                self.rgtp_context.prototypes,
                self.rgtp_context.valid_mask,
                self.rgtp_context.confidence,
                prototype_temperature=self.rgtp.settings.prototype_temperature,
                prototype_fusion_weight=(
                    self.rgtp.settings.prototype_fusion_weight
                ),
                hard=False,
            )
        elif self.learned_rpa is not None:
            if self.learned_rpa_context is None:
                raise RuntimeError(
                    "E7 class prototypes must be built before mask generation"
                )
            if not self.learned_rpa_has_effect:
                # Exact E3 fallback for beta_max=0 or an all-empty retrieval.
                mask, simmap = self.masker.forward_seg(
                    image_feat,
                    text_emb,
                    hard=False,
                )
            else:
                mask, simmap = self.masker.forward_seg_with_prototypes(
                    image_feat,
                    text_emb,
                    self.learned_rpa_context.prototypes,
                    self.learned_rpa_context.valid_mask,
                    self.learned_rpa_context.beta,
                    prototype_temperature=(
                        self.learned_rpa.settings.prototype_temperature
                    ),
                    prototype_fusion_weight=1.0,
                    hard=False,
                )
        else:
            if self.balanced_rpa_context is None:
                raise RuntimeError(
                    "E8 class prototypes must be built before mask generation"
                )
            if not self.balanced_rpa_has_effect:
                # Exact E3 fallback for beta_max=0 or all-empty retrieval.
                mask, simmap = self.masker.forward_seg(
                    image_feat,
                    text_emb,
                    hard=False,
                )
            else:
                mask, simmap = (
                    self.masker.forward_seg_with_balanced_prototypes(
                        image_feat,
                        text_emb,
                        self.balanced_rpa_context.prototypes,
                        self.balanced_rpa_context.valid_mask,
                        self.balanced_rpa_context.beta,
                        prototype_temperature=(
                            self.balanced_rpa.settings.prototype_temperature
                        ),
                        responsibility_temperature=(
                            self.balanced_rpa.settings
                            .responsibility_temperature
                        ),
                        hard=False,
                    )
                )
        
        if self.with_bg_clean:
            mask = self.similarity_assignment_weighted(mask, image_feat, self_attn_maps, text_emb, lambda_bg)

        # resize
        mask = F.interpolate(mask, (pH, pW), mode='bilinear', align_corners=True)  # [B, N, H, W]

        if apply_pamr:
            for c in range(0, mask.shape[1], 30):
                mask[:, c:c + 30] = self.apply_pamr(ori_image, mask[:, c:c + 30])

        assert mask.shape[2] == H and mask.shape[3] == W, f"shape mismatch: ({H}, {W}) / {mask.shape}"

        return mask, simmap
    
    def similarity_assignment_weighted(self, mask, image_feat, self_attn_maps, text_emb, lambda_bg=0.2):
        bs, c, h, w = image_feat.shape
        bs, num_classes, h, w = mask.shape
        bs, num_heads, hw = self_attn_maps.shape
        image_feat = image_feat.reshape(bs, c, hw)
        num_classes, c = text_emb.shape
        avg_head_embed = (self_attn_maps.unsqueeze(2) * image_feat.unsqueeze(1)).mean(dim=-1)
        avg_head_embed = avg_head_embed / avg_head_embed.norm(dim=-1, keepdim=True)
        avg_head_embed = avg_head_embed.permute(0, 2, 1) # [B, C, M]
        head_text_sim = text_emb.unsqueeze(0) @ avg_head_embed # [B, M, N]
        head_text_sim = (head_text_sim).softmax(dim=-1)
        head_text_sim_sum = head_text_sim.sum(dim=-1)
        
        self_attn_maps_repeat = self_attn_maps.unsqueeze(1).repeat(1, num_classes, 1, 1)
        head_text_sim_repeat = head_text_sim.unsqueeze(-1).repeat(1, 1, 1, hw)
        avg_self_attn_per_class = (self_attn_maps_repeat * head_text_sim_repeat).sum(dim=2) / head_text_sim_sum.unsqueeze(-1).repeat(1, 1, hw)
        avg_self_attn_per_class = avg_self_attn_per_class.softmax(dim=-1)
        
        min_self_attn = avg_self_attn_per_class.min().item()
        max_self_attn = avg_self_attn_per_class.max().item()
        max_self_attn = max(max_self_attn, max_self_attn - min_self_attn)
        avg_self_attn_per_class = avg_self_attn_per_class - min_self_attn
        avg_self_attn_per_class = avg_self_attn_per_class / max_self_attn
        avg_self_attn_per_class = avg_self_attn_per_class * (mask.max() - mask.min()) + mask.min()
        mask = mask.reshape(num_classes, hw) # [N, P]
        mask_output = (mask + lambda_bg * avg_self_attn_per_class).reshape(bs, num_classes, h, w) / (1 + lambda_bg)
        return mask_output
