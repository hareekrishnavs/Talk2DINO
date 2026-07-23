import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.model import CLIPLastLayer

class Contrastive(nn.Module):
    def __init__(self, sim=None, margin=0, max_violation=False, ltype='triplet'):
        super(Contrastive, self).__init__()
        self.margin = margin
        self.sim = sim
        self.max_violation = max_violation
        self.ltype = ltype
        
        self.register_buffer(
            "logit_scale",
            torch.tensor(np.log(1 / 0.07), dtype=torch.float32),
        )

    def compute_contrastive_loss(self, scores):
        if self.ltype in {'infonce', 'infonce_rdcd'}:
            # cosine similarity as logits
            logit_scale = self.logit_scale.to(
                device=scores.device,
                dtype=scores.dtype,
            ).exp()
            logits_per_image = logit_scale * scores
            logits_per_text = logits_per_image.t()

            # compute bidirectional CE loss
            num_logits = logits_per_image.shape[0]
            labels = torch.arange(num_logits, device=logits_per_image.device, dtype=torch.long)
            loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
                ) / 2

        elif self.ltype == 'triplet':
            diagonal = scores.diag().view(scores.size(0), 1)
            d1 = diagonal.expand_as(scores)
            d2 = diagonal.t().expand_as(scores)

            # compare every diagonal score to scores in its column
            # caption retrieval
            cost_s = (self.margin + scores - d1).clamp(min=0)
            # compare every diagonal score to scores in its row
            # image retrieval
            cost_im = (self.margin + scores - d2).clamp(min=0)

            # clear diagonals
            mask = torch.eye(scores.size(0)) > .5
            I = mask
            if torch.cuda.is_available():
                I = I.to(scores.device)
            cost_s = cost_s.masked_fill_(I, 0)
            cost_im = cost_im.masked_fill_(I, 0)

            # keep the maximum violating negative for each query
            if self.max_violation:
                cost_s = cost_s.max(1)[0]
                cost_im = cost_im.max(0)[0]

            loss = cost_s.sum() + cost_im.sum()
            
        else:
            raise ValueError(f'{self.ltype} not known!')
            
        return loss / scores.shape[0]**2 # normalization by the batch size**2

class ContrastiveLoss(Contrastive):
    """
    Compute contrastive loss
    """

    def __init__(self, sim, margin=0, max_violation=False, ltype='triplet'):
        super(ContrastiveLoss, self).__init__(sim=sim, margin=margin, max_violation=max_violation, ltype=ltype)
        

    def forward(self, im, s, return_similarity_mat=False, self_attn_maps=None, cls=None, text_input_mask=None, text_argmax=None, return_index=False, patch_tokens=None, return_rdcd_components=False):
        if self.ltype == 'infonce_rdcd':
            if patch_tokens is None or self_attn_maps is None:
                raise ValueError(
                    "infonce_rdcd requires patch_tokens and self_attn_maps"
                )
            if return_index:
                raise ValueError("infonce_rdcd does not support return_index")
            textual_embedding, visual_embedding = self.sim(
                im,
                s,
                ret_similarity_matrix=True,
                ret_embeds=True,
                self_attn_maps=None,
                cls=None,
                text_input_mask=text_input_mask,
            )
            components = compute_rdcd_components(
                textual_embedding,
                visual_embedding,
                patch_tokens,
                self_attn_maps,
                routing_temperature=self.sim.routing_temperature,
                dense_temperature=self.sim.dense_temperature,
                attention_map_format=self.sim.attention_map_format,
            )
            scores = components["pcrr_scores"]
            pcrr_loss = self.compute_contrastive_loss(scores)
            dense_loss = components["dense_loss"]
            loss = pcrr_loss + self.sim.dense_loss_weight * dense_loss
            components.update(
                {
                    "pcrr_scores": scores,
                    "pcrr_loss": pcrr_loss,
                    "total_loss": loss,
                }
            )
            to_return = [loss]
            if return_similarity_mat:
                to_return.append(scores)
            if return_rdcd_components:
                to_return.append(components)
            return tuple(to_return) if len(to_return) > 1 else to_return[0]

        # compute image-sentence score matrix
        if type(self.sim) == CLIPLastLayer:
            scores = self.sim(im, s, ret_similarity_matrix=True, self_attn_maps=self_attn_maps, cls=cls, text_input_mask=text_input_mask, text_argmax=text_argmax)
        else:
            if return_index:
                scores, index = self.sim(im, s, ret_similarity_matrix=True, self_attn_maps=self_attn_maps, cls=cls, text_input_mask=text_input_mask, return_index=return_index)
            else:
                scores = self.sim(im, s, ret_similarity_matrix=True, self_attn_maps=self_attn_maps, cls=cls, text_input_mask=text_input_mask, return_index=return_index)
        loss = self.compute_contrastive_loss(scores)
        
        to_return = [loss]
        if return_similarity_mat:
            to_return.append(scores)
        if return_index:
            to_return.append(index)
        if len(to_return) > 1:
            to_return = tuple(to_return)
        else:
            to_return = to_return[0]
        return to_return


def prepare_attention_probabilities(self_attn_maps, attention_map_format, eps=1e-8):
    if not torch.is_tensor(self_attn_maps) or self_attn_maps.ndim != 3:
        shape = getattr(self_attn_maps, "shape", None)
        raise ValueError(
            "self_attn_maps must have shape [B,H,P], but received "
            f"{tuple(shape) if shape is not None else type(self_attn_maps)}"
        )
    if not torch.isfinite(self_attn_maps).all():
        raise ValueError("self_attn_maps contains non-finite values")
    if attention_map_format == "probabilities":
        if (self_attn_maps < 0).any():
            raise ValueError("probability attention maps contain negative values")
        row_sums = self_attn_maps.sum(dim=-1, keepdim=True)
        if (row_sums <= 0).any():
            raise ValueError("probability attention maps contain a zero-sum row")
        attention_prob = self_attn_maps.clamp_min(0)
        attention_prob = attention_prob / row_sums.clamp_min(eps)
    elif attention_map_format == "logits":
        attention_prob = torch.softmax(self_attn_maps, dim=-1)
    else:
        raise ValueError(
            "attention_map_format must be 'probabilities' or 'logits', but "
            f"received {attention_map_format!r}"
        )
    if not torch.isfinite(attention_prob).all():
        raise ValueError("normalized attention maps contain non-finite values")
    return attention_prob


def compute_rdcd_components(
    text,
    heads,
    patch_tokens,
    self_attn_maps,
    *,
    routing_temperature,
    dense_temperature,
    attention_map_format,
    eps=1e-8,
):
    if routing_temperature <= 0 or dense_temperature <= 0:
        raise ValueError("routing_temperature and dense_temperature must be positive")
    if text.ndim != 2 or heads.ndim != 3:
        raise ValueError(
            f"RDCD requires text [B,D] and heads [B,H,D], got {tuple(text.shape)} "
            f"and {tuple(heads.shape)}"
        )
    if patch_tokens.ndim != 3 or self_attn_maps.ndim != 3:
        raise ValueError(
            "RDCD requires patch_tokens [B,P,D] and self_attn_maps [B,H,P], "
            f"got {tuple(patch_tokens.shape)} and {tuple(self_attn_maps.shape)}"
        )
    batch, heads_count, embed_dim = heads.shape
    if text.shape != (batch, embed_dim):
        raise ValueError("text and head batch/embedding dimensions do not match")
    if patch_tokens.shape[0] != batch or patch_tokens.shape[2] != embed_dim:
        raise ValueError("patch token batch/embedding dimensions do not match")
    if self_attn_maps.shape != (
        batch,
        heads_count,
        patch_tokens.shape[1],
    ):
        raise ValueError("attention-map H/P dimensions do not match heads/patches")
    for name, tensor in (
        ("text", text),
        ("heads", heads),
        ("patch_tokens", patch_tokens),
    ):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")

    attention_prob = prepare_attention_probabilities(
        self_attn_maps.to(device=text.device, dtype=text.dtype),
        attention_map_format,
        eps=eps,
    )
    positive_affinities = torch.einsum("bd,bhd->bh", text, heads)
    routing_weights = torch.softmax(
        positive_affinities / routing_temperature,
        dim=-1,
    )
    routed_visual = torch.einsum("bh,bhd->bd", routing_weights, heads)
    routed_visual = F.normalize(routed_visual, p=2, dim=-1)
    pcrr_scores = text @ routed_visual.transpose(0, 1)

    teacher_saliency = torch.einsum(
        "bh,bhp->bp",
        routing_weights.detach(),
        attention_prob,
    )
    teacher_saliency = teacher_saliency.clamp_min(eps)
    teacher_saliency = teacher_saliency / teacher_saliency.sum(
        dim=-1,
        keepdim=True,
    )
    teacher_saliency = teacher_saliency.detach()

    patches = F.normalize(
        patch_tokens.to(device=text.device, dtype=text.dtype),
        p=2,
        dim=-1,
    )
    patch_logits = torch.einsum("bd,bpd->bp", text, patches)
    student_log_prob = torch.log_softmax(
        patch_logits / dense_temperature,
        dim=-1,
    )
    dense_kl = F.kl_div(
        student_log_prob,
        teacher_saliency,
        reduction="batchmean",
    )
    dense_loss = dense_kl / batch**2
    return {
        "positive_affinities": positive_affinities,
        "routing_weights": routing_weights,
        "routed_visual": routed_visual,
        "pcrr_scores": pcrr_scores,
        "attention_prob": attention_prob,
        "teacher_saliency": teacher_saliency,
        "patches": patches,
        "patch_logits": patch_logits,
        "student_log_prob": student_log_prob,
        "dense_kl": dense_kl,
        "dense_loss": dense_loss,
    }


def main():
    pass
    
if __name__ == '__main__':
    main()
