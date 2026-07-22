import torch
import torch.nn.functional as F
from torch import nn

from src.loss import Contrastive, ContrastiveLoss


def direct_multi_positive_loss(scores, owner, num_images, scale):
    logits = scale * scores
    t2i = F.cross_entropy(logits, owner)
    per_image = []
    for image_index in range(num_images):
        positives = [
            logits[row, image_index]
            for row in range(logits.shape[0])
            if owner[row].item() == image_index
        ]
        numerator = torch.logsumexp(torch.stack(positives), dim=0)
        denominator = torch.logsumexp(logits[:, image_index], dim=0)
        per_image.append(-(numerator - denominator))
    return (t2i + torch.stack(per_image).mean()) / 2 / num_images**2


def test_logit_scale_is_a_fixed_buffer():
    criterion = Contrastive(ltype="infonce")

    assert "logit_scale" in dict(criterion.named_buffers())
    assert "logit_scale" not in dict(criterion.named_parameters())
    assert not criterion.logit_scale.requires_grad


def test_backward_does_not_create_logit_scale_gradient():
    criterion = Contrastive(ltype="infonce")
    scores = torch.randn(3, 3, requires_grad=True)

    criterion.compute_contrastive_loss(scores).backward()

    assert scores.grad is not None
    assert criterion.logit_scale.grad is None


def test_multi_positive_loss_matches_brute_force():
    criterion = Contrastive(ltype="multi_positive_infonce")
    criterion.logit_scale.data.fill_(0.0)
    scores = torch.tensor(
        [[2.0, 0.0], [1.0, -1.0], [-0.5, 1.5]], dtype=torch.float64
    )
    owner = torch.tensor([0, 0, 1])

    actual = criterion.compute_contrastive_loss(scores, owner, num_images=2)
    expected = direct_multi_positive_loss(scores, owner, 2, scale=1.0)

    torch.testing.assert_close(actual, expected)


def test_text_to_image_targets_caption_owner():
    criterion = Contrastive(ltype="multi_positive_infonce")
    criterion.logit_scale.data.fill_(0.0)
    scores = torch.tensor([[4.0, -4.0], [3.0, -3.0], [-2.0, 2.0]])
    owner = torch.tensor([0, 0, 1])
    loss = criterion.compute_contrastive_loss(scores, owner, 2)
    assert loss < 0.02


def test_image_to_text_numerator_includes_all_same_image_captions():
    scores = torch.tensor([[2.0, -1.0], [2.0, -1.0], [-2.0, 2.0]])
    owner = torch.tensor([0, 0, 1])
    logits = scores
    numerator = torch.logsumexp(logits[owner == 0, 0], dim=0)
    denominator = torch.logsumexp(logits[:, 0], dim=0)
    expected_image_zero = -(numerator - denominator)

    single_positive_numerator = logits[0, 0]
    single_positive_loss = -(single_positive_numerator - denominator)

    assert expected_image_zero < single_positive_loss


def test_batch_size_one_has_zero_loss():
    criterion = Contrastive(ltype="multi_positive_infonce")
    scores = torch.tensor([[0.2], [0.7], [-0.1]])
    owner = torch.zeros(3, dtype=torch.long)
    torch.testing.assert_close(
        criterion.compute_contrastive_loss(scores, owner, 1), torch.tensor(0.0)
    )


def test_rejects_image_without_positive_caption():
    criterion = Contrastive(ltype="multi_positive_infonce")
    with torch.no_grad():
        try:
            criterion.compute_contrastive_loss(
                torch.randn(2, 2), torch.zeros(2, dtype=torch.long), 2
            )
        except ValueError as error:
            assert "no positive captions" in str(error)
        else:
            raise AssertionError("Expected missing-positive ValueError")


def test_existing_infonce_is_numerically_unchanged():
    criterion = Contrastive(ltype="infonce")
    scores = torch.tensor([[1.0, 0.2], [-0.3, 0.8]], dtype=torch.float64)
    scale = criterion.logit_scale.exp()
    labels = torch.arange(2)
    expected = (
        F.cross_entropy(scale * scores, labels)
        + F.cross_entropy((scale * scores).t(), labels)
    ) / 2 / scores.shape[0] ** 2
    torch.testing.assert_close(criterion.compute_contrastive_loss(scores), expected)


def test_fixed_scale_infonce_matches_direct_reference():
    criterion = Contrastive(ltype="infonce")
    scores = torch.tensor([[0.8, -0.2], [0.1, 0.7]], dtype=torch.float64)
    logits = scores * (1 / 0.07)
    labels = torch.arange(2)
    expected = (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.t(), labels)
    ) / 2 / 2**2

    actual = criterion.compute_contrastive_loss(scores)

    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)


def test_fixed_scale_multi_positive_matches_direct_reference():
    criterion = Contrastive(ltype="multi_positive_infonce")
    scores = torch.tensor(
        [[0.7, -0.1], [0.5, 0.0], [-0.2, 0.8]], dtype=torch.float64
    )
    owner = torch.tensor([0, 0, 1])

    actual = criterion.compute_contrastive_loss(scores, owner, num_images=2)
    expected = direct_multi_positive_loss(scores, owner, 2, scale=1 / 0.07)

    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)


@torch.no_grad()
def test_fixed_scale_follows_score_device_and_dtype():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = Contrastive(ltype="infonce").to(device)
    scores = torch.eye(2, device=device, dtype=torch.float64)

    loss = criterion.compute_contrastive_loss(scores)

    assert loss.device == device
    assert loss.dtype == torch.float64


def test_existing_triplet_is_numerically_unchanged():
    criterion = Contrastive(ltype="triplet", margin=0.2, max_violation=False)
    scores = torch.tensor([[1.0, 0.7], [0.6, 0.9]])
    diagonal = scores.diag().view(2, 1)
    cost_s = (0.2 + scores - diagonal.expand_as(scores)).clamp(min=0)
    cost_im = (0.2 + scores - diagonal.t().expand_as(scores)).clamp(min=0)
    mask = torch.eye(2, dtype=torch.bool)
    expected = (
        cost_s.masked_fill(mask, 0).sum()
        + cost_im.masked_fill(mask, 0).sum()
    ) / scores.shape[0] ** 2
    torch.testing.assert_close(criterion.compute_contrastive_loss(scores), expected)


def test_legacy_similarity_call_does_not_receive_caption_mask_keyword():
    class LegacySimilarity(nn.Module):
        def forward(
            self,
            image,
            text,
            ret_similarity_matrix=True,
            self_attn_maps=None,
            cls=None,
            text_input_mask=None,
            return_index=False,
        ):
            return text @ image.t()

    criterion = ContrastiveLoss(LegacySimilarity(), ltype="infonce")
    loss = criterion(torch.eye(2), torch.eye(2))
    assert torch.isfinite(loss)
