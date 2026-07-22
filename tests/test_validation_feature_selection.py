from src.training_features import validation_feature_name


def test_max_score_preserves_disentangled_validation_substitution():
    assert (
        validation_feature_name("disentangled_self_attn", "max_score")
        == "avg_self_attn_out"
    )


def test_paired_soft_routing_preserves_validation_head_features():
    assert (
        validation_feature_name("disentangled_self_attn", "paired_soft_routing")
        == "disentangled_self_attn"
    )


def test_unrelated_feature_name_is_unchanged():
    assert validation_feature_name("patch_tokens", "max_score") == "patch_tokens"
