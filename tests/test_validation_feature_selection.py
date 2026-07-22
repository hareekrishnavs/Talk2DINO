from src.training_features import validation_feature_name


def test_max_score_preserves_disentangled_validation_substitution():
    assert (
        validation_feature_name("disentangled_self_attn", "max_score")
        == "avg_self_attn_out"
    )


def test_all_pairs_lse_preserves_validation_head_features():
    assert (
        validation_feature_name("disentangled_self_attn", "all_pairs_lse")
        == "disentangled_self_attn"
    )
