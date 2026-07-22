def validation_feature_name(feature_name, alignment_strategy):
    if (
        feature_name == 'disentangled_self_attn'
        and alignment_strategy != 'paired_soft_routing'
    ):
        return 'avg_self_attn_out'
    return feature_name
