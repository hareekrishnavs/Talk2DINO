def validation_feature_name(feature_name, alignment_strategy):
    if (
        feature_name == 'disentangled_self_attn'
        and alignment_strategy not in {
            'paired_soft_routing',
            'paired_soft_routing_rdcd',
        }
    ):
        return 'avg_self_attn_out'
    return feature_name
