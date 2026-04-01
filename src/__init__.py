import importlib as _importlib
import sys as _sys


from src.datasets import (  # noqa: F401
    make_modular_addition,
    make_modular_multiplication,
    make_modular_subtraction,
    make_ring_addition,
    make_s3_group,
    make_d5_group,
    make_a4_group,
    make_s4_group,
    build_loaders,
    COMPLEXITY_MEASURES,
    get_complexity_score,
    get_complexity_score_v2,
)


try:
    import torch as _torch  # noqa: F401
except ImportError:
    pass
else:
    try:
        import transformers as _tf_mod
        if not hasattr(_tf_mod, "TRANSFORMERS_CACHE"):
            import os as _os
            _tf_mod.TRANSFORMERS_CACHE = _os.path.join(
                _os.path.expanduser("~"), ".cache", "huggingface", "transformers"
            )
    except Exception:
        pass

    from src.train import (
        train_experiment,
        multi_seed_experiment,
        multi_p_experiment,
        TrainConfig,
    )
    from src.analysis import (
        fourier_embedding_analysis,
        discrete_log_embedding_analysis,
        logit_attribution,
        activation_patch_heads,
        detect_grokking_phases,
        get_attention_patterns,
        probe_representation,
        cka_similarity,
        cka_matrix,
        aggregate_multi_seed,
        nonabelian_fourier_analysis,
        causal_dlog_verification,
        complexity_delay_regression,
        extract_weight_norms,
        bootstrap_confidence_interval,
        representation_formation_tracker,
        grokking_leading_indicator,
        controlled_complexity_ablation,
        describe_learned_circuit,
    )
    from src.visualise import (
        fig_grokking_curves,
        fig_grokking_curves_multiseed,
        fig_grokking_comparison,
        fig_fourier_spectrum,
        fig_dlog_spectrum,
        fig_dlog_analysis_panel,
        fig_logit_attribution,
        fig_activation_patching,
        fig_attention_patterns,
        fig_grokking_delay_comparison,
        fig_complexity_delay_errorbar,
        fig_cka_heatmap,
        fig_multi_p_delay,
        fig_representation_vs_grokking,
        fig_weight_norm_trajectory,
        fig_ablation_rank_order,
        fig_circuit_summary_table,
        fig_multi_seed_grokking_delay_cdf,
    )

__all__ = [
    "make_modular_addition", "make_modular_multiplication",
    "make_modular_subtraction", "make_ring_addition",
    "make_s3_group", "make_d5_group", "make_a4_group",
    "make_s4_group",
    "build_loaders",
    "COMPLEXITY_MEASURES",
    "get_complexity_score",
    "get_complexity_score_v2",
    "train_experiment", "multi_seed_experiment", "multi_p_experiment", "TrainConfig",
    "fourier_embedding_analysis", "discrete_log_embedding_analysis",
    "logit_attribution", "activation_patch_heads", "detect_grokking_phases",
    "get_attention_patterns", "probe_representation", "cka_similarity",
    "cka_matrix", "aggregate_multi_seed",
    "nonabelian_fourier_analysis",
    "causal_dlog_verification",
    "complexity_delay_regression",
    "extract_weight_norms",
    "bootstrap_confidence_interval",
    "representation_formation_tracker",
    "grokking_leading_indicator",
    "controlled_complexity_ablation",
    "describe_learned_circuit",
    "fig_grokking_curves", "fig_grokking_curves_multiseed",
    "fig_grokking_comparison", "fig_fourier_spectrum",
    "fig_dlog_spectrum", "fig_dlog_analysis_panel",
    "fig_logit_attribution", "fig_activation_patching",
    "fig_attention_patterns", "fig_grokking_delay_comparison",
    "fig_complexity_delay_errorbar", "fig_cka_heatmap", "fig_multi_p_delay",
    "fig_representation_vs_grokking",
    "fig_weight_norm_trajectory",
    "fig_ablation_rank_order",
    "fig_circuit_summary_table",
    "fig_multi_seed_grokking_delay_cdf",
]
