"""Deterministic section-law identification and the bonded-frame lining example.

Runtime inputs live under data/ and configs/; numerical dependencies are
bundled under src/. Results are written to results/analysis/. The optional
jax_family and bayes modules retain the earlier posterior study, which is
not used by the current manuscript or its reproduction commands.
"""

__all__ = ["paths", "member_data", "section_family", "refit", "lining_inputs", "lining_model"]
