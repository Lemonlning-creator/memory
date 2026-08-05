"""Legacy REALTALK persona-simulation auxiliary runner.

This package is retained for reproducibility only. The active advisor Exp2 is
``src.experiments.exp2_predictive_empathy``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .runner import Exp2UserModelingConfig

__all__ = ["Exp2UserModelingConfig", "run_user_modeling_evaluation"]
LEGACY_AUXILIARY_ONLY = True


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .runner import (
            Exp2UserModelingConfig,
            run_user_modeling_evaluation,
        )

        return {
            "Exp2UserModelingConfig": Exp2UserModelingConfig,
            "run_user_modeling_evaluation": run_user_modeling_evaluation,
        }[name]
    raise AttributeError(name)
