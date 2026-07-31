"""REALTALK-based user-modeling evaluation for revised Experiment 2."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .runner import Exp2UserModelingConfig

__all__ = ["Exp2UserModelingConfig", "run_user_modeling_evaluation"]


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
