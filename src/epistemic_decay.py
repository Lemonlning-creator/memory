"""
Epistemic Value Decay: omega(t)

Implements the spontaneous decay of epistemic value over time (Innovation 3).
As the agent accumulates knowledge about the user through the 5-layer profile,
the marginal value of learning new information decreases. omega(t) modulates
the Explore-vs-Exploit decision in the empathy alignment reasoning.

Decay model:
    omega(t) = omega_0 * exp(-lambda * t) * (1 - completeness)

where:
    omega_0      = initial epistemic value (default 1.0)
    lambda       = decay rate (default 0.01 per interaction turn)
    t            = number of interaction turns since the relationship started
    completeness = profile completeness in [0, 1], measured by the proportion
                   of populated attributes across all 5 layers

The formula combines two decay pressures:
  1. Temporal decay: older relationships have less novel information to discover.
  2. Completeness decay: a richer profile means less room for new discoveries.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional


# Five layers of the hierarchical user profile
PROFILE_LAYERS = ("core", "regulation", "cognitive_style", "behavior_preference", "social_physical")


def compute_profile_completeness(static_profile: Dict[str, Any],
                                 max_attrs_per_layer: int = 8) -> float:
    """Compute profile completeness in [0, 1].

    Measures the fraction of populated attributes relative to a theoretical
    maximum (5 layers x max_attrs_per_layer).
    """
    if not static_profile:
        return 0.0

    total_possible = len(PROFILE_LAYERS) * max_attrs_per_layer
    populated = 0
    for layer in PROFILE_LAYERS:
        section = static_profile.get(layer, {})
        if isinstance(section, dict):
            for key, val in section.items():
                if isinstance(val, dict) and val.get("value"):
                    populated += 1
                elif val:
                    populated += 1

    completeness = min(populated / total_possible, 1.0) if total_possible > 0 else 0.0
    return round(completeness, 4)


def compute_omega(
    interaction_count: int,
    static_profile: Optional[Dict[str, Any]] = None,
    omega_0: float = 1.0,
    decay_lambda: float = 0.01,
    max_attrs_per_layer: int = 8,
) -> float:
    """Compute the epistemic value omega(t).

    Args:
        interaction_count: Number of interaction turns since the relationship started.
        static_profile: The user's static profile (5-layer). If None, completeness = 0.
        omega_0: Initial epistemic value.
        decay_lambda: Temporal decay rate per interaction turn.
        max_attrs_per_layer: Max attributes expected per layer (for completeness calc).

    Returns:
        omega value in [0, omega_0]. Higher = more exploration warranted.
    """
    completeness = 0.0
    if static_profile:
        completeness = compute_profile_completeness(static_profile, max_attrs_per_layer)

    # Temporal component: exponential decay over interactions
    temporal = math.exp(-decay_lambda * max(interaction_count, 0))

    # Completeness component: less room for discovery as profile fills up
    room_for_discovery = 1.0 - completeness

    omega = omega_0 * temporal * room_for_discovery
    return round(max(omega, 0.0), 4)


def get_exploration_label(omega: float) -> str:
    """Map omega to a human-readable exploration label."""
    if omega >= 0.6:
        return "explore"
    elif omega >= 0.25:
        return "balanced"
    else:
        return "exploit"


class EpistemicDecayTracker:
    """Stateful tracker for epistemic value decay across an interaction session.

    Usage:
        tracker = EpistemicDecayTracker()
        # After each turn:
        tracker.increment()
        omega = tracker.compute(static_profile)
    """

    def __init__(
        self,
        omega_0: float = 1.0,
        decay_lambda: float = 0.01,
        max_attrs_per_layer: int = 8,
        initial_count: int = 0,
    ):
        self.omega_0 = omega_0
        self.decay_lambda = decay_lambda
        self.max_attrs_per_layer = max_attrs_per_layer
        self.interaction_count = initial_count

    def increment(self) -> None:
        self.interaction_count += 1

    def compute(self, static_profile: Optional[Dict[str, Any]] = None) -> float:
        return compute_omega(
            interaction_count=self.interaction_count,
            static_profile=static_profile,
            omega_0=self.omega_0,
            decay_lambda=self.decay_lambda,
            max_attrs_per_layer=self.max_attrs_per_layer,
        )

    def label(self, static_profile: Optional[Dict[str, Any]] = None) -> str:
        return get_exploration_label(self.compute(static_profile))
