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

Exploration modes (for Experiment 3 ablation):
    adaptive  — omega(t) decays naturally (our method)
    no_exploration    — omega(t) = 0 always (pure exploit)
    fixed_exploration — omega(t) = fixed_value always
    always_exploration — omega(t) = 1.0 always (pure explore)
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional


# Five layers of the hierarchical user profile
PROFILE_LAYERS = ("core", "regulation", "cognition", "identity", "behavior")

# Valid exploration modes
EXPLORATION_MODES = ("adaptive", "no_exploration", "fixed_exploration", "always_exploration")


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


def compute_portrait_entropy(static_profile: Dict[str, Any]) -> float:
    """Compute the posterior entropy of the user profile.

    Uses the confidence values from Bayesian updating to measure uncertainty.
    For each attribute with confidence c:
      H_i = -c * log2(c) - (1-c) * log2(1-c)  (binary entropy)
    The total entropy is the average across all attributes.

    Attributes without confidence values are treated as c=0.5 (maximum uncertainty).
    Empty profile returns maximum entropy (1.0).

    Returns:
        Portrait entropy in [0, 1]. Lower = more certain profile.
    """
    if not static_profile:
        return 1.0

    entropies = []
    for layer in PROFILE_LAYERS:
        section = static_profile.get(layer, {})
        if not isinstance(section, dict):
            continue
        for key, val in section.items():
            if isinstance(val, dict):
                c = val.get("confidence", 0.5)
                if not isinstance(c, (int, float)):
                    c = 0.5
            else:
                c = 0.5  # no confidence info = max uncertainty

            # Clamp to avoid log(0)
            c = max(0.001, min(0.999, float(c)))
            h = -c * math.log2(c) - (1 - c) * math.log2(1 - c)
            entropies.append(h)

    if not entropies:
        return 1.0

    return round(sum(entropies) / len(entropies), 4)


class EpistemicDecayTracker:
    """Stateful tracker for epistemic value decay across an interaction session.

    Supports multiple exploration modes for ablation experiments:
        adaptive          — omega(t) decays naturally (our method)
        no_exploration    — omega(t) = 0 always
        fixed_exploration — omega(t) = fixed_value always
        always_exploration — omega(t) = 1.0 always

    Usage:
        tracker = EpistemicDecayTracker(mode="adaptive")
        tracker.increment()
        omega = tracker.compute(static_profile)
    """

    def __init__(
        self,
        omega_0: float = 1.0,
        decay_lambda: float = 0.01,
        max_attrs_per_layer: int = 8,
        initial_count: int = 0,
        mode: str = "adaptive",
        fixed_value: float = 0.5,
    ):
        if mode not in EXPLORATION_MODES:
            raise ValueError(f"Unknown exploration mode: {mode}. Must be one of {EXPLORATION_MODES}")
        self.omega_0 = omega_0
        self.decay_lambda = decay_lambda
        self.max_attrs_per_layer = max_attrs_per_layer
        self.interaction_count = initial_count
        self.mode = mode
        self.fixed_value = fixed_value

    def increment(self) -> None:
        self.interaction_count += 1

    def compute(self, static_profile: Optional[Dict[str, Any]] = None) -> float:
        """Compute omega(t) based on the current exploration mode."""
        if self.mode == "no_exploration":
            return 0.0
        elif self.mode == "always_exploration":
            return 1.0
        elif self.mode == "fixed_exploration":
            return self.fixed_value
        else:  # adaptive
            return compute_omega(
                interaction_count=self.interaction_count,
                static_profile=static_profile,
                omega_0=self.omega_0,
                decay_lambda=self.decay_lambda,
                max_attrs_per_layer=self.max_attrs_per_layer,
            )

    def label(self, static_profile: Optional[Dict[str, Any]] = None) -> str:
        return get_exploration_label(self.compute(static_profile))
