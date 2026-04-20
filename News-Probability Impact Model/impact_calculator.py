"""Simple helpers for adjusting prediction probabilities with an impact score."""


def adjust_probability(base_prob: float, impact: float) -> float:
    """Adjust a probability using a simple impact score and clamp the result.

    The idea is:
    - if the probability is already above 0.5, impact pushes it higher
    - if the probability is below 0.5, impact pushes it lower
    - if the probability is exactly 0.5, it stays unchanged

    Examples:
        adjust_probability(0.6, 0.8) -> 0.68
        adjust_probability(0.4, 0.8) -> 0.32
        adjust_probability(0.5, 0.8) -> 0.5
    """
    if not isinstance(base_prob, (int, float)):
        raise ValueError("base_prob must be a number.")

    if not isinstance(impact, (int, float)):
        raise ValueError("impact must be a number.")

    if not 0 <= base_prob <= 1:
        raise ValueError("base_prob must be between 0 and 1.")

    # Move the probability farther from 0.5 based on impact strength.
    adjusted = base_prob + (base_prob - 0.5) * impact

    # Clamp the final value so it always stays in the valid probability range.
    return max(0.0, min(1.0, round(adjusted, 4)))


__all__ = ["adjust_probability"]
