"""Evaluation metrics parsing from aic_scoring output."""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def parse_scoring_results(results_dir: str) -> dict[str, float]:
    """Parse scoring.yaml from aic_engine evaluation.

    Args:
        results_dir: Path to AIC results directory (e.g. ~/aic_results).

    Returns:
        Dict of metric name -> value, aggregated across trials.
    """
    results_path = Path(results_dir)
    scoring_path = results_path / "scoring.yaml"

    if not scoring_path.exists():
        logger.warning(f"scoring.yaml not found at {scoring_path}")
        return {}

    with open(scoring_path) as f:
        scoring = yaml.safe_load(f)

    if scoring is None:
        logger.warning("Empty scoring.yaml")
        return {}

    metrics = {}

    # Parse per-trial scores
    trials = scoring.get("trials", {})
    for trial_id, trial_data in trials.items():
        prefix = f"trial_{trial_id}"

        # Tier 1
        tier1 = trial_data.get("tier_1", {})
        metrics[f"{prefix}/tier1_score"] = tier1.get("score", 0.0)

        # Tier 2
        tier2 = trial_data.get("tier_2", {})
        metrics[f"{prefix}/tier2_score"] = tier2.get("score", 0.0)
        metrics[f"{prefix}/tier2_smoothness"] = tier2.get("smoothness", 0.0)
        metrics[f"{prefix}/tier2_duration"] = tier2.get("duration", 0.0)
        metrics[f"{prefix}/tier2_efficiency"] = tier2.get("efficiency", 0.0)
        metrics[f"{prefix}/tier2_force_penalty"] = tier2.get("force_penalty", 0.0)
        metrics[f"{prefix}/tier2_contact_penalty"] = tier2.get("contact_penalty", 0.0)

        # Tier 3
        tier3 = trial_data.get("tier_3", {})
        metrics[f"{prefix}/tier3_score"] = tier3.get("score", 0.0)

        # Total
        metrics[f"{prefix}/total"] = trial_data.get("total", 0.0)

    # Aggregate scores
    if trials:
        num_trials = len(trials)
        metrics["eval/total_score"] = sum(
            t.get("total", 0.0) for t in trials.values()
        )
        metrics["eval/avg_score"] = metrics["eval/total_score"] / num_trials
        metrics["eval/num_trials"] = num_trials

    return metrics
