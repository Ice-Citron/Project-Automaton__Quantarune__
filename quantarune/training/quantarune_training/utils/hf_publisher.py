"""Utilities for publishing training runs to the Hugging Face Hub."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def sanitize_branch_name(value: str) -> str:
    """Convert arbitrary text into a branch-safe identifier."""
    sanitized = re.sub(r"[^A-Za-z0-9._/-]+", "-", value.strip())
    sanitized = re.sub(r"-{2,}", "-", sanitized).strip("-/. ")
    if sanitized:
        return sanitized
    return f"run-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


def resolve_branch_name(cfg, output_dir: Path) -> str:
    """Resolve the destination branch name for this training run."""
    explicit_branch = cfg.huggingface.get("branch")
    if explicit_branch:
        return sanitize_branch_name(str(explicit_branch))

    prefix = str(cfg.huggingface.get("branch_prefix", "run"))
    base = output_dir.name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return sanitize_branch_name(f"{prefix}-{base}")


def repo_web_url(repo_id: str, repo_type: str, revision: str | None = None) -> str:
    """Return the canonical web URL for a Hub repo or branch."""
    repo_type = repo_type or "model"
    if repo_type == "model":
        base = f"https://huggingface.co/{repo_id}"
    elif repo_type == "dataset":
        base = f"https://huggingface.co/datasets/{repo_id}"
    elif repo_type == "space":
        base = f"https://huggingface.co/spaces/{repo_id}"
    else:
        base = f"https://huggingface.co/{repo_id}"

    if revision and revision != "main":
        return f"{base}/tree/{revision}"
    return base


def build_ignore_patterns(cfg) -> list[str]:
    """Build upload ignore patterns based on config."""
    patterns = [
        "**/__pycache__/**",
        "**/*.pyc",
        "wandb/**",
        "**/wandb/**",
        "*.log",
        "**/*.log",
        ".hydra/**",
        "**/.hydra/**",
        "checkpoints/*/training_state/**",
        "**/training_state/**",
    ]
    if not bool(cfg.huggingface.get("upload_checkpoints", True)):
        patterns.extend(
            [
                "checkpoint_*/checkpoint.pt",
                "checkpoint_*/model.safetensors",
                "checkpoint_*/normalization_stats.safetensors",
            ]
        )
    return patterns


def write_run_metadata(
    output_dir: Path,
    cfg,
    branch_name: str,
    wandb_url: str | None = None,
) -> None:
    """Write a model card and resolved Hydra config into the run directory."""
    resolved_cfg_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(cfg, resolved_cfg_path)

    run_title = str(cfg.huggingface.get("run_title", "AIC Training Run"))
    run_summary = str(
        cfg.huggingface.get(
            "run_summary",
            (
                "This published run is currently a quick validation pass on the "
                "`experiment_8` dataset. The goal of this checkpoint is to verify "
                "conversion, training, export, and Gazebo deployment end-to-end "
                "before scaling up."
            ),
        )
    )
    dataset_path = str(
        cfg.get("data", {}).get(
            "dataset_local_path",
            cfg.get("data", {}).get("dataset_root", "unknown"),
        )
    )
    device = str(cfg.get("device", "unknown"))
    eval_enabled = bool(cfg.get("eval", {}).get("enable", False))

    lines = [
        f"# {run_title}",
        "",
        run_summary,
        "",
        f"- branch: `{branch_name}`",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- dataset_local_path: `{dataset_path}`",
        f"- device: `{device}`",
        f"- eval_enabled: `{eval_enabled}`",
        f"- upload_main_enabled: `{bool(cfg.huggingface.upload_main)}`",
    ]

    if wandb_url:
        lines.append(f"- wandb_run: {wandb_url}")

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `resolved_config.yaml` contains the fully resolved Hydra config for this run.",
            "- `checkpoint_*` directories contain model checkpoints and copied normalization stats.",
            "",
            "## Notes",
            "",
            "- This branch stores generated training artifacts that are intentionally ignored in the Git repo.",
            "- The Hugging Face `main` branch is updated with the latest completed run when `huggingface.upload_main=true`.",
            "- W&B graph images are not exported yet; the W&B run link above is the source of truth for charts.",
            "",
        ]
    )

    (output_dir / "README.md").write_text("\n".join(lines))


def create_branch_if_needed(api: HfApi, repo_id: str, branch_name: str, repo_type: str) -> None:
    """Create the destination branch if it does not already exist."""
    try:
        api.create_branch(
            repo_id=repo_id,
            branch=branch_name,
            repo_type=repo_type,
            exist_ok=True,
        )
    except TypeError:
        try:
            api.create_branch(repo_id=repo_id, branch=branch_name, repo_type=repo_type)
        except Exception as exc:  # pragma: no cover - depends on hub state
            message = str(exc).lower()
            if "already exists" not in message and "409" not in message:
                raise
    except Exception as exc:  # pragma: no cover - depends on hub state
        message = str(exc).lower()
        if "already exists" not in message and "409" not in message:
            raise


def publish_run_to_huggingface(output_dir: Path, cfg, wandb_url: str | None = None):
    """Upload the completed training run to a timestamped branch and optionally sync main."""
    repo_id = str(cfg.huggingface.repo_id)
    repo_type = str(cfg.huggingface.get("repo_type", "model"))
    private = bool(cfg.huggingface.get("private", False))
    branch_name = resolve_branch_name(cfg, output_dir)
    ignore_patterns = build_ignore_patterns(cfg)

    write_run_metadata(output_dir, cfg, branch_name, wandb_url=wandb_url)

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    create_branch_if_needed(api, repo_id, branch_name, repo_type)

    api.upload_folder(
        repo_id=repo_id,
        folder_path=output_dir,
        path_in_repo=".",
        repo_type=repo_type,
        revision=branch_name,
        ignore_patterns=ignore_patterns,
        commit_message=f"Upload training run {branch_name}",
    )
    branch_url = repo_web_url(repo_id, repo_type, revision=branch_name)
    logger.info("Uploaded training run to %s", branch_url)

    main_url = None
    if bool(cfg.huggingface.get("upload_main", True)):
        api.upload_folder(
            repo_id=repo_id,
            folder_path=output_dir,
            path_in_repo=".",
            repo_type=repo_type,
            revision="main",
            ignore_patterns=ignore_patterns,
            commit_message=f"Sync latest training run from {branch_name}",
        )
        main_url = repo_web_url(repo_id, repo_type, revision="main")
        logger.info("Synced latest run to %s", main_url)

    return branch_name, branch_url, main_url
