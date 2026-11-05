from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from quantarune_training.lerobot_video_patch import patch_lerobot_video_backend


def patch_dataloader_defaults() -> None:
    """Raise dataloader throughput without editing site-packages."""
    dataloader_cls = torch.utils.data.DataLoader
    if getattr(dataloader_cls, "_aic_patched_init", False):
        return

    original_init = dataloader_cls.__init__

    def patched_init(self, *args, **kwargs):
        num_workers = int(kwargs.get("num_workers", 0) or 0)
        if num_workers > 0:
            kwargs.setdefault("persistent_workers", True)
            kwargs["prefetch_factor"] = int(
                os.environ.get("AIC_TRAIN_PREFETCH_FACTOR", "1")
            )
        return original_init(self, *args, **kwargs)

    dataloader_cls.__init__ = patched_init
    dataloader_cls._aic_patched_init = True


def main() -> Any:
    patch_lerobot_video_backend()
    patch_dataloader_defaults()
    print(
        "AIC ACT dataloader patch:"
        f" workers_env={os.environ.get('AIC_ACT_TRAIN_NUM_WORKERS', '<unset>')}"
        f" prefetch_env={os.environ.get('AIC_TRAIN_PREFETCH_FACTOR', '<unset>')}",
        flush=True,
    )
    train_module = importlib.import_module("lerobot.scripts.lerobot_train")
    return train_module.main()


if __name__ == "__main__":
    main()
