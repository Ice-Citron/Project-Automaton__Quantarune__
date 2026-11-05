"""Learned segmentation utilities for connector and port perception."""

from .dataset import TaskConditionedSegmentationDataset
from .inference import PortSegmenterPredictor, PortSegmentationPrediction
from .localization_dataset import TaskConditionedLocalizationDataset
from .localization_losses import localization_loss, spatial_softargmax2d
from .localization_metrics import compute_localization_metrics
from .localization_model import (
    LOCALIZATION_LABEL_NAMES,
    TaskConditionedPortLocalizer,
    load_port_localizer_checkpoint,
    save_port_localizer_checkpoint,
)
from .losses import segmentation_loss
from .metrics import compute_batch_metrics
from .model import (
    TaskConditionedPortSegmenter,
    load_port_segmenter_checkpoint,
    save_port_segmenter_checkpoint,
)
from .schema import (
    SEGMENTATION_LABEL_NAMES,
    TASK_CONDITION_NAMES,
    TASK_CONDITION_SCALES,
    TaskCondition,
    task_condition_array_from_mapping,
)

__all__ = [
    "PortSegmentationPrediction",
    "PortSegmenterPredictor",
    "LOCALIZATION_LABEL_NAMES",
    "SEGMENTATION_LABEL_NAMES",
    "TASK_CONDITION_NAMES",
    "TASK_CONDITION_SCALES",
    "TaskCondition",
    "TaskConditionedLocalizationDataset",
    "TaskConditionedPortLocalizer",
    "TaskConditionedPortSegmenter",
    "compute_localization_metrics",
    "TaskConditionedSegmentationDataset",
    "compute_batch_metrics",
    "load_port_localizer_checkpoint",
    "load_port_segmenter_checkpoint",
    "localization_loss",
    "save_port_segmenter_checkpoint",
    "save_port_localizer_checkpoint",
    "segmentation_loss",
    "spatial_softargmax2d",
    "task_condition_array_from_mapping",
]
