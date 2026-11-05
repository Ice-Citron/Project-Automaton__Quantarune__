"""Main training loop for flow matching policy."""

import copy
import logging
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from quantarune_training.data.dataset import FlowMatchingDataset
from quantarune_training.models.action_head import build_model
from quantarune_training.training.losses import compute_action_mse, flow_matching_loss
from quantarune_training.utils.hf_publisher import publish_run_to_huggingface

logger = logging.getLogger(__name__)


def resolve_device(requested_device: str) -> torch.device:
    """Resolve and validate the requested training device."""
    device = torch.device(requested_device)
    if device.type != 'cuda':
        return device

    if not torch.cuda.is_available():
        raise RuntimeError(
            'Training requested device=cuda, but torch.cuda.is_available() is false.'
        )

    capability = torch.cuda.get_device_capability()
    arch = f'sm_{capability[0]}{capability[1]}'
    supported_arches = set(torch.cuda.get_arch_list())
    if arch not in supported_arches:
        raise RuntimeError(
            'This PyTorch build does not support the installed GPU architecture. '
            f'Detected GPU arch: {arch}. '
            f'Supported arches in current wheel: {sorted(supported_arches)}. '
            f'torch={torch.__version__}, torch.cuda={torch.version.cuda}. '
            'Use `device=cpu` for smoke tests, or install a newer PyTorch build '
            'that supports this GPU.'
        )

    return device


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    def update(self, model: nn.Module):
        with torch.no_grad():
            for shadow_param, model_param in zip(
                self.shadow.parameters(), model.parameters()
            ):
                shadow_param.data.mul_(self.decay).add_(
                    model_param.data,
                    alpha=1.0 - self.decay,
                )

    def state_dict(self):
        return self.shadow.state_dict()


def create_scheduler(optimizer, cfg):
    """Create learning rate scheduler with warmup."""
    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=cfg.warmup_steps,
    )
    decay = CosineAnnealingLR(
        optimizer,
        T_max=cfg.max_steps - cfg.warmup_steps,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, decay],
        milestones=[cfg.warmup_steps],
    )


def log_checkpoint_artifact(wandb_logger, ckpt_dir: Path, step: int) -> None:
    """Upload a checkpoint directory as a W&B artifact."""
    if wandb_logger is None:
        return

    try:
        import wandb

        artifact_name = f'{wandb_logger.id}-checkpoint-{step}'
        artifact = wandb.Artifact(artifact_name, type='model')
        artifact.add_dir(str(ckpt_dir))
        wandb_logger.log_artifact(artifact, aliases=[f'step-{step}', 'latest'])
    except Exception as exc:  # pragma: no cover - defensive logging path
        logger.warning('Checkpoint artifact upload failed at step %s: %s', step, exc)


def train(cfg, wandb_logger=None):
    """Main training function."""
    device = resolve_device(cfg.device)

    dataset = FlowMatchingDataset(
        data_dir=cfg.data.dataset_local_path,
        normalize=True,
        image_augmentation=cfg.data.image_augmentation,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )
    data_iter = iter(dataloader)

    model = build_model(cfg.model).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Model parameters (trainable): %s', f'{param_count:,}')

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = create_scheduler(optimizer, cfg.training)
    ema = EMA(model, decay=cfg.training.ema_decay)

    scaler = None
    if cfg.training.mixed_precision and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if wandb_logger is not None:
        wandb_logger.summary['output_dir'] = str(output_dir)
        wandb_logger.summary['device'] = str(device)
        wandb_logger.summary['trainable_parameters'] = param_count

    logger.info('Starting training for %s steps', cfg.training.max_steps)

    for step in range(1, cfg.training.max_steps + 1):
        model.train()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                losses = flow_matching_loss(model, batch, device)
            total_loss = losses['flow_loss']
            if 'stiffness_loss' in losses:
                total_loss = (
                    total_loss
                    + cfg.training.stiffness_loss_weight * losses['stiffness_loss']
                )
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.training.gradient_clip,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            losses = flow_matching_loss(model, batch, device)
            total_loss = losses['flow_loss']
            if 'stiffness_loss' in losses:
                total_loss = (
                    total_loss
                    + cfg.training.stiffness_loss_weight * losses['stiffness_loss']
                )
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.training.gradient_clip,
            )
            optimizer.step()

        scheduler.step()
        ema.update(model)

        if step % cfg.training.log_every == 0:
            log_dict = {
                'train/flow_loss': losses['flow_loss'].item(),
                'train/total_loss': total_loss.item(),
                'train/lr': scheduler.get_last_lr()[0],
                'train/grad_norm': (
                    grad_norm.item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                ),
                'train/step': step,
            }
            per_dim = losses['per_dim_mse']
            for index, value in enumerate(per_dim):
                log_dict[f'train/per_dim_mse_{index}'] = value.item()
            if 'stiffness_loss' in losses:
                log_dict['train/stiffness_loss'] = losses['stiffness_loss'].item()

            logger.info(
                'Step %s/%s | flow_loss: %.6f | lr: %.2e | grad_norm: %.4f',
                step,
                cfg.training.max_steps,
                losses['flow_loss'].item(),
                scheduler.get_last_lr()[0],
                log_dict['train/grad_norm'],
            )
            if wandb_logger is not None:
                wandb_logger.log(log_dict, step=step)

        if step % (cfg.training.log_every * 10) == 0:
            model.eval()
            action_mse = compute_action_mse(
                model,
                batch,
                device,
                num_ode_steps=cfg.model.inference.num_ode_steps,
            )
            logger.info('Step %s | action_mse: %.6f', step, action_mse.item())
            if wandb_logger is not None:
                wandb_logger.log({'eval/action_mse': action_mse.item()}, step=step)

        if step % cfg.training.save_every == 0:
            ckpt_dir = save_checkpoint(model, ema, optimizer, scheduler, step, output_dir, cfg)
            if wandb_logger is not None and cfg.wandb.log_checkpoints:
                log_checkpoint_artifact(wandb_logger, ckpt_dir, step)

        if cfg.eval.enable and step % cfg.training.eval_every == 0:
            ckpt_dir = output_dir / f'checkpoint_{step}'
            if not ckpt_dir.exists():
                ckpt_dir = save_checkpoint(
                    model,
                    ema,
                    optimizer,
                    scheduler,
                    step,
                    output_dir,
                    cfg,
                )
            _run_sim_eval(ckpt_dir, cfg, step, wandb_logger)

    final_ckpt_dir = output_dir / f'checkpoint_{cfg.training.max_steps}'
    if not final_ckpt_dir.exists():
        final_ckpt_dir = save_checkpoint(
            model,
            ema,
            optimizer,
            scheduler,
            cfg.training.max_steps,
            output_dir,
            cfg,
        )
    if wandb_logger is not None and cfg.wandb.log_checkpoints:
        log_checkpoint_artifact(wandb_logger, final_ckpt_dir, cfg.training.max_steps)

    if cfg.eval.enable:
        _run_sim_eval(final_ckpt_dir, cfg, cfg.training.max_steps, wandb_logger)
    else:
        logger.info('Skipping sim evaluation because eval.enable=false')

    if cfg.huggingface.enable:
        wandb_url = getattr(wandb_logger, 'url', None) if wandb_logger is not None else None
        try:
            branch_name, branch_url, main_url = publish_run_to_huggingface(
                output_dir=output_dir,
                cfg=cfg,
                wandb_url=wandb_url,
            )
            logger.info('Hugging Face branch upload complete: %s', branch_url)
            if main_url:
                logger.info('Latest run synced to main: %s', main_url)
            if wandb_logger is not None:
                wandb_logger.summary['huggingface_branch'] = branch_name
                wandb_logger.summary['huggingface_branch_url'] = branch_url
                if main_url:
                    wandb_logger.summary['huggingface_main_url'] = main_url
        except Exception as exc:
            logger.warning('Hugging Face publish failed (non-fatal): %s', exc)

    logger.info('Training complete')


def _run_sim_eval(ckpt_dir, cfg, step, wandb_logger=None):
    """Run sim evaluation, handling failures gracefully."""
    try:
        from quantarune_training.eval.sim_eval import run_sim_eval

        logger.info('Step %s | Starting sim evaluation...', step)
        eval_metrics = run_sim_eval(
            checkpoint_dir=str(ckpt_dir),
            policy_module=cfg.eval.policy_module,
            engine_config=cfg.eval.get('engine_config'),
            results_dir=cfg.eval.get('results_dir'),
            timeout=cfg.eval.timeout_per_trial * cfg.eval.num_trials,
        )

        if eval_metrics:
            logger.info('Step %s | Sim eval results:', step)
            for key, value in sorted(eval_metrics.items()):
                logger.info('  %s: %s', key, value)
            if wandb_logger is not None:
                wandb_logger.log(
                    {f'sim_eval/{key}': value for key, value in eval_metrics.items()},
                    step=step,
                )
        else:
            logger.warning('Step %s | Sim eval returned no metrics', step)

    except Exception as exc:
        logger.warning('Step %s | Sim eval failed (non-fatal): %s', step, exc)
        logger.warning(
            'Sim eval requires the AIC simulation stack (Gazebo + aic_engine). Continuing training.'
        )


def save_checkpoint(model, ema, optimizer, scheduler, step, output_dir, cfg):
    """Save training checkpoint and return its directory."""
    ckpt_dir = output_dir / f'checkpoint_{step}'
    ckpt_dir.mkdir(exist_ok=True)

    torch.save(
        {
            'step': step,
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        },
        ckpt_dir / 'checkpoint.pt',
    )

    from safetensors.torch import save_file

    save_file(ema.state_dict(), str(ckpt_dir / 'model.safetensors'))

    from omegaconf import OmegaConf

    OmegaConf.save(cfg, ckpt_dir / 'config.yaml')

    data_dir = Path(cfg.data.dataset_local_path)
    norm_stats_src = data_dir / 'normalization_stats.safetensors'
    if norm_stats_src.exists():
        shutil.copy2(norm_stats_src, ckpt_dir / 'normalization_stats.safetensors')

    logger.info('Checkpoint saved: %s', ckpt_dir)
    return ckpt_dir
