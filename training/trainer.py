"""
Training loop with:
  - AMP (automatic mixed precision) for faster Kaggle P100 training
  - Cosine LR schedule with linear warmup
  - Differential learning rates (backbone gets 10x smaller LR)
  - Staged unfreezing: head only for first N epochs, then full fine-tune
  - W&B logging of loss, metrics, LR, and gradient norms
  - Best-checkpoint saving based on macro AUC-ROC
  - Early stopping
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from models.classifier import ChestAIClassifier
from training.losses import build_loss
from training.metrics import compute_metrics, metrics_dataframe
from data.dataset import build_datasets


class Trainer:
    """
    Manages the full training lifecycle.

    Usage:
        trainer = Trainer(cfg)
        trainer.fit()
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[Trainer] Device: {self.device}")

        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_loss()
        self._setup_wandb()

        self.checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = GradScaler(enabled=cfg["training"]["amp"])
        self.best_auc = 0.0
        self.epochs_no_improve = 0

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_data(self) -> None:
        datasets = build_datasets(self.cfg)
        num_workers = self.cfg["data"]["num_workers"]
        pin_memory  = self.cfg["data"]["pin_memory"]
        bs  = self.cfg["training"]["batch_size"]
        vbs = self.cfg["training"]["val_batch_size"]

        self.train_loader = DataLoader(
            datasets["train"],
            batch_size=bs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            datasets["val"],
            batch_size=vbs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.train_pos_weights = datasets["train"].get_pos_weights().to(self.device)
        print(
            f"[Trainer] Train: {len(datasets['train'])} | "
            f"Val: {len(datasets['val'])} samples"
        )

    def _setup_model(self) -> None:
        self.model = ChestAIClassifier(
            backbone_name=self.cfg["model"]["backbone"],
            num_classes=self.cfg["model"]["num_classes"],
            dropout_rate=self.cfg["model"]["dropout_rate"],
            freeze_backbone=True,   # start frozen; unfreeze after warm-up
        ).to(self.device)
        print(f"[Trainer] {self.model.summary()}")

    def _setup_optimizer(self) -> None:
        lr = self.cfg["training"]["learning_rate"]
        bb_lr = lr * self.cfg["training"]["backbone_lr_multiplier"]
        wd = self.cfg["training"]["weight_decay"]

        # Separate param groups for differential LRs.
        backbone_params = list(self.model.backbone.parameters())
        head_params     = list(self.model.head.parameters())

        self.optimizer = AdamW([
            {"params": backbone_params, "lr": bb_lr},
            {"params": head_params,     "lr": lr},
        ], weight_decay=wd)

        epochs  = self.cfg["training"]["epochs"]
        warmup  = self.cfg["training"]["warmup_epochs"]

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs - warmup,
            eta_min=1e-7,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup],
        )

    def _setup_loss(self) -> None:
        self.criterion = build_loss(self.cfg, self.train_pos_weights)

    def _setup_wandb(self) -> None:
        if not WANDB_AVAILABLE:
            return
        proj_cfg = self.cfg["project"]
        wandb.init(
            project=proj_cfg["wandb_project"],
            entity=proj_cfg.get("entity") or None,
            config=self.cfg,
            name=f"chestai-{time.strftime('%m%d-%H%M')}",
        )
        wandb.watch(self.model, log="gradients", log_freq=100)

    # ------------------------------------------------------------------
    # Train / val epochs
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(self.train_loader):
            images, labels = batch[0].to(self.device), batch[1].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.cfg["training"]["amp"]):
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            # Gradient clipping before optimizer step.
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg["training"]["grad_clip"],
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            if step % 50 == 0:
                lr = self.optimizer.param_groups[-1]["lr"]
                print(
                    f"  Epoch {epoch} | Step {step}/{len(self.train_loader)} "
                    f"| loss={loss.item():.4f} | lr={lr:.2e}"
                )

        return {"train/loss": total_loss / len(self.train_loader)}

    @torch.no_grad()
    def _val_epoch(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_targets, all_probs = [], []

        for batch in self.val_loader:
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            with autocast(enabled=self.cfg["training"]["amp"]):
                logits = self.model(images)
                loss   = self.criterion(logits, labels)
            total_loss += loss.item()
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(labels.cpu().numpy())

        targets = np.concatenate(all_targets)
        probs   = np.concatenate(all_probs)
        metrics = compute_metrics(targets, probs, prefix="val/")
        metrics["val/loss"] = total_loss / len(self.val_loader)
        return metrics

    # ------------------------------------------------------------------
    # Main fit loop
    # ------------------------------------------------------------------

    def fit(self) -> None:
        freeze_epochs = self.cfg["model"]["freeze_backbone_epochs"]
        epochs        = self.cfg["training"]["epochs"]
        patience      = self.cfg["training"]["early_stopping_patience"]

        for epoch in range(1, epochs + 1):
            # Staged unfreezing: unfreeze backbone after warm-up phase.
            if epoch == freeze_epochs + 1:
                print(f"[Trainer] Epoch {epoch}: Unfreezing backbone (last 6 blocks)")
                self.model.unfreeze_backbone(last_n_blocks=6)

            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._val_epoch()
            self.scheduler.step()

            elapsed = time.time() - t0
            macro_auc = val_metrics["val/macro/auc"]
            print(
                f"\nEpoch {epoch}/{epochs} | "
                f"train_loss={train_metrics['train/loss']:.4f} | "
                f"val_auc={macro_auc:.4f} | "
                f"time={elapsed:.1f}s"
            )
            print(metrics_dataframe(val_metrics).to_string(index=False))

            all_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            if WANDB_AVAILABLE:
                wandb.log(all_metrics)

            # Checkpoint
            if macro_auc > self.best_auc:
                self.best_auc = macro_auc
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, macro_auc, is_best=True)
                print(f"  ✓ New best AUC: {macro_auc:.4f} — checkpoint saved")
            else:
                self.epochs_no_improve += 1
                self._save_checkpoint(epoch, macro_auc, is_best=False)

            if self.epochs_no_improve >= patience:
                print(f"[Trainer] Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        if WANDB_AVAILABLE:
            wandb.finish()
        print(f"\n[Trainer] Training complete. Best val AUC: {self.best_auc:.4f}")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, auc: float, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_auc": self.best_auc,
            "config": self.cfg,
        }
        path = self.checkpoint_dir / f"epoch_{epoch:03d}_auc{auc:.4f}.pt"
        torch.save(state, path)
        if is_best:
            torch.save(state, self.checkpoint_dir / "best_model.pt")
