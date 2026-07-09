"""Lightning DataModule for AmbiDOA with legacy-compatible dataset contract.

The DataModule is intentionally thin:
- It loads dataset settings from the same JSON format used previously.
- It keeps manifest-based dataset construction unchanged.
- It preserves split-by-file validation logic to avoid data leakage between
  train/validation windows from the same recording.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:
    # Backward-compatible import path for older installs.
    import pytorch_lightning as pl

# Make project imports stable regardless of launch location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import AmbiDOADataset
from train_utils import split_by_file
from utils_device import dataloader_kwargs_for


def _flatten_config(config):
    """Flatten one level of grouped config sections into a flat dict."""
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


class AmbiDOADataModule(pl.LightningDataModule):
    """Prepare AmbiDOA train/validation datasets and dataloaders.

    Either `config_path` or in-memory `config` can be passed. This lets the
    same module be reused by:
    - regular training scripts
    - compatibility tests
    - debug/fast-dev runs with temporary config overrides
    """

    def __init__(
        self,
        config_path=None,
        config=None,
        run_txt_log_name=None,
        run_txt_timestamp=True,
    ):
        super().__init__()
        # Load JSON config only when caller did not provide an in-memory dict.
        if config is None:
            if config_path is None:
                config_path = Path(__file__).resolve().parents[1] / "config.json"
            with open(config_path, encoding="utf-8-sig") as f:
                config = _flatten_config(json.load(f))

        self.config = dict(config)
        self.config_path = Path(config_path) if config_path is not None else None

        # Paths and split/runtime controls.
        self.manifest_path = self.config["manifest_path"]
        self.audio_dir = Path(self.config["audio_dir"])
        self.batch_size = int(self.config.get("batch_size", 64))
        self.num_workers = int(self.config.get("num_workers", 8))
        self.train_num_workers = int(self.config.get("train_num_workers", self.num_workers))
        # On Windows, validation worker spawn can OOM on large datasets; keep safe default.
        self.val_num_workers = int(
            self.config.get(
                "val_num_workers",
                0 if os.name == "nt" else self.train_num_workers,
            )
        )
        self.val_ratio = float(self.config.get("val_ratio", 0.2))
        self.split_seed = int(self.config.get("split_seed", 42))

        # Keep the current dataset contract and allow optional overrides in config.json.
        # This preserves compatibility with existing config and manifest generation.
        # `subsample_pct` is a debug feature that drops a random fraction of
        # manifest entries before window enumeration — the run is otherwise
        # identical to a regular train/validation run (logging, checkpoints,
        # callbacks all active).
        subsample_pct = self.config.get("subsample_pct")
        if subsample_pct is not None:
            subsample_pct = float(subsample_pct)
        self.subsample_pct = subsample_pct
        self.dataset_kwargs = dict(
            sr=int(self.config.get("sr", 11025)),
            window_len_ms=int(self.config.get("window_len_ms", 250)),
            dbfs_threshold=float(self.config.get("dbfs_threshold", -80.0)),
            max_sources=int(self.config.get("max_sources", 12)),
            subsample_pct=subsample_pct,
            subsample_seed=int(self.config.get("split_seed", 42)),
        )

        self.train_set = None
        self.val_set = None
        self.run_txt_log_name = run_txt_log_name
        self.run_txt_timestamp = bool(run_txt_timestamp)
        self._run_txt_log_path = None

    def _resolve_run_txt_log_path(self):
        if self._run_txt_log_path is not None:
            return self._run_txt_log_path

        logger = self.trainer.logger if self.trainer is not None else None
        if logger is None:
            return None
        save_dir = getattr(logger, "save_dir", None)
        if not save_dir:
            return None

        run_logs_dir = Path(save_dir)
        run_logs_dir.mkdir(parents=True, exist_ok=True)
        if self.run_txt_log_name:
            log_name = self.run_txt_log_name
        else:
            log_name = f"log_AmbiDOA_{datetime.now().strftime('%Y-%m-%d')}.txt"
            self.run_txt_log_name = log_name
        self._run_txt_log_path = run_logs_dir / log_name
        return self._run_txt_log_path

    def _append_run_txt_log(self, text):
        prefix = ""
        if self.run_txt_timestamp:
            prefix = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        line = f"{prefix}{text}"
        print(line)
        log_path = self._resolve_run_txt_log_path()
        if log_path is None:
            return
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    @staticmethod
    def _count_unique_files(subset):
        if not hasattr(subset, "indices") or not hasattr(subset, "dataset"):
            return None
        base_ds = subset.dataset
        if not hasattr(base_ds, "windows"):
            return None
        try:
            return len({base_ds.windows[int(i)]["audio_path"] for i in subset.indices})
        except Exception:
            return None

    def setup(self, stage=None):
        """Construct datasets once and reuse them across Lightning calls."""
        # Lightning may call setup multiple times (fit/validate/test stages).
        # Guard to avoid repeated expensive manifest/audio initialization.
        if self.train_set is not None and self.val_set is not None:
            return

        # Build full dataset from manifest + audio directory.
        dataset = AmbiDOADataset(self.manifest_path, self.audio_dir, **self.dataset_kwargs)

        # Mark the run as subsampled in the run TXT log so checkpoints from
        # this run aren't mistaken for production-quality artifacts.
        if self.subsample_pct is not None and 0 < self.subsample_pct < 100:
            kept = len(dataset.entries)
            total = getattr(dataset, "_full_manifest_size", kept)
            self._append_run_txt_log(
                f"[SUBSAMPLED] Run uses {self.subsample_pct:.2f}% of manifest "
                f"({kept}/{total} files, seed={self.dataset_kwargs['subsample_seed']})"
            )

        # Split by source file identity to avoid leakage across windows.
        self.train_set, self.val_set = split_by_file(
            dataset,
            val_ratio=self.val_ratio,
            seed=self.split_seed,
            # Lightning writes run-scoped TXT logs itself; avoid legacy global log file here.
            log_fn=None,
        )

        train_frames = len(self.train_set)
        val_frames = len(self.val_set)
        train_files = self._count_unique_files(self.train_set)
        val_files = self._count_unique_files(self.val_set)
        if train_files is not None and val_files is not None:
            line = (
                f"Liczba ramek (train/val): {train_frames} / {val_frames}  | "
                f"pliki: {train_files} / {val_files}"
            )
        else:
            line = f"Liczba ramek (train/val): {train_frames} / {val_frames}"
        self._append_run_txt_log(line)
        self._append_run_txt_log(
            f"DataLoader workers (train/val): {self.train_num_workers} / {self.val_num_workers}"
        )

    def _loader_kwargs(self, split):
        """Select safe DataLoader runtime options for active compute device."""
        # Prefer Lightning-resolved root device when Trainer is attached.
        device = (
            self.trainer.strategy.root_device
            if self.trainer is not None and getattr(self.trainer, "strategy", None) is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        workers = self.train_num_workers if split == "train" else self.val_num_workers
        # Centralized helper manages pin_memory/persistent_workers heuristics.
        return dataloader_kwargs_for(device, num_workers=workers)

    def train_dataloader(self):
        """Training DataLoader with shuffle enabled."""
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            **self._loader_kwargs(split="train"),
        )

    def val_dataloader(self):
        """Validation DataLoader with deterministic order (no shuffle)."""
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            **self._loader_kwargs(split="val"),
        )
