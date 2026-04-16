"""Lightning integration package for AmbiDOA.

Module layout:
- `data_module.py`: Manifest-driven DataModule with legacy-compatible dataset settings.
- `lit_module.py`: LightningModule that preserves original AmbiDOA loss/optimization flow.
- `finetune_utils.py`: Checkpoint loading and selective parameter freezing helpers.
- `train_lightning.py`: End-to-end training CLI entrypoint.
- `config_compatibility_test.py`: Pre-flight config validation and fast-dev smoke test.
"""

__all__ = [
    "data_module",
    "lit_module",
    "finetune_utils",
    "train_lightning",
    "config_compatibility_test",
]
