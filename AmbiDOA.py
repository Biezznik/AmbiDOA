import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "Logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
log_filename = LOGS_DIR / f"log_AmbiDOA_{datetime.now().strftime('%Y-%m-%d')}.txt"


def log_print(*args, sep=' ', end='\n', file=None):
    """Print and append to a daily log file (shared logging helper).

    Used across the pipeline (dataset, model, train_utils). The Lightning run owns
    its own run-scoped logging; set AMBIDOA_DISABLE_LEGACY_LOG=1 to suppress the
    legacy daily file in AmbiDOA/Logs.
    """
    text = sep.join(str(a) for a in args) + end
    print(text, end='')
    if os.environ.get("AMBIDOA_DISABLE_LEGACY_LOG", "0") == "1":
        return
    try:
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(text)
    except PermissionError as e:
        print(f"[WARN] Legacy log write skipped ({e}).", end='\n')
