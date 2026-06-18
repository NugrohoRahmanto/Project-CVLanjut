from __future__ import annotations

import sys

from yolo_bdd10k_pipeline import main


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
