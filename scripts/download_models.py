"""Download the CorridorKey 2.0 model files without starting ComfyUI.

    python scripts/download_models.py <path to ComfyUI>/models/corridorkey2

Skips files that are already present and resumes partial downloads.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ck_comfy.download import main  # noqa: E402

if __name__ == "__main__":
    main()
