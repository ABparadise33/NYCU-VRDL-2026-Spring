from __future__ import annotations

import os


os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

from tools.inference_utils import main


if __name__ == "__main__":
    main()
