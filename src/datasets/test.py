import os
from pathlib import Path
print(os.path.join(Path(__file__).resolve().parents[2], "data/raw"))