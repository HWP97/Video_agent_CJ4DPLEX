from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ours/b2"))
from latest_inference import main
if __name__ == '__main__':
    main('B2-C01')
