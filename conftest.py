import sys
from pathlib import Path

# Add the src directory to sys.path so packages inside it can be imported
SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))