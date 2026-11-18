"""Central paths/config for PathSnap. Override the data location with env var PATHSNAP_DATA."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", Path.home() / "Downloads" / "Student folder"))
TRAIN_DIR = DATA_DIR / "train"
STUDENT_TEST_DIR = DATA_DIR / "student_test"
# Evaluator trips are only placed on Laptop 1 at demo time; never used in development.
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DATA", DATA_DIR / "evaluator_test"))
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
SEED = 42
