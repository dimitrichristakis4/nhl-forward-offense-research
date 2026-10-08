from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NST = ROOT / "data" / "nst"
PRIVATE = ROOT / "data" / "private"
WORK = ROOT / "data" / "work"
RESULTS = ROOT / "results"

SEASONS = ["21-22", "22-23", "23-24", "24-25", "25-26"]

# git doesn't keep empty folders, so make the output folders if they're missing
WORK.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
