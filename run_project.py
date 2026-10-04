import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

steps = [
    [sys.executable, "src/train.py"],
    [sys.executable, "src/explain.py"],
]

for command in steps:
    subprocess.run(command, cwd=ROOT, check=True)

print("\nProject training and explainability pipeline completed.")
print("Run: streamlit run app/app.py")
