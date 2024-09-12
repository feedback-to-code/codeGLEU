import os
from pathlib import Path
file = Path(os.path.realpath(__file__))
root = file.parent
modelfolder = root.joinpath(Path("data/models/lite"))

done = os.listdir(str(root.joinpath(Path("results/lite"))))
for model in os.listdir(str(modelfolder)):
    if model + ".txt" in done:
        continue
    os.system(f"python evaluate.py --model lite/{model}")