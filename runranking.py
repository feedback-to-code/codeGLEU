import os
import json
from pathlib import Path
file = Path(os.path.realpath(__file__))
root = file.parent
modelfolder = root.joinpath(Path("data/models"))

prefix = "lite"
if prefix:
    modelfolder = modelfolder.joinpath(prefix)

done = []
models = []
with open(str(root.joinpath(Path("results/rankingscores.txt"))), "r+") as scores:
    for line in scores:
        done.append(json.loads(line)["model"])
print("already completed " + str(done))
for model in os.listdir(str(modelfolder)):
    models.append(prefix + "/" + model)

todo = list(set(models) - set(done))
for model in todo:
    os.system("python evaluate.py --model " + model)