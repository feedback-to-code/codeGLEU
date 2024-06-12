import os
from codegleu import calc_codegleu


SOURCE = "/codegleu/data/testsource.txt"
HYPOTHESES = "/codegleu/data/testhypo1.txt"
REFERENCES = ["/codegleu/data/testref1.txt", "/codegleu/data/testref2.txt"]
NUMGRAM = 4

dir = os.getcwd()
sources = [source for source in open(dir + SOURCE).readlines()]
references = [[ref for ref in open(dir + ref).readlines()] for ref in REFERENCES]
references = list(map(list, zip(*references)))  # from column-wise to row-wise
hypotheses = [hypo for hypo in open(dir + HYPOTHESES).readlines()]
n_weights = [1 / NUMGRAM] * NUMGRAM

runs: list[dict] = []
runs.append({"title": "total"} | calc_codegleu(sources, references, hypotheses, "java", weights=n_weights))
for index, (source, reference, hypothesis) in enumerate(zip(sources, references, hypotheses)):
    runs.append(
        {"title": f"row {index+1}"} | calc_codegleu([source], [reference], [hypothesis], "java", weights=n_weights)
    )

print(f"GLEU+")
cols = list(runs[0].keys())
if "title" in cols:
    cols.remove("title")
    cols.insert(0, "title")
maxlen = max([len(c) for c in cols])
for col in cols:
    print(col + " " * (maxlen - len(col)), end="")
print("")
for run in runs:
    for col in cols:
        val = str(run.get(col, None))
        print(val + " " * (maxlen - len(val)), end="")
    print("")
