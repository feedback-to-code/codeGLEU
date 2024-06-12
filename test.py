import os

from codegleu.codegleu import calc_codegleu


def pad(s, ln, pos: str):
    match pos:
        case "r":
            return s + " " * (ln - len(s))
        case "l":
            return " " * (ln - len(s)) + s
        case "m":
            return " " * int((ln - len(s)) / 2) + s + " " * int((ln - len(s)) / 2 + (len(s) % 2 == 1))
        case _:
            return s


SOURCE = "/codegleu/data/testsource.txt"
HYPOTHESES = "/codegleu/data/testhypo1.txt"
REFERENCES = ["/codegleu/data/testref1.txt", "/codegleu/data/testref2.txt"]
NUMGRAM = 4
LANGUAGE = "java"

dir = os.getcwd()

# from codegleu.__main__ import main
# main(dir + SOURCE, [dir + r for r in REFERENCES], dir + HYPOTHESES, "java")

sources = [source for source in open(dir + SOURCE).readlines()]
references = [[ref for ref in open(dir + ref).readlines()] for ref in REFERENCES]
references = list(map(list, zip(*references)))  # from column-wise to row-wise
hypotheses = [hypo for hypo in open(dir + HYPOTHESES).readlines()]
n_weights = (1 / NUMGRAM,) * NUMGRAM

runs: list[dict] = []
runs.append({"title": "total"} | calc_codegleu(sources, references, hypotheses, "java", weights=n_weights))
titles = ["exactmatch", "sensible", "sensible 2", "repeat, wrong syntax", "repeat, wrong ngrams"]
for index, (source, reference, hypothesis) in enumerate(zip(sources, references, hypotheses)):
    title = {"title": titles[index] if index < len(titles) else f"row {index+1}"}
    runs.append(title | calc_codegleu([source], [reference], [hypothesis], "java", weights=n_weights))

print("GLEU+")
cols = list(runs[0].keys())
if "title" in cols:
    cols.remove("title")
    cols.insert(0, "title")
maxlen = max([len(c) for c in cols])
print(" | ".join([pad(x, maxlen, "m") for x in cols]))
for run in runs:
    print(" | ".join([pad(str(run.get(x, None)), maxlen, "r") for x in cols]))
