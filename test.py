import os
from codegleu import calc_codegleu


SOURCE = "/codegleu/data/testsource.txt"
HYPOTHESIS = "/codegleu/data/testhypo1.txt"
REFERENCES = ["/codegleu/data/testref1.txt", "/codegleu/data/testref2.txt"]
NUMGRAM = 4

dir = os.getcwd()
sources = [source for source in open(dir + SOURCE).readlines()]
references = [[ref for ref in open(dir + ref).readlines()] for ref in REFERENCES]
references = list(map(list, zip(*references))) # from column-wise to row-wise
hypothesis = [hypo for hypo in open(dir + HYPOTHESIS).readlines()]
n_weights = [1/NUMGRAM] * NUMGRAM

gl = calc_codegleu(sources, references, hypothesis, "java", weights = n_weights)
print(f"GLEU+: {gl}")