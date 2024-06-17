import json
import os
import regex as re
import tqdm
from pathlib import Path
from git_utils import GitRepo
from scipy.stats import pearsonr

import codebleu
import codegleu.codegleu

instances_dir = "./data/instances"
preds_loc = "./data/sweagent_preds.jsonl"
results_loc = "./data/results.json"
preprocessed_loc = "./data/preprocessed_instances.jsonl"
scored_loc = "./data/scored_instances.jsonl"
experiments_dir = "./experiments"

hypotheses = {}
with open(preds_loc) as fp:
    for line in fp.readlines():
        prediction = json.loads(line)
        hypotheses[prediction["instance_id"]] = prediction

pred_iids = set()
for pred in hypotheses:
    pred_iids.add(pred)

inst_iids = set()
instances_by_repo = {}
for file in os.listdir(instances_dir):
    with open(f"{instances_dir}/{file}") as fp:
        for line in fp:
            instance = json.loads(line)
            if instance["repo"] not in instances_by_repo:
                instances_by_repo[instance["repo"]] = []
            instances_by_repo[instance["repo"]] += [instance]
            inst_iids.add(instance["instance_id"])

missing = list(inst_iids ^ pred_iids)
# data for all pred_iids should be in data folder
assert missing == []

print("Collecting files and applying patches")
known_iids = []
if not os.path.exists(preprocessed_loc):
    open(preprocessed_loc, "a").close()
with open(preprocessed_loc, "r") as fp:
    for line in fp.readlines():
        known_iids.append(json.loads(line)["instance_id"])
output = open(preprocessed_loc, "a+")
corpus = []
for reponame in instances_by_repo:
    repoinstances = instances_by_repo[reponame]
    repoinstances = [instance for instance in repoinstances if instance["instance_id"] not in known_iids]
    print(f"Collecting {len(instances_by_repo[reponame])} instances for repo {reponame}.")
    print(f"Collecting {len(repoinstances)} instances not in output file.")
    for instance in tqdm.tqdm(repoinstances):
        source_files_content = {}
        reference_files_content = {}
        hypothesis_files_content = {}
        try:
            hypothesis = hypotheses[instance["instance_id"]]
            repo = GitRepo(reponame.split("/")[0], reponame.split("/")[1], experiments_dir)

            reference_patch: str = instance["patch"]
            reference_files: set = repo.find_absolute_file_paths_from_patch(reference_patch)
            hypothesis_patch: str = hypothesis["model_patch"]
            if not isinstance(hypothesis_patch, str):
                raise ValueError(f"Null patch in instance {instance['instance_id']}")
            hypothesis_files: set = repo.find_absolute_file_paths_from_patch(hypothesis_patch)

            def get_file_contents(files: set[Path]) -> dict[Path, str]:
                # for each file, extract the content
                files_content: dict[Path, str] = {}
                for file in files:
                    if os.path.exists(file):
                        with open(file, "r", encoding="utf-8") as f:
                            files_content[str(file)] = f.read()
                    else:
                        files_content[str(file)] = ""
                return files_content

            relevant_files = reference_files & hypothesis_files
            if relevant_files == set():
                raise ValueError("No overlap in files - not comparable.")
            repo.reset_to_base_commit(instance["base_commit"])
            source_files_content = get_file_contents(relevant_files)
            repo.apply_patch(reference_patch)
            reference_files_content = get_file_contents(relevant_files)

            repo.reset_to_base_commit(instance["base_commit"])
            repo.apply_patch(hypothesis_patch)
            hypothesis_files_content = get_file_contents(relevant_files)
        except Exception as e:
            instance["exception"] = str(e)
            print(e)
        finally:
            instance["source_files_content"] = source_files_content
            instance["reference_files_content"] = reference_files_content
            instance["hypothesis_files_content"] = hypothesis_files_content
            output.write(json.dumps(instance) + "\n")

output.close()

comparable = []
with open(preprocessed_loc, "r") as fp:
    for line in fp:
        instance = json.loads(line)
        if "exception" not in instance or not instance["exception"]:
            instance["reference_files_content"] = {key: val for key, val in instance["reference_files_content"].items() if key.endswith(".py")}
            instance["source_files_content"] = {key: val for key, val in instance["source_files_content"].items() if key.endswith(".py")}
            instance["hypothesis_files_content"] = {key: val for key, val in instance["hypothesis_files_content"].items() if key.endswith(".py")}
            assert len(instance["reference_files_content"]) == len(instance["source_files_content"])
            assert len(instance["reference_files_content"]) == len(instance["hypothesis_files_content"])
            if(len(instance["reference_files_content"]) != 0):  
                comparable.append(instance)

known_iids = []
if not os.path.exists(scored_loc):
    open(scored_loc, "a").close()
with open(scored_loc, "r") as fp:
    for line in fp.readlines():
        known_iids.append(json.loads(line)["instance_id"])
tobescored = [instance for instance in comparable if instance["instance_id"] not in known_iids]

print("Calculating scores")
with open(scored_loc, "a") as fp:
    for instance in tqdm.tqdm(tobescored):
        reference = [val for key, val in sorted(instance["reference_files_content"].items())]
        source = [val for key, val in sorted(instance["source_files_content"].items())]
        hypothesis = [val for key, val in sorted(instance["hypothesis_files_content"].items())]
        instance["codebleu"] = codebleu.calc_codebleu(references = reference, predictions = hypothesis, lang = "python")
        instance["bleu"] = instance["codebleu"]["ngram_match_score"]
        instance["codegleu"] = codegleu.calc_codegleu(sources = source, references = reference, predictions = hypothesis, lang = "python", penalty=1.5)
        fp.write(json.dumps(instance) + "\n")

with open(scored_loc, "r") as fp:
    scored = [json.loads(line) for line in fp.readlines()]
with open(results_loc, "r") as fp:
    results = json.load(fp)
resolved, notresolved = [], []
for instance in scored:
    if instance["instance_id"] in results["resolved"]:
        resolved.append(instance)
    elif instance["instance_id"] in results["applied"]:
        notresolved.append(instance)

res_bleu = 0
res_codebleu = 0
res_codegleu = 0
for instance in resolved:
    res_bleu += instance["bleu"]
    res_codebleu += instance["codebleu"]["codebleu"]
    res_codegleu += instance["codegleu"]["codegleu"]
res_bleu /= len(resolved)
res_codebleu /= len(resolved)
res_codegleu /= len(resolved)

nres_bleu = 0
nres_codebleu = 0
nres_codegleu = 0
for instance in notresolved:
    nres_bleu += instance["bleu"]
    nres_codebleu += instance["codebleu"]["codebleu"]
    nres_codegleu += instance["codegleu"]["codegleu"]
nres_bleu /= len(notresolved)
nres_codebleu /= len(notresolved)
nres_codegleu /= len(notresolved)

resornot = [1] * len(resolved) + [0] * len(notresolved)
bleu_pearson =     pearsonr(resornot, [i["bleu"] for i in resolved] + [i["bleu"] for i in notresolved])
codebleu_pearson = pearsonr(resornot, [i["codebleu"]["codebleu"] for i in resolved] + [i["codebleu"]["codebleu"] for i in notresolved])
codegleu_pearson = pearsonr(resornot, [i["codegleu"]["codegleu"] for i in resolved] + [i["codegleu"]["codegleu"] for i in notresolved])

print(f"Resolved Instance Averages:     BLEU: {res_bleu} CodeBLEU: {res_codebleu} CodeGLEU: {res_codegleu}")
print(f"Non-Resolved Instance Averages: BLEU: {nres_bleu} CodeBLEU: {nres_codebleu} CodeGLEU: {nres_codegleu}")
print(f"Pearson Correlation:            BLEU: {bleu_pearson[0]} CodeBLEU: {codebleu_pearson[0]} CodeGLEU: {codegleu_pearson[0]}")
print(f"Pearson Correlation P:          BLEU: {bleu_pearson[1]} CodeBLEU: {codebleu_pearson[1]} CodeGLEU: {codegleu_pearson[1]}")