import json
import os
import random
from pathlib import Path

import regex as re
import tqdm
import tqdm.contrib
import tqdm.contrib.concurrent
from generate_snippets import generate_snippets
from git_utils import GitRepo
from scipy.stats import pearsonr

import codegleu.codebleu.codebleu as codebleu
import codegleu.codegleu as codegleu
from codegleu.dataflow_match import try_remove_comments_and_docstrings

instances_dir = "./data/instances"
preds_loc = "./data/sweagent_preds.jsonl"
results_loc = "./data/results.json"
preprocessed_loc = "./data/preprocessed_instances.jsonl"
snippeted_loc = "./data/snippeted_instances.jsonl"
scored_loc = "./data/scored_instances.jsonl"
experiments_dir = "./experiments"
trim = -1  # size to trim dataset to after filtering

def prepare_instances():
    hypotheses = {}
    with open(preds_loc) as fp:
        for line in fp.readlines():
            prediction = json.loads(line)
            hypotheses[prediction["instance_id"]] = prediction

    pred_iids = set()
    for pred in hypotheses:
        pred_iids.add(pred)

    inst_iids = set()
    instances_by_repo: dict[str, list[dict]] = {}
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
    with open(preprocessed_loc, "r") as output:
        for line in output:
            known_iids += [json.loads(line)["instance_id"]]

    with open(preprocessed_loc, "a+") as output:
        for reponame in instances_by_repo:
            repoinstances = instances_by_repo[reponame]
            repoinstances = [instance for instance in repoinstances if instance["instance_id"] not in known_iids]
            if len(repoinstances) > 0:
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

                        def get_file_contents(files: set[Path]) -> dict[str, str]:
                            # for each file, extract the content
                            files_content: dict[str, str] = {}
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
            else:
                print(f"Already collected instances for repo {reponame}.")


def snippet_instances():
    # filter to python files and prepare snippets
    if not os.path.exists(snippeted_loc):
        open(snippeted_loc, "a").close()
    known_iids = []
    with open(snippeted_loc, "r") as output:
        for line in output:
            known_iids += [json.loads(line)["instance_id"]]
    with open(preprocessed_loc, "r") as input:
        tobesnippeted = [json.loads(line) for line in input.readlines()]
        tobesnippeted = [i for i in tobesnippeted if i["instance_id"] not in known_iids]
        tobesnippeted = [i for i in tobesnippeted if "exception" not in i or not i["exception"]]
        tobesnippeted = [i for i in tobesnippeted if len([k for k in i["reference_files_content"] if k.endswith(".py")]) != 0]

    if len(tobesnippeted) > 0:
        print(f"Snippeting {len(tobesnippeted)} instances")
        with open(snippeted_loc, "+a") as output:
            for instance in tqdm.tqdm(tobesnippeted):
                instance["reference_files_content"] = {key: val for key, val in instance["reference_files_content"].items() if key.endswith(".py")}
                instance["source_files_content"] = {key: val for key, val in instance["source_files_content"].items() if key.endswith(".py")}
                instance["hypothesis_files_content"] = {key: val for key, val in instance["hypothesis_files_content"].items() if key.endswith(".py")}
                assert len(instance["reference_files_content"]) == len(instance["source_files_content"])
                assert len(instance["reference_files_content"]) == len(instance["hypothesis_files_content"])
                for i in ["source", "reference", "hypothesis"]:
                    instance[f"{i}_snippets_content"] = {
                        k: generate_snippets(try_remove_comments_and_docstrings(v, lang="python")) for k, v in instance[f"{i}_files_content"].items()
                    }
                output.write(json.dumps(instance) + "\n")
    else:
        print(f"Already done snippeting")


def score_instances():
    # Calculate scores
    if not os.path.exists(scored_loc):
        open(scored_loc, "a").close()
    known_iids = []
    with open(scored_loc, "r") as output:
        for line in output:
            known_iids += [json.loads(line)["instance_id"]]
    with open(snippeted_loc, "r") as input:
        tobescored = [json.loads(line) for line in input.readlines()]
        tobescored = [instance for instance in tobescored if instance["instance_id"] not in known_iids]

    if len(tobescored) > 0:
        print(f"Calculating {len(tobescored)} scores")
        with open(scored_loc, "a") as output:
            for instance in tqdm.tqdm(tobescored):
                reference = [val for _, val in sorted(instance["reference_files_content"].items())]
                source = [val for _, val in sorted(instance["source_files_content"].items())]
                hypothesis = [val for _, val in sorted(instance["hypothesis_files_content"].items())]
                instance["codebleu"] = codebleu.calc_codebleu(references=reference, predictions=hypothesis, lang="python")
                instance["bleu"] = instance["codebleu"]["ngram_match_score"]
                if "intermediates" not in instance or not instance["intermediates"]:
                    cg = codegleu.calc_codegleu(sources=source, references=reference, predictions=hypothesis, lang="python", penalty=codegleu_penalty, ret_intermediates=True)
                    intermediates = cg.pop("intermediates")
                    instance["codegleu"] = cg
                else:
                    instance["codegleu"] = codegleu.calc_codegleu(sources=source, references=reference, predictions=hypothesis, lang="python", penalty=codegleu_penalty, intermediates=instance["intermediates"])
                output.write(json.dumps(instance | {"intermediates": intermediates}) + "\n")
    else:
        print(f"Already done scoring")


def recalc(instance):
    instance = json.loads(instance)
    instance["codegleu"] = codegleu.calc_codegleu([], [], [], lang="python", penalty=codegleu_penalty, intermediates=instance["intermediates"])
    return {k: instance[k] for k in ["codebleu", "codegleu", "bleu", "instance_id"]}


codegleu_penalty = (0,0,0,0)


def main():
    # collect_instances() not implemented, manual labour via swebench collect
    prepare_instances()
    snippet_instances()
    score_instances()

    print(f"Recalculating scores")
    scored = []
    processed = 0
    totalsize = os.path.getsize(scored_loc)
    mem = min(max(250_000_000, int(totalsize/5)), 500_000_000)
    print(f"Allocating {mem} bytes of buffersize")
    with open(scored_loc, "r") as fp:
        buffer = fp.readlines(mem)
        while buffer:
            scored += tqdm.contrib.concurrent.process_map(recalc, buffer, chunksize=5, max_workers=5)
            processed += 1
            print(f"Processed a total of {len(scored)} instances, {processed}/{int(totalsize / mem + 0.5)} expected runs")
            buffer = fp.readlines(mem)

    resolved, notresolved = [], []
    with open(results_loc, "r") as fp:
        results = json.load(fp)
        for instance in scored:
            if instance["instance_id"] in results["resolved"]:
                resolved.append(instance)
            elif instance["instance_id"] in results["applied"]:
                notresolved.append(instance)

    total = len(resolved) + len(notresolved)
    targetsize = min(trim, total) if trim != -1 else total
    resolved = random.sample(resolved, int(targetsize * len(resolved) / total + 0.5))
    notresolved = random.sample(notresolved, int(targetsize * len(notresolved) / total + 0.5))

    res_bleu = sum([i["bleu"] for i in resolved]) / len(resolved)
    res_codebleu = sum([i["codebleu"]["codebleu"] for i in resolved]) / len(resolved)
    res_codegleu = sum([i["codegleu"]["codegleu"] for i in resolved]) / len(resolved)

    nres_bleu = sum([i["bleu"] for i in notresolved]) / len(notresolved)
    nres_codebleu = sum([i["codebleu"]["codebleu"] for i in notresolved]) / len(notresolved)
    nres_codegleu = sum([i["codegleu"]["codegleu"] for i in notresolved]) / len(notresolved)

    toscore = resolved + notresolved
    resornot = [1] * len(resolved) + [0] * len(notresolved)

    print(f"Resolved Instance Averages:     BLEU: {res_bleu} CodeBLEU: {res_codebleu} CodeGLEU: {res_codegleu}")
    print(f"Non-Resolved Instance Averages: BLEU: {nres_bleu} CodeBLEU: {nres_codebleu} CodeGLEU: {nres_codegleu}")

    padlen = 26
    for group in ["bleu", "codebleu", "codegleu"]:
        print(f"Performing Ablation Study for {group}")
        scores = toscore[0][group]
        if isinstance(scores, dict):
            for score in scores:
                comp = [i[group][score] for i in toscore]
                pr = pearsonr(resornot, comp)
                print(f"    {score + ' ' * (padlen-len(score))}  Correlation: {'%.10f' % pr[0]}, P-Value {pr[1]}")
        else:
            pr = pearsonr(resornot, [i[group] for i in toscore])
            print(f"    {group + ' ' * (padlen-len(group))}  Correlation: {'%.10f' % pr[0]}, P-Value {pr[1]}")

if __name__ == "__main__":
    main()
pass
