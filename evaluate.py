import json
import os
import random
import threading
from multiprocessing import Pool
from pathlib import Path

import black
import matplotlib.pyplot as plt
import pandas as pd
import regex as re
import seaborn as sns
import tqdm
import tqdm.contrib
import tqdm.contrib.concurrent
import unidiff
import wandb
from generate_snippets import generate_snippets
from git_utils import GitRepo
from scipy.stats import pearsonr

import codegleu.codebleu.codebleu as codebleu
import codegleu.codegleu as codegleu
from codegleu.dataflow_match import try_remove_comments_and_docstrings
from codegleu.utils import GenWrapper


def prepare_instances():
    hypotheses = {}
    with open(conf["preds_loc"]) as fp:
        for line in fp.readlines():
            prediction = json.loads(line)
            hypotheses[prediction["instance_id"]] = prediction

    pred_iids = set()
    for pred in hypotheses:
        pred_iids.add(pred)

    inst_iids = set()
    instances_by_repo: dict[str, list[dict]] = {}
    for file in os.listdir(conf["instances_dir"]):
        with open(f"{conf['instances_dir']}/{file}") as fp:
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
    if not os.path.exists(conf["preprocessed_loc"]):
        open(conf["preprocessed_loc"], "a").close()
    with open(conf["preprocessed_loc"], "r") as output:
        for line in output:
            known_iids += [json.loads(line)["instance_id"]]

    with open(conf["preprocessed_loc"], "a+") as output:
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
                        repo = GitRepo(reponame.split("/")[0], reponame.split("/")[1], conf["experiments_dir"])

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
    if not os.path.exists(conf["snippeted_loc"]):
        open(conf["snippeted_loc"], "a").close()
    known_iids = []
    with open(conf["snippeted_loc"], "r") as output:
        for line in output:
            known_iids += [json.loads(line)["instance_id"]]
    with open(conf["preprocessed_loc"], "r") as input:
        tobesnippeted = (json.loads(line) for line in input)
        tobesnippeted = (i for i in tobesnippeted if i["instance_id"] not in known_iids)
        tobesnippeted = (i for i in tobesnippeted if "exception" not in i or not i["exception"])
        tobesnippeted = (i for i in tobesnippeted if len([k for k in i["reference_files_content"] if k.endswith(".py")]) != 0)
        genwrap = GenWrapper(tobesnippeted)
        if genwrap.__nonzero__():
            print(f"Snippeting instances")
            with open(conf["snippeted_loc"], "+a") as output:
                for instance in tqdm.tqdm(genwrap):
                    instance["reference_files_content"] = {
                        key: val for key, val in instance["reference_files_content"].items() if key.endswith(".py")
                    }
                    instance["source_files_content"] = {key: val for key, val in instance["source_files_content"].items() if key.endswith(".py")}
                    instance["hypothesis_files_content"] = {
                        key: val for key, val in instance["hypothesis_files_content"].items() if key.endswith(".py")
                    }
                    assert len(instance["reference_files_content"]) == len(instance["source_files_content"])
                    assert len(instance["reference_files_content"]) == len(instance["hypothesis_files_content"])
                    for i in ["source", "reference", "hypothesis"]:
                        instance[f"{i}_snippets_content"] = {
                            k: generate_snippets(try_remove_comments_and_docstrings(v, lang="python"))
                            for k, v in instance[f"{i}_files_content"].items()
                        }
                    output.write(json.dumps(instance) + "\n")
        else:
            print(f"Already done snippeting")


def clean_code(code: str):
    try:
        code = black.format_file_contents(code, fast=True, mode=black.Mode(target_versions={black.TargetVersion.PY311}, line_length=200))
    except Exception:
        pass
    for c in ["(", ")", "[", "]", "{", "}"]:
        code = re.sub(f"\{c}", f" {c} ", code)
    return code


def score_instances():
    # Calculate scores
    if not os.path.exists(conf["scored_loc"]):
        open(conf["scored_loc"], "a").close()
    known_iids = []
    with open(conf["scored_loc"], "r") as output:
        for line in output:
            known_iids += [json.loads(line)["instance_id"]]
    with open(conf["snippeted_loc"], "r") as input:
        print(f"Calculating scores")
        with open(conf["scored_loc"], "a") as output:
            tobescored = (json.loads(line) for line in input)
            tobescored = (instance for instance in tobescored if instance["instance_id"] not in known_iids)
            genwrap = GenWrapper(tobescored)
            if not genwrap.__nonzero__():
                print(f"Already done scoring")
                return
            input_lock = threading.Lock()
            output_lock = threading.Lock()

            def worker():
                while True:
                    with input_lock:
                        instance = next(genwrap, None)
                    if instance is None:
                        break
                    reference = [clean_code(val) for _, val in sorted(instance["reference_files_content"].items())]
                    hypothesis = [clean_code(val) for _, val in sorted(instance["hypothesis_files_content"].items())]
                    source = [clean_code(val) for _, val in sorted(instance["source_files_content"].items())]
                    instance["codebleu"] = codebleu.calc_codebleu(references=reference, predictions=hypothesis, lang="python")
                    instance["bleu"] = instance["codebleu"]["ngram_match_score"]
                    if "intermediates" not in instance or not instance["intermediates"]:
                        cg = codegleu.calc_codegleu(
                            sources=source,
                            references=reference,
                            predictions=hypothesis,
                            lang="python",
                            penalty=conf["codegleu_penalty"],
                            ret_intermediates=True,
                            n_weights=conf["n_weights"],
                        )
                        intermediates = cg.pop("intermediates")
                        instance["codegleu"] = cg
                    else:
                        instance["codegleu"] = codegleu.calc_codegleu(
                            sources=source,
                            references=reference,
                            predictions=hypothesis,
                            lang="python",
                            penalty=conf["codegleu_penalty"],
                            intermediates=instance["intermediates"],
                            n_weights=conf["n_weights"],
                        )
                    with output_lock:
                        output.write(json.dumps(instance | {"intermediates": intermediates}) + "\n")

            threads = [threading.Thread(target=worker) for _ in range(0, 8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()


def recalc(instance):
    # if not instance["resolved"]:
    instance["codegleu"] = codegleu.calc_codegleu(
        [], [], [], lang="python", penalty=conf["codegleu_penalty"], intermediates=instance["intermediates"], weights=(1, 1, 1, 1)
    )
    # reference = [clean_code(val) for _, val in sorted(instance["reference_files_content"].items())]
    # hypothesis = [clean_code(val) for _, val in sorted(instance["hypothesis_files_content"].items())]
    # source = [clean_code(val) for _, val in sorted(instance["source_files_content"].items())]
    # instance["codegleu"] = codegleu.calc_codegleu(source, reference, hypothesis, lang="python", penalty=conf['codegleu_penalty'], n_weights=conf['n_weights'],)
    # instance["codebleu"] = codebleu.calc_codebleu(reference, hypothesis, lang="python")
    # if instance["codebleu"]["syntax_match_score"] != instance["codegleu"]["syntax_match_score"]:
    #     pass
    filelen = 0
    patchlen = 0
    for patchedFile in unidiff.PatchSet(instance["patch"]):
        hpaths = {"/".join(p.split("\\")[4:]): f for p, f in instance["hypothesis_files_content"].items()}
        if patchedFile.path in hpaths:
            filelen += len(hpaths[patchedFile.path])
            patchlen += len(str(patchedFile))
    instance["patchpercentage"] = 1 - (patchlen / filelen)
    return {k: instance[k] for k in ["codebleu", "codegleu", "bleu", "instance_id", "patchpercentage"]}


conf = {
    "instances_dir": "./data/instances",
    "preds_loc": "./data/sweagent_preds.jsonl",
    "results_loc": "./data/results.json",
    "preprocessed_loc": "./data/preprocessed_instances.jsonl",
    "snippeted_loc": "./data/snippeted_instances.jsonl",
    "scored_loc": "./data/scored_instances.jsonl",
    "experiments_dir": "./experiments",
    "codegleu_penalty": (1, 1, 1, 1),
    "n_weights": (0.1,) * 10,
    "trim": -1,  # size to trim dataset to after filtering
}


def main():
    os.environ["WANDB_SILENT"] = "true"
    wandb.login()
    wandb.init(project="codegleu", config=conf)
    # collect_instances() not implemented, manual labour via swebench collect
    prepare_instances()
    snippet_instances()
    score_instances()

    with open(conf["results_loc"], "r") as fp:
        results = json.load(fp)

    print(f"Recalculating scores")
    scored = []
    totalsize = os.path.getsize(conf["scored_loc"])
    processedruns = 0
    processedinstances = 0
    mem = min(max(256_000_000, int(totalsize / 10)), 512_000_000)
    print(f"Allocating {mem} bytes of buffersize")
    with open(conf["scored_loc"], "r") as fp:
        with tqdm.tqdm(total=1) as pbar:
            while True:
                buffer = fp.readlines(mem)
                if not buffer:
                    break
                for index, line in enumerate(buffer):
                    buffer[index] = json.loads(line)
                    buffer[index]["resolved"] = buffer[index]["instance_id"] in results["resolved"]
                with Pool(5) as pool:
                    rets = pool.map(recalc, buffer, chunksize=5)
                del buffer
                scored += rets
                processedinstances += len(rets)
                processedruns += 1
                pbar.total = int(max(totalsize / mem / processedruns, 1) * processedinstances)
                pbar.update(len(rets))
                pbar.refresh()

    resolved, notresolved = [], []
    with open(conf["results_loc"], "r") as fp:
        results = json.load(fp)
        for instance in scored:
            if instance["instance_id"] in results["resolved"]:
                resolved.append(instance)
            elif instance["instance_id"] in results["applied"]:
                notresolved.append(instance)

    total = len(resolved) + len(notresolved)
    targetsize = min(conf["trim"], total) if conf["trim"] != -1 else total
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

    pr = pearsonr(resornot, [i["patchpercentage"] for i in toscore])
    print(f"Patch percentage vs resolved:   Correlation: {pr[0]} P-Value: {pr[1]} ")

    toscore = [i | {"bleu": {"bleu": i["bleu"]}} for i in toscore]
    padlen = 26
    print(" " * 32 + "metric vs resolved          " + "metric vs patchpercent")
    print(" " * 32 + "Correlation   P-Value       " + "Correlation   P-Value")
    for group in ["bleu", "codebleu", "codegleu"]:
        print(f"Performing Ablation Study for {group}")
        scores = toscore[0][group]
        if isinstance(scores, dict):
            for score in scores:
                print(f"    {score + ' ' * (padlen-len(score))} ", end="")
                pr = pearsonr(resornot, [i[group][score] for i in toscore])
                print(f" {'%.10f' % pr[0]}, {'%.10f' % pr[1]} ", end="")
                wandb.log({f"mvr_{group}_{score}_corr": pr[0], f"mvr_{group}_{score}_p": pr[1]})
                pr = pearsonr([i[group][score] for i in toscore], [i["patchpercentage"] for i in toscore])
                print(f" {'%.10f' % pr[0]}, {'%.10f' % pr[1]} ")
                wandb.log({f"mvp_{group}_{score}_corr": pr[0], f"mvp_{group}_{score}_p": pr[1]})
    data = [s["codegleu"] | {"group": "passed" if r else "failed"} for s, r in zip(toscore, resornot)]
    fig, axs = plt.subplots(ncols=5, figsize=(20, 5))
    for i, k in enumerate(toscore[0]["codegleu"]):
        sns.histplot(x=k, hue="group", data=pd.DataFrame(data), palette={"passed": "green", "failed": "red"}, binwidth=0.02, ax=axs[i])
    fig.savefig(f"./figs/codegleu_scores.png")
    wandb.finish()


if __name__ == "__main__":
    main()
pass
