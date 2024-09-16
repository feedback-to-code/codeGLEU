# mypy: ignore-errors
import itertools
import json
import os
import random
import threading
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import argparse
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
from collect import tasks_for_repo
from generate_snippets import generate_snippets
from git_utils import GitRepo
from scipy.stats import pearsonr
from secret_secrets import GITHUB_TOKENS

import codegleu.codebleu.codebleu as codebleu
import codegleu.codegleu.codegleu as codegleu
import codegleu.diffsim as diffsim
from codegleu.dataflow_match import try_remove_comments_and_docstrings
from codegleu.utils import GenWrapper


class EvalError(Exception):
    pass


def dprint(*args, **kwargs):
    if conf["verbose"]:
        print(*args, **kwargs)


def collect_instances():
    global pred_instances
    global valid_instances
    global invalid_instances
    global total_instances
    invalid_loc = conf["data_dir"] + "/invalid_instances.jsonl"
    collected_iids = set()
    coll = []
    for file in os.listdir(conf["instances_dir"]):
        with open(f"{conf['instances_dir']}/{file}") as fp:
            for line in fp:
                collected_iids.add(json.loads(line)["instance_id"])
                coll.append(json.loads(line)["instance_id"])
                valid_instances += 1
                total_instances += 1

    invalid_iids = []
    if os.path.exists(invalid_loc):
        with open(invalid_loc, mode="r+") as invalid:
            invalid_iids = [iid.removesuffix("\n") for iid in invalid.readlines()]
            invalid_instances += len(invalid_iids)
            total_instances += len(invalid_iids)
            collected_iids.update(invalid_iids)

    pred_iids = []
    missing_by_repo: dict[str, set] = {}
    with open(conf["preds_loc"], mode="r+") as fp:
        for line in fp:
            pred_instances += 1
            iid = json.loads(line)["instance_id"]
            pred_iids += [iid]
            rm = re.match(r"(.*?)__(.*?)-([0-9]+)", iid)
            repo = rm.group(1) + "/" + rm.group(2)
            if iid not in collected_iids:
                if repo not in missing_by_repo:
                    missing_by_repo[repo] = set()
                if rm.group(3) not in missing_by_repo[repo]:
                    missing_by_repo[repo].add(rm.group(3))
    coll = {k: v for k, v in Counter(coll).items() if v > 1}
    if not missing_by_repo:
        print("Already done collecting files")
        return

    with open(invalid_loc, mode="a+") as invalid:
        for repo in missing_by_repo:
            with open(conf["instances_dir"] + f"/{repo.split('/')[1]}-task-instances.jsonl", mode="a+") as output:
                for task in tasks_for_repo(repo, missing_by_repo[repo], GITHUB_TOKENS):
                    if task["valid"]:
                        output.write(json.dumps(task) + "\n")
                        output.flush()
                        valid_instances += 1
                    else:
                        invalid.write(task["instance_id"] + "\n")
                        invalid.flush()
                        invalid_instances += 1
                    total_instances += 1
    print(
        (f"pred_instances = {pred_instances}\n"),
        (f"invalid_instances = {invalid_instances}\n"),
        (f"valid_instances = {valid_instances}\n"),
        (f"total_instances = {total_instances}\n"),
    )


def prepare_instances():
    global no_comparable_files
    global error_applying_patches
    global prepared_file_contents
    global total_prepared_file_contents

    inst_iids = set()
    instances_by_repo: dict[str, dict[str, dict]] = {}
    for file in os.listdir(conf["instances_dir"]):
        with open(f"{conf['instances_dir']}/{file}") as fp:
            for line in fp:
                instance = json.loads(line)
                repo = instance["repo"]
                iid = instance["instance_id"]
                if repo not in instances_by_repo:
                    instances_by_repo[repo] = {}
                instances_by_repo[repo][iid] = instance
                inst_iids.add(iid)

    hypotheses_by_repo: dict[str, list[dict]] = {}
    with open(conf["preds_loc"]) as fp:
        for line in fp:
            prediction = json.loads(line)
            iid = prediction["instance_id"]
            rm = re.match(r"(.*?)__(.*?)-([0-9]+)", iid)
            repo = rm.group(1) + "/" + rm.group(2)
            if iid in inst_iids:
                if repo not in hypotheses_by_repo:
                    hypotheses_by_repo[repo] = []
                hypotheses_by_repo[repo] += [prediction]

    known_iids = set()
    if not os.path.exists(conf["preprocessed_loc"]):
        open(conf["preprocessed_loc"], "a").close()
    with open(conf["preprocessed_loc"], "r") as output:
        for line in output:
            known_iids.add(json.loads(line)["instance_id"])

    if not (inst_iids ^ known_iids):
        print(f"Already done preprocessing files")
        return
    print("Collecting files and applying patches")
    with open(conf["preprocessed_loc"], "a+") as output:
        for reponame in hypotheses_by_repo:
            repo = GitRepo(reponame.split("/")[0], reponame.split("/")[1], conf["experiments_dir"])
            repohypos = hypotheses_by_repo[reponame]
            repohypos = [hypo for hypo in repohypos if hypo["instance_id"] not in known_iids]
            if len(repohypos) > 0:
                print(f"Collecting {len(hypotheses_by_repo[reponame])} instances for repo {reponame}.")
                print(f"Collecting {len(repohypos)} instances not in output file.")
                for hypothesis in tqdm.tqdm(repohypos):
                    source_files_content = {}
                    reference_files_content = {}
                    hypothesis_files_content = {}
                    try:
                        instance = instances_by_repo[reponame][hypothesis["instance_id"]]

                        reference_patch: str = instance["patch"]
                        hypothesis_patch: str = hypothesis["model_patch"]
                        instance["model_patch"] = hypothesis_patch
                        repo.reset_to_base_commit(instance["base_commit"])
                        reference_files: set = repo.find_absolute_file_paths_from_patch(reference_patch)
                        if not isinstance(hypothesis_patch, str) or not hypothesis_patch:
                            error_applying_patches += 1
                            raise EvalError(f"Null patch in instance {instance['instance_id']}")
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
                            no_comparable_files += 1
                            raise EvalError("No overlap in files - not comparable.")
                        repo.reset_to_base_commit(instance["base_commit"])
                        source_files_content = get_file_contents(relevant_files)
                        repo.apply_patch(reference_patch)
                        reference_files_content = get_file_contents(relevant_files)

                        repo.reset_to_base_commit(instance["base_commit"])
                        repo.apply_patch(hypothesis_patch)
                        hypothesis_files_content = get_file_contents(relevant_files)
                        prepared_file_contents += 1
                    except EvalError as e:
                        instance["exception"] = str(e)
                    except Exception as e:
                        instance["exception"] = str(e)
                        error_applying_patches += 1
                        # print(e)
                    finally:
                        instance["source_files_content"] = source_files_content
                        instance["reference_files_content"] = reference_files_content
                        instance["hypothesis_files_content"] = hypothesis_files_content
                        total_prepared_file_contents += 1
                        output.write(json.dumps(instance) + "\n")
            else:
                print(f"Already preprocessed files for repo {reponame}.")
    print(
        (f"no_comparable_files = {no_comparable_files}\n"),
        (f"error_applying_patches = {error_applying_patches}\n"),
        (f"prepared_file_contents = {prepared_file_contents}\n"),
        (f"total_prepared_file_contents = {total_prepared_file_contents}\n"),
    )


def clip_instances():
    global no_code_files
    global selected_code_files
    global different_files
    global total_code_files
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
        genwrap = GenWrapper(tobesnippeted)
        if not genwrap.__nonzero__():
            print(f"Already done snippeting")
            return
        print(f"Snippeting instances")
        with open(conf["snippeted_loc"], "+a") as output:
            for instance in tqdm.tqdm(genwrap):
                total_code_files += 1
                if len([k for k in instance["reference_files_content"] if k.endswith(".py")]) == 0:
                    no_code_files += 1
                    continue
                instance["reference_files_content"] = {key: val for key, val in instance["reference_files_content"].items() if key.endswith(".py")}
                instance["source_files_content"] = {key: val for key, val in instance["source_files_content"].items() if key.endswith(".py")}
                instance["hypothesis_files_content"] = {key: val for key, val in instance["hypothesis_files_content"].items() if key.endswith(".py")}
                try:
                    assert len(instance["reference_files_content"]) == len(instance["source_files_content"])
                    assert len(instance["reference_files_content"]) == len(instance["hypothesis_files_content"])
                except:
                    different_files += 1
                    continue
                # for i in ["source", "reference", "hypothesis"]:
                #     instance[f"{i}_snippets_content"] = {
                #         k: generate_snippets(try_remove_comments_and_docstrings(v, lang="python")) for k, v in instance[f"{i}_files_content"].items()
                #     }
                selected_code_files += 1
                output.write(json.dumps(instance) + "\n")
    print(
        (f"no_code_files = {no_code_files}\n"),
        (f"different_files = {different_files}\n"),
        (f"selected_code_files = {selected_code_files}\n"),
        (f"total_code_files = {total_code_files}\n"),
    )


def clean_code(code: str):
    try:
        code = black.format_file_contents(code, fast=True, mode=black.Mode(target_versions={black.TargetVersion.PY311}, line_length=200))
    except Exception:
        pass
    for c in ["(", ")", "[", "]", "{", "}"]:
        code = re.sub(f"\{c}", f" {c} ", code)
    return code


input_lock = threading.Lock()
output_lock = threading.Lock()


def score_instances():
    # Calculate scores
    if not os.path.exists(conf["scored_loc"]):
        open(conf["scored_loc"], "a").close()
    input_lines = 0
    output_lines = 0
    known_iids = []
    with open(conf["scored_loc"], "r") as output:
        for line in output:
            known_iids += [json.loads(line)["instance_id"]]
            output_lines += 1
    with open(conf["snippeted_loc"], "r") as input:
        for line in input:
            input_lines += 1
    with open(conf["snippeted_loc"], "r") as input:
        with open(conf["scored_loc"], "a") as output:
            tobescored = (json.loads(line) for line in input)
            tobescored = (instance for instance in tobescored if instance["instance_id"] not in known_iids)
            genwrap = GenWrapper(tobescored)
            if not genwrap.__nonzero__():
                print(f"Already done scoring")
                return
            print(f"Calculating scores")
            pbar = tqdm.tqdm(total = input_lines - output_lines, smoothing = 0)

            def worker():
                while True:
                    s = time.time()
                    with input_lock:
                        instance = next(genwrap, None)
                    if instance is None:
                        break
                    dprint(f"acquired instance for {instance['instance_id']} in {time.time() - s}s")
                    s = time.time()
                    reference = [clean_code(val) for _, val in sorted(instance["reference_files_content"].items())]
                    hypothesis = [clean_code(val) for _, val in sorted(instance["hypothesis_files_content"].items())]
                    source = [clean_code(val) for _, val in sorted(instance["source_files_content"].items())]
                    dprint(f"cleaned code for {instance['instance_id']} in {time.time() - s}s")
                    s = time.time()
                    instance["codebleu"] = codebleu.calc_codebleu(references=reference, predictions=hypothesis, lang="python")
                    instance["codebleu_patch"] = codebleu.calc_codebleu([instance["patch"]], [instance["model_patch"]], lang="python")
                    instance["bleu"] = instance["codebleu"]["ngram_match_score"]
                    if "intermediates" not in instance or not instance["intermediates"]:
                        cg = diffsim.calc_diffsim(
                            sources=source,
                            references=reference,
                            hypotheses=hypothesis,
                            lang="python",
                            penalty=conf["diffsimpenalty"],
                            ret_intermediates=True,
                            n_weights=conf["n_weights"],
                        )
                        intermediates = cg.pop("intermediates")
                        instance["diffsim"] = cg
                    else:
                        instance["diffsim"] = diffsim.calc_diffsim(
                            sources=source,
                            references=reference,
                            hypotheses=hypothesis,
                            lang="python",
                            penalty=conf["diffsimpenalty"],
                            intermediates=instance["intermediates"],
                            n_weights=conf["n_weights"],
                        )
                    instance["codegleu"] = codegleu.calc_codegleu(
                        [], [], [], lang="python", penalty=conf["codegleupenalty"], intermediates=intermediates, weights=conf["weights"]
                    )
                    filelen = 0
                    patchlen = 0
                    for patchedFile in unidiff.PatchSet(instance["patch"]):
                        hpaths: dict[str, Any] = instance["hypothesis_files_content"]
                        if list(hpaths.keys())[0].split("\\")[0] == "experiments":
                            hpaths = {"/".join(p.split("\\")[4:]): f for p, f in hpaths.items()}
                        if list(hpaths.keys())[0].split("/")[0] == "experiments":
                            hpaths = {"/".join(p.split("/")[4:]): f for p, f in hpaths.items()}
                        if patchedFile.path in hpaths:
                            filelen += len(hpaths[patchedFile.path])
                            patchlen += len(str("".join("".join(str(line) for line in hunk) for hunk in patchedFile))) / 2
                    instance["unchangedpercentage"] = max(0, 1 - (patchlen / filelen))
                    dprint(f"calculated scores for {instance['instance_id']} in {time.time() - s}s")
                    with output_lock:
                        pbar.update(1)
                        output.write(json.dumps(instance | {"intermediates": intermediates}) + "\n")

            # threads = [threading.Thread(target=worker) for _ in range(0, 2)]
            # for thread in threads:
            #     thread.start()
            # for thread in threads:
            #     thread.join()
            worker()


def recalc(instance):
    # if instance["instance_id"] in extraref_intermediates:
    #     newintermediates = instance["intermediates"]
    #     for key in newintermediates:
    #         newintermediates[key]["r_interms"] += extraref_intermediates[instance["instance_id"]][key]["r_interms"]
    #     instance["diffsim"] = diffsim.calc_diffsim([], [], [], lang="python", penalty=conf["diffsimpenalty"], intermediates=newintermediates, weights=conf["weights"])
    if "unchangedpercentage" not in instance or not instance["unchangedpercentage"]:
        filelen = 0
        patchlen = 0
        for patchedFile in unidiff.PatchSet(instance["patch"]):
            hpaths: dict[str, Any] = instance["hypothesis_files_content"]
            if list(hpaths.keys())[0].split("\\")[0] == "experiments":
                hpaths = {"/".join(p.split("\\")[4:]): f for p, f in hpaths.items()}
            if list(hpaths.keys())[0].split("/")[0] == "experiments":
                hpaths = {"/".join(p.split("/")[4:]): f for p, f in hpaths.items()}
            if patchedFile.path in hpaths:
                filelen += len(hpaths[patchedFile.path])
                patchlen += len(str("".join("".join(str(line) for line in hunk) for hunk in patchedFile))) / 2
        instance["unchangedpercentage"] = max(0, 1 - (patchlen / filelen))
    return {k: instance[k] for k in ["codebleu", "codebleu_patch", "codegleu", "diffsim", "bleu", "instance_id", "unchangedpercentage"]}


def write_log(*args, **kwargs):
    with open(conf["log_loc"], "a+") as log_file:
        print(*args, file=log_file, **kwargs)
    print(*args, **kwargs)


conf = {
    "data_dir": "./data",
    "experiments_dir": "./experiments",
    "trim": -1,  # size to trim dataset to after filtering
    "model": "20240402_sweagent_gpt4",
    "diffsimpenalty": (0.25, 0.25, 0.25, 0.25),
    "codegleupenalty": (1, 1, 1, 1),
    "weights": (1,) * 4,
    "n_weights": (0.25,) * 4,
    "verbose": False,
    "rankingscores_loc": f"./results/rankingscores.txt", 
}

# extraref_intermediates = {}
# with open("data/models/20240820_honeycomb/scored_instances.jsonl") as fp:
#     for line in tqdm.tqdm(fp):
#         inst = json.loads(line)
#         intermediates = inst["intermediates"]
#         for key in intermediates:
#             intermediates[key] = {"r_interms": [intermediates[key]["h_interm"]]}
#         extraref_intermediates[inst["instance_id"]] = intermediates
#         del inst
# pass
def main():
    conf["instances_dir"] = f"{conf['data_dir']}/instances"
    conf["model_path"] = f"{conf['data_dir']}/models/{conf['model']}"
    conf["preds_loc"] = f"{conf['model_path']}/all_preds.jsonl"
    conf["results_loc"] = f"{conf['model_path']}/results/results.json"
    conf["preprocessed_loc"] = f"{conf['model_path']}/preprocessed_instances.jsonl"
    conf["snippeted_loc"] = f"{conf['model_path']}/snippeted_instances.jsonl"
    conf["scored_loc"] = f"{conf['model_path']}/scored_instances.jsonl"
    conf["figs_loc"] = f"./figs/{conf['model']}"
    conf["log_loc"] = f"./results/{conf['model']}.txt"
    os.environ["WANDB_SILENT"] = "true"
    wandb.login()
    wandb.init(project="codegleu", config=conf)
    collect_instances()
    prepare_instances()
    clip_instances()
    score_instances()
    print(f"eval for model {conf['model']} \n")
    with open(conf["log_loc"], "w+") as log_file:
        print(f"log for model {conf['model']} \n", file=log_file)
    write_log(
        (f"pred_instances = {pred_instances}\n"),
        (f"invalid_instances = {invalid_instances}\n"),
        (f"valid_instances = {valid_instances}\n"),
        (f"total_instances = {total_instances}\n"),
        (f"no_comparable_files = {no_comparable_files}\n"),
        (f"error_applying_patches = {error_applying_patches}\n"),
        (f"prepared_file_contents = {prepared_file_contents}\n"),
        (f"total_prepared_file_contents = {total_prepared_file_contents}\n"),
        (f"no_code_files = {no_code_files}\n"),
        (f"selected_code_files = {selected_code_files}\n"),
        (f"different_files = {different_files}\n"),
        (f"total_code_files = {total_code_files}\n"),
    )

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
                processedinstances += len(buffer)
                processedruns += 1
                pbar.total = int(max(totalsize / mem / processedruns, 1) * processedinstances)
                pbar.refresh()
                for index, line in enumerate(buffer):
                    buffer[index] = json.loads(line)
                    buffer[index]["resolved"] = buffer[index]["instance_id"] in results["resolved"]
                # with Pool(5) as pool:
                #     rets = pool.map(recalc, buffer, chunksize=5)
                rets = [recalc(i) for i in buffer]
                del buffer
                scored += rets
                pbar.update(len(rets))
                pbar.refresh()

    resolved, notresolved = [], []
    with open(conf["results_loc"], "r") as fp:
        results = json.load(fp)
        for instance in scored:
            # if instance["instance_id"] not in extraref_intermediates:
            #     continue
            if instance["instance_id"] in results["resolved"]:
                resolved.append(instance)
            # elif instance["instance_id"] in results["applied"]:
            else:
                notresolved.append(instance)

    # total = len(resolved) + len(notresolved)
    # targetsize = min(conf["trim"], total) if conf["trim"] != -1 else total
    # resolved = random.sample(resolved, int(targetsize * len(resolved) / total + 0.5))
    # notresolved = random.sample(notresolved, int(targetsize * len(notresolved) / total + 0.5))

    if len(resolved) > 0:
        res_bleu = sum([i["bleu"] for i in resolved]) / len(resolved)
        res_codebleu = sum([i["codebleu"]["codebleu"] for i in resolved]) / len(resolved)
        res_codebleu_p = sum([i["codebleu_patch"]["codebleu"] for i in resolved]) / len(resolved)
        res_codegleu = sum([i["codegleu"]["codegleu"] for i in resolved]) / len(resolved)
        res_diffsim = sum([i["diffsim"]["diffsim"] for i in resolved]) / len(resolved)
    else:
        res_bleu = res_codebleu = res_codebleu_p = res_codegleu = res_diffsim = 0

    if len(notresolved) > 0:
        nres_bleu = sum([i["bleu"] for i in notresolved]) / len(notresolved)
        nres_codebleu = sum([i["codebleu"]["codebleu"] for i in notresolved]) / len(notresolved)
        nres_codebleu_p = sum([i["codebleu_patch"]["codebleu"] for i in notresolved]) / len(notresolved)
        nres_codegleu = sum([i["codegleu"]["codegleu"] for i in notresolved]) / len(notresolved)
        nres_diffsim = sum([i["diffsim"]["diffsim"] for i in notresolved]) / len(notresolved)
    else:
        nres_bleu = nres_codebleu = nres_codebleu_p = nres_codegleu = nres_diffsim = 0

    toscore = resolved + notresolved
    resornot = [1] * len(resolved) + [0] * len(notresolved)

    tot_bleu = sum([i["bleu"] for i in toscore]) / len(toscore)
    tot_codebleu = sum([i["codebleu"]["codebleu"] for i in toscore]) / len(toscore)
    tot_codebleu_p = sum([i["codebleu_patch"]["codebleu"] for i in toscore]) / len(toscore)
    tot_codegleu = sum([i["codegleu"]["codegleu"] for i in toscore]) / len(toscore)
    tot_diffsim = sum([i["diffsim"]["diffsim"] for i in toscore]) / len(toscore)

    entries = []
    with open(conf["rankingscores_loc"], "r+") as fp:
        for line in fp:
            js = json.loads(line)
            if "model" in js and js["model"] != conf["model"]:
                entries.append(js)
    scores = {"model": conf["model"], "bleu": tot_bleu, "codebleu": tot_codebleu, "codegleu": tot_codegleu, "codebleu_p": tot_codebleu_p, "diffsim": tot_diffsim, "actual": len(resolved) / len(toscore)}
    entries.append(scores)
    with open(conf["rankingscores_loc"], "w+") as fp:
        for js in entries:
            fp.write(json.dumps(js) + "\n")

    write_log(f"Instances. {len(toscore)}. resolved: {len(resolved)}. not resolved: {len(notresolved)}. % resolved: {len(resolved) / len(toscore)}")
    write_log(f"Total Instance Averages:        BLEU: {tot_bleu} CodeBLEU: {tot_codebleu} CodeGLEU: {tot_codegleu} CodeBLEU_p: {tot_codebleu_p} FRITS: {tot_diffsim}")
    write_log(f"Resolved Instance Averages:     BLEU: {res_bleu} CodeBLEU: {res_codebleu} CodeGLEU: {res_codegleu} CodeBLEU_p: {res_codebleu_p} FRITS: {res_diffsim}")
    write_log(f"Non-Resolved Instance Averages: BLEU: {nres_bleu} CodeBLEU: {nres_codebleu} CodeGLEU: {nres_codegleu} CodeBLEU_p: {nres_codebleu_p} FRITS: {nres_diffsim}")

    pr = pearsonr(resornot, [i["unchangedpercentage"] for i in toscore])
    write_log(f"Patch percentage vs resolved:   Correlation: {pr[0]} P-Value: {pr[1]} ")

    toscore = [i | {"bleu": {"bleu": i["bleu"]}} for i in toscore]
    padlen = 26
    write_log(" " * 32 + "metric vs resolved          " + "metric vs unchangedpercent")
    write_log(" " * 32 + "Correlation   P-Value       " + "Correlation   P-Value")
    for group in ["bleu", "codebleu", "codebleu_patch", "codegleu", "diffsim"]:
        write_log(f"Performing Ablation Study for {group}")
        scores = toscore[0][group]
        if isinstance(scores, dict):
            for score in scores:
                write_log(f"    {score + ' ' * (padlen-len(score))} ", end="")
                pr = pearsonr(resornot, [i[group][score] for i in toscore])
                write_log(f" {'%.10f' % pr[0]}, " + str(pr[1]), end="")
                wandb.log({f"mvr_{group}_{score}_corr": pr[0], f"mvr_{group}_{score}_p": pr[1]})
                pr = pearsonr([i[group][score] for i in toscore], [i["unchangedpercentage"] for i in toscore])
                write_log(f" {'%.10f' % pr[0]}, " + str(pr[1]))
                wandb.log({f"mvp_{group}_{score}_corr": pr[0], f"mvp_{group}_{score}_p": pr[1]})
    if not os.path.exists(conf["figs_loc"]):
        os.mkdir(conf["figs_loc"])
    for n in ["codebleu", "codebleu_patch", "codegleu", "diffsim"]:
        data = [s[n] | {"group": "resolved" if r else "not resolved"} for s, r in zip(toscore, resornot)]
        fig, axs = plt.subplots(ncols=5, figsize=(25, 5))
        plots = []
        for i, k in enumerate(toscore[0][n]):
            p = sns.histplot(x=k, hue="group", data=pd.DataFrame(data), palette={"resolved": "green", "not resolved": "red"}, binwidth=0.02, ax=axs[i])
            names = {"ngram_match_score": "Match$_{df}$", "weighted_ngram_match_score": "Match$_{kw}$", "syntax_match_score": "Match$_{ast}$", "dataflow_match_score": "Match$_{df}$"}
            p.set(xlabel=(names[k] if k in names else k))
            plots.append(p)
        maxlim = max([p.get_ylim()[1] for p in plots])
        for p in plots:
            p.set_ylim(top = maxlim, bottom = 0)
        plt.subplots_adjust(hspace = 0.4, wspace = 0.4)
        fig.savefig(f"{conf['figs_loc']}/{n}_scores.png")
        plt.close()
    sns.regplot(x=[i["diffsim"]["diffsim"] for i in toscore], y=resornot)
    plt.savefig(f"{conf['figs_loc']}/resornot_vs_diffsim.png")
    plt.close()
    sns.regplot(x=[i["unchangedpercentage"] for i in toscore], y=[i["codebleu"]["codebleu"] for i in toscore])
    plt.savefig(f"{conf['figs_loc']}/codebleu_vs_unchangedpercent.png")
    plt.close()
    scoregroups = [
        {"name": "bleu", "scores": [i["bleu"]["bleu"] for i in toscore]},
        {"name": "codebleu", "scores": [i["codebleu"]["codebleu"] for i in toscore]},
        {"name": "codebleu_patch", "scores": [i["codebleu_patch"]["codebleu"] for i in toscore]},
        {"name": "codegleu", "scores": [i["codegleu"]["codegleu"] for i in toscore]},
        {"name": "diffsim", "scores": [i["diffsim"]["diffsim"] for i in toscore]},
    ]
    scoregroups = [i | {"mvr": pearsonr(resornot, i["scores"])} for i in scoregroups]
    scoregroups = [i | {"mvr_c": int(100 * i["mvr"][0]), "mvr_p": int(100 * i["mvr"][1])} for i in scoregroups]
    scoregroups = [i | {"mvp": pearsonr([i["unchangedpercentage"] for i in toscore], i["scores"])} for i in scoregroups]
    scoregroups = [i | {"mvp_c": int(100 * i["mvp"][0]), "mvp_p": int(100 * i["mvp"][1])} for i in scoregroups]
    df = pd.DataFrame(scoregroups)
    for n in ["mvr_c", "mvr_p", "mvp_c", "mvp_p"]:
        sns.barplot(x=df["name"], y=df[n])
        plt.savefig(f"{conf['figs_loc']}/score {n} bars.png")
        plt.close()
    wandb.finish()


if __name__ == "__main__":
    pred_instances = 0
    invalid_instances = 0
    valid_instances = 0
    total_instances = 0

    no_comparable_files = 0
    error_applying_patches = 0
    prepared_file_contents = 0
    total_prepared_file_contents = 0

    no_code_files = 0
    selected_code_files = 0
    different_files = 0
    total_code_files = 0
    parser = argparse.ArgumentParser(prog='Evaluate', description='Run evaluation pipeline')
    parser.add_argument("--model", help="model subpath to evaluate", default="20240402_sweagent_gpt4", type=str)
    args = parser.parse_args()
    conf["model"] = args.model
    main()
