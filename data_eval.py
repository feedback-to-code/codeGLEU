import regex as re
import json
import unidiff
import tqdm
import copy
from pathlib import Path

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

preds = "./data/models/20240402_sweagent_gpt4/all_preds.jsonl"
preprocessed = "./data/models/20240402_sweagent_gpt4/preprocessed_instances.jsonl"

data = {}
default_data = {
    r"# instances": 0,
    r"# None instances": 0,
    r"% Code Files": [],
    r"# Files Edited": [],
    r"# Hunks Edited": [],
    r"# Lines Edited": [],
    r"# Lines Added": [],
    r"# Lines Removed": [],
    r"# Lines per File": [],
    r"% File Unmodified": [], 
}

with open(preds) as fp:
    for line in tqdm.tqdm(fp):
        instance = json.loads(line)
        iid = instance["instance_id"]
        rm = re.match(r"(.*?)__(.*?)-([0-9]+)", iid)
        repo = rm.group(2)
        if repo not in data:
            data[repo] = copy.deepcopy(default_data)
        data[repo]["# instances"] += 1
        if instance["model_patch"] is None or not instance["model_patch"]:
            data[repo]["# None instances"] += 1
            continue
        filesnum = 0
        codefiles = 0
        addednum = 0
        removednum = 0
        hunknum = 0
        for file in unidiff.PatchSet(instance["model_patch"]):
            filesnum += 1
            addednum += file.added
            removednum += file.removed
            if file.path.endswith("py"):
                codefiles += 1
            for hunk in file:
                hunknum += 1
        editednum = addednum + removednum
        data[repo]["# Files Edited"].append(filesnum)
        data[repo]["% Code Files"].append(codefiles / filesnum)
        data[repo]["# Lines Edited"].append(editednum)
        data[repo]["# Lines Added"].append(addednum)
        data[repo]["# Lines Removed"].append(removednum)
        data[repo]["# Hunks Edited"].append(hunknum)

with open(preprocessed) as fp:
    for line in tqdm.tqdm(fp):
        instance = json.loads(line)
        iid = instance["instance_id"]
        rm = re.match(r"(.*?)__(.*?)-([0-9]+)", iid)
        repo = rm.group(2)
        if repo not in data:
            data[repo] = copy.deepcopy(default_data)
        totallines = 0
        numfiles = 0
        patchfiles = 0
        editedlines = 0
        if instance["model_patch"] is None or not instance["model_patch"]:
            continue
        modifiedpercent = []
        for file in instance["source_files_content"]:
            totallines += len(instance["source_files_content"][file].splitlines())
            numfiles += 1
            for pfile in unidiff.PatchSet(instance["model_patch"]):
                if str(Path(file)).endswith(str(Path(pfile.path))):
                    patchfiles += 1
                    editedlines += (pfile.added + pfile.removed) / 2
                    if totallines == 0:
                        modifiedp = 1
                    else:
                        modifiedp = editedlines / totallines
                    modifiedpercent.append(modifiedp)
        if numfiles > 0:
            data[repo]["# Lines per File"].append(totallines / numfiles)
            data[repo][r"% File Unmodified"].append(1 - sum(modifiedpercent) / len(modifiedpercent))
total = default_data
for repo in data:
    for key in default_data:
        total[key] += data[repo][key]
data = {"total": total} | data
for repo in data:
    for key in default_data:
        if isinstance(data[repo][key], list):
            ls = data[repo][key]
            data[repo][key] = sum(ls)/len(ls)
        if isinstance(data[repo][key], float) and key.startswith("%"):
            data[repo][key] *= 100

# padlen = max([len(repo) for repo in data])
# entrylen = max([len(key) for key in default_data])
# print(" " * entrylen + " | " + " | ".join([pad(repo, padlen, "r") for repo in data]))
# for key in default_data:
#     print(pad(key, entrylen, "r") + " | "  + " | ".join([pad('%.2f' % data[repo][key] if isinstance(data[repo][key], float) else str(data[repo][key]), padlen, "r") for repo in data]))

repos = list(data.keys())
data1 = repos[:len(repos)//2 + 1]
data2 = repos[len(repos)//2 + 1:]

padlen = max([len(repo) for repo in data])
entrylen = max([len(key) for key in default_data])
print(" " * entrylen + " & " + " & ".join([pad(repo, padlen, "r") for repo in data1]) + "\\\\")
for key in default_data:
    print(pad(key, entrylen, "r") + " & "  + " & ".join([pad('%.2f' % data[repo][key] if isinstance(data[repo][key], float) else str(data[repo][key]), padlen, "r") for repo in data1]) + "\\\\")

entrylen = max([len(key) for key in default_data])
print(" " * entrylen + " & " + " & ".join([pad(repo, padlen, "r") for repo in data2]) + "\\\\")
for key in default_data:
    print(pad(key, entrylen, "r") + " & "  + " & ".join([pad('%.2f' % data[repo][key] if isinstance(data[repo][key], float) else str(data[repo][key]), padlen, "r") for repo in data2]) + "\\\\")
