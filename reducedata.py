import json

with open("data/models/20240820_honeycomb/results/results.json", "r") as fp:
    results = json.load(fp)

resolved = set(results["resolved"])

with open("data/models/20240820_honeycomb/all_preds.jsonl", "r") as fp:
    with open("data/models/20240820_honeycomb/resolved_preds.jsonl", "w+") as output:
        for line in fp:
            instance = json.loads(line)
            if instance["instance_id"] in resolved:
                output.write(json.dumps(instance) + "\n")