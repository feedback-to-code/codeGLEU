import json

entries = []
with open("results/rankingscores.txt") as fp:
    entries = [json.loads(line) for line in fp]

leaderboard = {
    "lite/20240402_sweagent_gpt4": 18.00,
    "lite/20240630_agentless_gpt4o": 27.33,
    "lite/20240615_appmap-navie_gpt4o": 21.67,
    "lite/20240617_moatless_gpt4o": 24.67,
    "lite/20240621_autocoderover-v20240620": 30.67,
    "lite/20240808_RepoGraph_gpt4o": 29.67,
    "lite/20231010_rag_claude2": 3.00,
    "lite/20231010_rag_gpt35": 0.33,
    "lite/20231010_rag_swellama13b": 1.00,
    "lite/20231010_rag_swellama7b": 1.33,
    "lite/20240402_rag_claude3opus": 4.33,
    "lite/20240402_rag_gpt4": 2.67,
    "lite/20240402_sweagent_claude3opus": 11.67,
    "lite/20240523_aider": 26.33,
    "lite/20240725_opendevin_codeact_v1.8_claude35sonnet": 26.67,
    "lite/20240728_sweagent_gpt4o": 18.33,
}

names = {
    "lite/20240402_sweagent_gpt4": "SWE-agent + GPT4",
    "lite/20240630_agentless_gpt4o": "Agentless + GPT4o",
    "lite/20240615_appmap-navie_gpt4o": "AppMap Navie + GPT4o",
    "lite/20240617_moatless_gpt4o": "Moatless Tools + GPT4o",
    "lite/20240621_autocoderover-v20240620": "AutoCodeRover + GPT4",
    "lite/20240808_RepoGraph_gpt4o": "Agentless + Repograph",
    "lite/20231010_rag_claude2": "RAG + Claude2",
    "lite/20231010_rag_gpt35": "RAG + GPT3.5",
    "lite/20231010_rag_swellama13b": "RAG + SWE-Llama13B",
    "lite/20231010_rag_swellama7b": "RAG + SWE-Llama7B",
    "lite/20240402_rag_claude3opus": "RAG + Claude3",
    "lite/20240402_rag_gpt4": "RAG + GPT4",
    "lite/20240402_sweagent_claude3opus": "SWA-agent + Claude3.5",
    "lite/20240523_aider": "Aider + GPT4o",
    "lite/20240725_opendevin_codeact_v1.8_claude35sonnet": "OpenDevin + CodeAct",
    "lite/20240728_sweagent_gpt4o": "SWE-agent + GPT4o",
}

scores = list(set(entries[0].keys()) - set(["model"]))
rankings = {}
for score in scores:
    rankings[score] = sorted([(names[entry["model"]], entry[score] * 100) for entry in entries], reverse=True, key=lambda x: x[1])

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

pprint = lambda x: x[0].replace("&", "\\&") + "&(" + '{:.2f}'.format(x[1]) + ")" 
torank = ["codegleu", "codebleu_p", "diffsim"]
print("&".join(["\\multicolumn{2}{c}{" + ranking + "}" for ranking in torank]) + "\\\\")
for i in range(0, len(rankings["actual"])):
    print("&".join([pad(pprint(rankings[ranking][i]), 1, "r") for ranking in torank]) + "\\\\")