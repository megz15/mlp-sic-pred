import json, os

def calc_model_score(rmse, mae, corr):
    return (1 / rmse) + (1 / mae) + (corr * 10)

dir = "multi_metadata"

model_names = [v.split("_meta.json")[0] for v in os.listdir(dir)]
model_metadata = {}

for model in model_names:
    with open(f"{dir}/{model}_meta.json", "r") as f:
        model_metadata[model] = json.load(f)

for model in model_metadata:
    rmse = model_metadata[model]["rmse"]
    mae = model_metadata[model]["mae"]
    corr = model_metadata[model]["corr"]
    
    score = calc_model_score(rmse, mae, corr)
    model_metadata[model]["score"] = score

print(json.dumps(model_metadata, indent=4))