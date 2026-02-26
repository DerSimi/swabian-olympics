import wandb
import pandas as pd

api = wandb.Api()

entity = "dersimi-t-bingen"
project = "sac_20m_selfplay"

runs = api.runs(f"{entity}/{project}")

all_runs_data = []
target_metric = "mean_eps_reward"

print(f"Found runs: {len(runs)}")

for run in runs:
    if run.state != "finished":
        continue

    print("Fetching", run.name)

    config = {k: v for k, v in run.config.items() if not k.startswith("_")}

    history_df = run.history(keys=["_step", target_metric], samples=100_000_000)

    if target_metric not in history_df.columns:
        continue

    history_df["run_name"] = run.name

    for param_name, param_value in config.items():
        history_df[param_name] = param_value

    all_runs_data.append(history_df)

full_df = pd.concat(all_runs_data)

full_df.to_csv(f"report/simon/data/{project}.csv", index=False)
