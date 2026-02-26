import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

run_name = "sac_paper_net_strong"

df = pd.read_csv(f"report/simon/data/{run_name}.csv")

df["config"] = df.apply(
    lambda row: "-".join([str(row["network_width"])] * int(row["hidden_depth"])), axis=1
)

print(df.head())

groups = df["config"].unique()
print("All groups:")
print(groups)

for g in groups:
    group = df[df["config"] == g]

    print(
        "Group",
        g,
        "with:",
        group["run_name"].unique(),
    )

bin_size = 2000
df["_step"] = (df["_step"] // bin_size) * bin_size

plt.figure(figsize=(10, 6))
sns.set_theme(style="darkgrid")

sns.lineplot(
    data=df,
    x="_step",
    y="mean_eps_reward",
    hue="config",
    estimator="mean",
    errorbar=("ci", 95),
)

fontsize = 22

# Layout
plt.xlabel("Update Steps", fontsize=fontsize)
plt.ylabel("Reward", fontsize=fontsize)
plt.title("")
plt.legend(title="Hidden Layers", fontsize=fontsize - 2, title_fontsize=fontsize)

plt.xticks(
    ticks=[0, 50000, 100000, 150000, 200000],
    labels=["0", "50k", "100k", "150k", "200k"],
    fontsize=fontsize,
)
plt.yticks(fontsize=fontsize)

plt.tight_layout()

plt.savefig(f"report/simon/plots/{run_name}.pdf")
plt.show()
