import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

run_name = "sac_paper_noise_strong"

df = pd.read_csv(f"report/simon/data/{run_name}.csv")

print(df.head())

groups = df["pink_noise"].unique()
print("All groups:")
print(groups)

for g in groups:
    group = df[df["pink_noise"] == g]

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
    hue="pink_noise",
    estimator="mean",
    errorbar=("ci", 95),
)

# Layout
plt.xlabel("Update Steps")
plt.ylabel("Reward")
plt.title("")

handles, labels = plt.gca().get_legend_handles_labels()
label_map = {"False": "Gaussian", "True": "Pink"}
labels = [label_map.get(l, l) for l in labels]
plt.legend(handles, labels)

# Ticks
plt.xticks(
    ticks=[0, 50000, 100000, 150000, 200000],
    labels=["0", "50k", "100k", "150k", "200k"],
)

plt.tight_layout()

plt.savefig(f"report/simon/plots/{run_name}.pdf")
plt.show()
