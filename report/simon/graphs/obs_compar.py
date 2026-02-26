import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
df_weak = pd.read_csv("report/simon/data/sac_paper_obs_weak.csv")
df_weak["type"] = "Weak"

df_strong = pd.read_csv("report/simon/data/sac_paper_obs_strong.csv")
df_strong["type"] = "Strong"

# Combine and Prepare
df = pd.concat([df_weak, df_strong], ignore_index=True)


# Label
def create_label(row):
    base = row["type"]
    if row["normalize_obs"]:
        return f"{base}/norm"
    return base


df["legend_label"] = df.apply(create_label, axis=1)

# Binning
bin_size = 2000
df["_step"] = (df["_step"] // bin_size) * bin_size

# 3. Plot
plt.figure(figsize=(10, 6))
sns.set_theme(style="darkgrid")

sns.lineplot(
    data=df,
    x="_step",
    y="mean_eps_reward",
    hue="legend_label",
    estimator="mean",
    errorbar=("ci", 95),
)

fontsize = 22

plt.xlabel("Update Steps", fontsize=fontsize)
plt.ylabel("Reward", fontsize=fontsize)
plt.legend(title="Algorithm", fontsize=fontsize - 2, title_fontsize=fontsize)

plt.xticks(
    ticks=[0, 50000, 100000, 150000, 200000],
    labels=["0", "50k", "100k", "150k", "200k"],
    fontsize=fontsize,
)
plt.yticks(fontsize=fontsize)

plt.tight_layout()
plt.savefig("report/simon/plots/obs_norm_compar.pdf")
plt.show()
