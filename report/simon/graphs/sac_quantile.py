import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df_quantile = pd.read_csv("report/simon/data/sac_paper_quantile.csv")
df_quantile["algo"] = "Quantile"

df_sac = pd.read_csv("report/simon/data/sac_paper_noise_strong.csv")
df_sac["algo"] = "SAC"

# 2. Combine and Prepare
df = pd.concat([df_quantile, df_sac], ignore_index=True)

# Create a readable label for the legend (e.g., "SAC (Pink)" or "Quantile (Gaussian)")
noise_map = {True: "Pink", False: "Gaussian"}
df["label"] = df.apply(
    lambda row: f"{row['algo']} ({noise_map[row['pink_noise']]})", axis=1
)

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
    hue="label",
    estimator="mean",
    errorbar=("ci", 95),
)

plt.xlabel("Update Steps")
plt.ylabel("Reward")
plt.legend(title="Algorithm (Noise)")

plt.xticks(
    ticks=[0, 50000, 100000, 150000, 200000],
    labels=["0", "50k", "100k", "150k", "200k"],
)

plt.tight_layout()
plt.savefig("report/simon/plots/combined_sac_quantile.pdf")
plt.show()
