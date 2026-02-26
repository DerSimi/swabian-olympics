import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

run_name = "sac_20m_selfplay"

df = pd.read_csv(f"report/simon/data/{run_name}.csv")

bin_size = 2000
df["_step"] = (df["_step"] // bin_size) * bin_size

plt.figure(figsize=(10, 6))
sns.set_theme(style="darkgrid")

sns.lineplot(
    data=df,
    x="_step",
    y="mean_eps_reward",
)

fontsize = 22

# Layout
plt.xlabel("Update Steps", fontsize=fontsize)
plt.ylabel("Reward", fontsize=fontsize)
plt.title("")

plt.xticks(
    ticks=[0, 5000000, 10000000, 15000000, 20000000],
    labels=["0", "5M", "10M", "15M", "20M"],
    fontsize=fontsize,
)
plt.yticks(fontsize=fontsize)

plt.axvline(x=16600000, color='red', linestyle='--')
plt.text(
    16600000, 
    0.5, 
    "Used Checkpoint at 16.6M", 
    rotation=90, 
    color='red', 
    fontsize=18, 
    va='center', 
    ha='right', 
    transform=plt.gca().get_xaxis_transform()
)

plt.tight_layout()

plt.savefig(f"report/simon/plots/{run_name}.pdf")
plt.show()
