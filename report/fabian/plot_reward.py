import os
import sys
from gc import enable

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, os.path.abspath(".."))

from plot_data import get_data, smooth

df = get_data()
df, _, _ = smooth(df)

# Convert from wide → long format
df_long = df.melt(
    id_vars="step",
    value_vars=["reward"],
    var_name="Loss",
    value_name="Value",
)

sns.set_theme(style="darkgrid")

plt.figure(figsize=(9, 5))

sns.lineplot(
    data=df_long,
    x="step",
    y="Value",
    hue="Loss",
    linewidth=1.0,
    alpha=0.85,
)

plt.ylim(-20, 12)
plt.legend().remove()

plt.xlabel("Steps")
plt.ylabel("Reward")
plt.tight_layout()

plt.gca().xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k")
)

plt.savefig("checkpoints/default.mbpo_sac/plot_reward.png")
plt.show()
