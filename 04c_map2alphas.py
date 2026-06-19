# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from consts import *
import numpy as np
import json
from itertools import combinations
import networkx as nx

plt.rcParams.update({"font.size": 9})
plt.rcParams.update({"figure.figsize": (16 / 2.54, 9 / 2.54)})
sns.set_style("ticks")
sns.set_context("paper")

# %%
df = pd.read_csv("processed_data/2026-05-13_allBilendiData.csv")
df_p = pd.read_csv("processed_data/2026-05-19_data_processed_participant_withIssueWeights.csv")

# %%
def plot_map(ax, x, q, pos_processed, wave):
    colors = colorsOrig if q=="" else [x["player." + f"{'own2' if p=='self' else p}__{q}"].values[wave - 1] for p in peeps]
    for marker, inds in zip(["X","o"], [[0], list(range(1,len(pos_processed)))]):
        ax.scatter(
            [pos_processed[inds, 0]],
            [pos_processed[inds, 1]],
            c=[colors[i] for i in inds],
            s=10,
            cmap=plt.get_cmap("coolwarm"),
            vmin=-100,
            vmax=100,
            marker=marker,
        )
    ax.set_aspect("equal")
    ax.set_xlim(0, MAX_PIXELPOS)
    ax.set_ylim(0, MAX_PIXELPOS)
    ax.set_xticks([])
    ax.set_yticks([])


# %%
# np.random.seed(2)
fig, axs = plt.subplots(2, 4, figsize=(16 / 2.54, 7 / 2.54), sharex=True, sharey=True)
id = df.bilendi_id.sample().values[0]
wave = np.random.choice(df.loc[df.bilendi_id == id, "wave"].values)

id, wave = (331600757996607, 1)
k = "exponential"
alpha_cols = [f"{k}_alpha_{q}" for q in questions_sc]
alphas = df_p.query(f"id=={id} and wave=={wave}")[alpha_cols]
params = df_p.query(f"id=={id} and wave=={wave}")[[] if k=="corr" else [f"{k}_param{n}" for n in [1,2]]]
pos = json.loads(df.loc[df.bilendi_id == id, "player.positions"].values[wave - 1])

pos_processed = {p["varname"].replace(" ", ""): np.array([p["x"], p["y"]]) for p in pos}
pos_processed = np.array(
    [
        (
            [np.nan, np.nan]
            if k.replace(" ", "") not in pos_processed
            else pos_processed[k]
        )
        for k in peeps
    ]
)

for q, ax in zip( [""]+questions_sc, axs.flatten()[[0,1,2,3,5,6,7]]):
    plot_map(ax, df.loc[df.bilendi_id == id], q, pos_processed, wave - 1)
    ax.set_title(('SpAM' if q=="" else (labelMap[q] + "\n" fr"$\alpha_{k[0]}={alphas[f'{k}_alpha_{q}'].values[0]:.3f}$")))
axs[1, 0].axis("off")
fig.text(0.05, 0.4, f"id: {id};\nwave {wave}", fontsize=7, ha="left")
paramString =(
    f"correlation "+r"$\alpha_q = corr( d_{ij}, \Delta X_{ij,q})$"
    if k =="corr" else (
        f"a_n = {params.values[0,0]:.2f}; " + f"b_n = {params.values[0,1]:.2f}" +"\n"+r"$d_{ij} = a_n + b_n \cdot \sum_q \alpha_q \cdot \Delta X_{ij,q}$"
        if k=="linear" else 
        f"$a_n = {params.values[0,0]:.2f}$; " + f"$b_n = {params.values[0,1]:.2f}$"+"\n"+r"$d_{ij} = 1 - exp[ b_n \cdot (\sum_q \alpha_q \cdot \Delta X_{ij,q})^{a_n}]$"
        )
    )
fig.text(0.05, 0.1, paramString, fontsize=7, ha="left")
fig.tight_layout(h_pad=0.2)
print(id, wave)
plt.show()


#%% 
df_p.loc[df_p[[f"corr_alpha_{q}" for q in questions_sc]].isna().any(axis=1)]#.query("wave==2")
# %% [markdown]
# ## Nice examples
#
# - id, wave = (330717073703941, 2)
#

# %% [markdown]
#

# %%
