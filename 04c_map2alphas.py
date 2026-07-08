# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from consts import *
import numpy as np
import json
from itertools import combinations
import networkx as nx
import matplotlib.colors as mcolors

plt.rcParams.update({"font.size": 9})
plt.rcParams.update({"figure.figsize": (16 / 2.54, 9 / 2.54)})
sns.set_style("ticks")
sns.set_context("paper")

# %%
df = pd.read_csv("processed_data/2026-05-13_allBilendiData.csv")
df_p = pd.read_csv("processed_data/2026-06-19_data_processed_participant_withAllIssueWeights.csv")

#%%
# (position 0-1, color) pairs — note the uneven spacing
nodes = [
    (0.00, "#40004b"),  # deep purple
    (0.15, "#762a83"),
    (0.35, "#af8dc3"),
    (0.46, "#e0d6e8"),  # light grey-purple
    (0.50, "#e6e6e6"),  # neutral grey at zero
    (0.54, "#d9ead3"),  # light grey-green
    (0.65, "#7fbf7b"),
    (0.85, "#1b7837"),
    (1.00, "#00441b"),  # deep green
]

cmap = mcolors.LinearSegmentedColormap.from_list("grey_diverge", nodes, N=256)

# %%
def plot_map(ax, x, q, pos_processed):
    colors = colorsOrig if q=="" else [x["player." + f"{'own2' if p=='self' else p}__{q}"].iloc[0] for p in peeps]
    for marker, inds in zip(["X","o"], [[0], list(range(1,len(pos_processed)))]):
        scatter = ax.scatter(
            [pos_processed[inds, 0]],
            [pos_processed[inds, 1]],
            c=[colors[i] for i in inds],
            s=18,
            cmap=cmap, #plt.get_cmap("viridis"),
            vmin=-100,
            vmax=100,
            marker=marker,
        )
    ax.set_aspect("equal")
    ax.set_xlim(0, MAX_PIXELPOS)
    ax.set_ylim(0, MAX_PIXELPOS)
    ax.set_xticks([])
    ax.set_yticks([])
    return scatter


# %%
# np.random.seed(2)
fig, axs = plt.subplots(2, 4, figsize=(16 / 2.54, 7 / 2.54), sharex=True, sharey=True)
id = df.bilendi_id.sample().values[0]
wave = np.random.choice(df.loc[df.bilendi_id == id, "wave"].values)

id, wave = (330717090224301, 2)
k = "corrS_alpha"
alpha_cols = [f"{k}_{q}" for q in questions_sc]
alphas = df_p.query(f"id=={id} and wave=={wave}")[alpha_cols]
if "exp" in k or "linear" in k: 
    params = df_p.query(f"id=={id} and wave=={wave}")[[] if k=="corr" else [f"{k}_param{n}" for n in [1,2]]]
pos = json.loads(df.loc[(df.wave==wave) & (df.bilendi_id == id), "player.positions"].iloc[0])

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
    plot_map(ax, df.loc[(df.bilendi_id == id) & (df.wave==wave)], q, pos_processed)
    ax.set_title(('SpAM' if q=="" else (labelMap[q] + "\n" fr"$\alpha_{k[0]}={alphas[f'{k}_{q}'].values[0]:.3f}$")))
axs[1, 0].axis("off")
fig.text(0.05, 0.4, f"id: {id};\nwave {wave}", fontsize=7, ha="left")
paramString =(
    f"correlation "+r"$\alpha_q = corr( d_{ij}, \Delta X_{ij,q})$"
    if "corr" in k else (
        f"a_n = {params.values[0,0]:.2f}; " + f"b_n = {params.values[0,1]:.2f}" +"\n"+r"$d_{ij} = a_n + b_n \cdot \sum_q \alpha_q \cdot \Delta X_{ij,q}$"
        if k=="linear" else 
        f"$a_n = {params.values[0,0]:.2f}$; " + f"$b_n = {params.values[0,1]:.2f}$"+"\n"+r"$d_{ij} = 1 - exp[ b_n \cdot (\sum_q \alpha_q \cdot \Delta X_{ij,q})^{a_n}]$"
        )
    )
# plt.colorbar(axs[-1,-1], ax=axs)
fig.text(0.05, 0.1, paramString, fontsize=7, ha="left")
fig.tight_layout(h_pad=0.2)
print(id, wave)
plt.show()


# %%
for q in [""]+questions_sc:

    fig, ax = plt.subplots(1,1, figsize=(2,2))
    id, wave = (330717090224301, 2)
    k = "corrP_alpha"
    alpha_cols = [f"{k}_{q}" for q in questions_sc]
    alphas = df_p.query(f"id=={id} and wave=={wave}")[alpha_cols]
    if "exp" in k or "linear" in k: 
        params = df_p.query(f"id=={id} and wave=={wave}")[[] if k=="corr" else [f"{k}_param{n}" for n in [1,2]]]
    pos = json.loads(df.loc[(df.wave==wave) & (df.bilendi_id == id), "player.positions"].iloc[0])

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
    scatter = plot_map(ax, df.loc[(df.bilendi_id == id) & (df.wave==wave)], q, pos_processed)
    ax.set_title(('\nspatial arrangement' if q=="" else (labelMap[q] + "\n" fr"$\alpha_{k[0]}={alphas[f'{k}_{q}'].values[0]:.3f}$")))
    ax.text(0.5, 0.01, f"id: {id}; wave: {wave}", fontsize=5, ha="center", transform=ax.transAxes)
    if q in questions_sc:
        cbar = plt.colorbar(
            scatter,
            ax=ax,
            shrink=0.8,      # scales the length of the colorbar
            fraction=0.086,  # width of colorbar as fraction of ax
            pad=0.04,        # spacing between ax and colorbar
            aspect=20,       # length-to-width ratio (higher = thinner)
        )
        cbar.set_ticks([-100, 0, 100])
        cbar.set_ticklabels(['-1', '0', '+1'])
    fig.tight_layout()
    plt.savefig(f"figs/maps/{id}-{wave}_{'orig' if q=="" else q}.png", dpi=300)

    fig, ax = plt.subplots(1,1, figsize=(1,1))
    id, wave = (330717090224301, 2)
    k = "corrP_alpha"
    alpha_cols = [f"{k}_{q}" for q in questions_sc]
    alphas = df_p.query(f"id=={id} and wave=={wave}")[alpha_cols]
    if "exp" in k or "linear" in k: 
        params = df_p.query(f"id=={id} and wave=={wave}")[[] if k=="corr" else [f"{k}_param{n}" for n in [1,2]]]
    pos = json.loads(df.loc[(df.wave==wave) & (df.bilendi_id == id), "player.positions"].iloc[0])

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
    scatter = plot_map(ax, df.loc[(df.bilendi_id == id) & (df.wave==wave)], q, pos_processed)
    fig.tight_layout()
    plt.savefig(f"figs/maps/{id}-{wave}_{'orig' if q=="" else q}_focus.png", dpi=300)

# %%
