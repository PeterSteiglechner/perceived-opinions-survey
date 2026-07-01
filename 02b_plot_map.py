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
colorsOrig = [
    (
        party_cmap["contact"]
        if "reference" in p
        else (
            party_cmap[p[5:].replace(" ", "")]
            if "voter" in p
            else party_cmap[p.replace(" ", "")]
        )
    )
    for p in peeps
]

# %%
df = pd.read_csv("processed_data/2026-05-13_allBilendiData.csv")


# %%


# %%
p = peeps[2]
df.loc[df.bilendi_id == id, ["player." + f"{'own2' if p=='self' else p}__{q}", "wave"]]


# %%
def plot_map(ax, x, q, pos_processed, wave):
    colors = (
        colorsOrig
        if q == ""
        else [
            x["player." + f"{'own2' if p=='self' else p}__{q}"].values[wave - 1]
            for p in peeps
        ]
    )
    if q == "":
        ax.scatter(
            [pos_processed[0, 0]], [pos_processed[0, 1]], c=colors[0], s=10, marker="X"
        )
        ax.scatter(
            pos_processed[1:, 0],
            pos_processed[1:, 1],
            c=colors[1:],
            s=10,
        )
    else:
        ax.scatter(
            [pos_processed[0, 0]],
            [pos_processed[0, 1]],
            c=colors[0],
            s=10,
            cmap=plt.get_cmap("PRGn"),
            vmin=-100,
            vmax=100,
            marker="X",
        )
        ax.scatter(
            pos_processed[1:, 0],
            pos_processed[1:, 1],
            c=colors[1:],
            s=10,
            cmap=plt.get_cmap("PRGn"),
            vmin=-100,
            vmax=100,
        )
    ax.set_aspect("equal")
    ax.set_xlim(0, MAX_PIXELPOS)
    ax.set_ylim(0, MAX_PIXELPOS)
    ax.set_xticks([])
    ax.set_yticks([])


# %%

fig, axs = plt.subplots(3, 3, figsize=(12 / 2.54, 12 / 2.54), sharex=True, sharey=True)

for id, ax in zip(df.bilendi_id.sample(len(axs.flatten())).values, axs.flatten()):

    wave = np.random.choice(df.loc[df.bilendi_id == id, "wave"].values)
    pos = json.loads(df.loc[df.bilendi_id == id, "player.positions"].values[wave - 1])
    pos_processed = {
        p["varname"].replace(" ", ""): np.array([p["x"], p["y"]]) for p in pos
    }
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
    # pos_processed_dict = {k: [np.nan, np.nan] if k not in pos_processed else pos_processed[k] for k in peeps}

    plot_map(ax, df.loc[df.bilendi_id == id], "", pos_processed, np.nan)
    ax.text(0.0, 1.02, f"id: {id} (w{wave})", transform=ax.transAxes, fontsize=5)
fig.tight_layout(h_pad=1, w_pad=1)
plt.show()

# %%
# np.random.seed(2)
fig, axs = plt.subplots(2, 4, figsize=(12 / 2.54, 7 / 2.54), sharex=True, sharey=True)
id = df.bilendi_id.sample().values[0]
wave = np.random.choice(df.loc[df.bilendi_id == id, "wave"].values)

# id, wave = (330183866033359, 1)
# id, wave = (331246904848564, 1)
id, wave = (331600764215531 1)
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

for q, ax in zip([""] + questions_sc, axs.flatten()[[0, 1, 2, 3, 5, 6, 7]]):
    plot_map(ax, df.loc[df.bilendi_id == id], q, pos_processed, wave - 1)
    ax.set_title("original map" if q == "" else dict(zip(questions_sc, qs_keys))[q])
axs[1, 0].axis("off")
fig.text(0.05, 0.39, f"id: {id}\nwave {wave}", fontsize=7, ha="left")
fig.tight_layout(h_pad=0.2)
print(id, wave)
plt.show()

# %% [markdown]
# ## Nice examples
#
# - id, wave = (330717073703941, 2)
#

# %% [markdown]
#
