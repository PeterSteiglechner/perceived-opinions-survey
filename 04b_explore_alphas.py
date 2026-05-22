#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from consts import *
import numpy as np
from itertools import combinations

plt.rcParams.update({"font.size":9})
plt.rcParams.update({"figure.figsize":(16/2.54, 9/2.54)})
sns.set_style("ticks")
sns.set_context("paper")

#%%
linear=lambda d, a, b:  a + b* d
s=lambda d, a, b:  1 -np.exp(-a * d**b)
sig=lambda d, a,b, c: a / (1 + np.exp(-b * (d-c)))
l1norm = lambda x: np.sum(np.abs(x[f"deltaX_"]))
# %%
df_p = pd.read_csv("processed_data/2026-05-19_data_processed_participant_withIssueWeights.csv")
df_diff = pd.read_csv("processed_data/2026-05-19_data_processed_differences_withIssueWeights.csv")
# %%
kernel = "exponential"
func=s if kernel=="exponential" else linear
wave = 1
fig, axs = plt.subplot_mosaic([["a", "b"], ["curve", "curve"]], height_ratios=[1,2], figsize=(16/2.54, 10/2.54))
for ax, col in zip([axs["a"], axs["b"]], [f"{kernel}_param1", f"{kernel}_param1"]): 
    sns.histplot(df_p, x=col, hue="wave", ax=ax)
sampleids = df_p.sample(100)["id"].values
xx = np.linspace(0,1)
for n,x in df_p.loc[(df_p["id"].isin(sampleids)) & (df_p['wave']==wave)].iterrows():
    params = (x[f"{kernel}_param1"], x[f"{kernel}_param2"] )
    axs["curve"].plot(xx, func(xx, params[0], params[1]), lw=0.5, alpha=0.4)
    axs["curve"].set_ylim(0,1)
    axs["curve"].set_xlim(0,1)

#%%
kernel= "linear"
fig, ax = plt.subplots(1,1)
sampleids = df_p.loc[df_p.wave==1].sample(10)["id"].values
for id in sampleids:
    current_data = df_diff.loc[(df_diff.wave==wave) & (df_diff['id']==id)]
    current_partic_data = df_p.loc[(df_p.wave==wave) & (df_p['id']==id)]
    alphas = current_partic_data[[f"{kernel}_alpha_{q}" for q in questions_sc]].values.reshape(1,6)
    deltas = current_data[[f'deltaX_{q}' for q in questions_sc]].values.T
    d =np.dot(alphas, deltas)[0].astype(float)
    predicted = func(d, params[0], params[1])
    pixel_d =  current_data["pixel_dist"]
    predicted = pd.DataFrame({"predicted":predicted, "true":pixel_d})
    sns.scatterplot(predicted, x="true", y="predicted", s=3)
    ax.plot([0,1], [0,1], "--", color="grey") 
    ax.set_aspect("equal")


#%%
df_p["best_kernel"].value_counts()

#%%
k="exponential"
alpha_cols = [f"{k}_alpha_{q}" for q in questions_sc]
fig, ax = plt.subplots(1,1)
sns.barplot(df_p[alpha_cols+["wave"]].melt(id_vars="wave", ).reset_index().replace(dict(zip(alpha_cols, [q.replace("_", "\n") for q in questions_sc]))), x="variable", y="value", hue="wave", palette="Set1")
fig.autofmt_xdate(rotation=20, ha="center")
ax.set_xlabel("")
ax.hlines(1/6, ax.get_xlim()[0], ax.get_xlim()[1], linestyles="--", colors="grey")
ax.set_ylabel(f"weight ({k})")
#%%
k="exponential"
alpha_cols = [f"{k}_alpha_{q}" for q in questions_sc]
fig, ax = plt.subplots(1,1)
ppp = parties + ["No party"]
sns.barplot(df_p[alpha_cols+["party_close"]].melt(id_vars="party_close", ).reset_index().replace(dict(zip(alpha_cols, [q.replace("_", "\n") for q in questions_sc]))), x="variable", y="value", hue="party_close", hue_order=ppp, palette=party_cmap, errwidth=0.4)
fig.autofmt_xdate(rotation=20, ha="center")
ax.set_xlabel("")
ax.hlines(1/6, ax.get_xlim()[0], ax.get_xlim()[1], linestyles="--", colors="grey")
ax.set_ylabel(f"weight ({k} kernel)")
plt.legend(ncols=4)
# %%
df_p.loc[df_p["id"].isin(sampleids)]
# %%
df_p.loc[df_diff['id']==x['id']]
# %%

kernel = "exponential"
fig, axs = plt.subplots(2,3, sharex=True, sharey=True)
for ax, q in zip(axs.flatten(), questions_sc):
    sns.histplot(df_p, x=f"{kernel}_alpha_{q}", hue="party_close", hue_order=parties_full, palette=party_cmap, ax=ax, legend=False, kde=True, stat="percent", common_norm=False)
    ax.set_title(q)
    ax.set_xlabel("")
# %%
k = "linear"
alphas = df_p[alpha_cols]

from sklearn.cluster import KMeans 

m = KMeans(4)
df_p["label"] = m.fit_predict(alphas.values)

pd.DataFrame(m.cluster_centers_, columns=questions_sc).plot.bar()

# %%
df_p["party_close_cat"] = pd.Categorical(df_p["party_close"], categories=parties_full)
sns.heatmap(df_p[["label", "party_close"]].value_counts().reset_index().pivot_table(index="label", columns="party_close", values="count"), cmap="viridis")

# %%
kernel= "linear"

sns.histplot(df_p, x= f"{kernel}_alpha_climate_concern", y = f"{kernel}_alpha_rights_indep_integration")
# %%
df_p["maxAlpha"] = df_p[alpha_cols].max(axis=1)
sns.histplot(df_p, x="maxAlpha", hue="party_close", palette=party_cmap, kde=True, bins=np.linspace(0,1), stat="proportion", common_norm=False,)
# %%



kernel = "linear"
fig, axs = plt.subplots(4,3, sharex=True, sharey=False, figsize=(16/2.54, 10/2.54), gridspec_kw={"height_ratios":[0.5,1,0.5,1]})
for ax, q in zip(axs[[1,3],:].flatten(), questions_sc):
    sns.stripplot(df_p.groupby("party_close")[f"{kernel}_alpha_{q}"].mean().reset_index(),  x=f"{kernel}_alpha_{q}", hue="party_close", hue_order=parties_full, palette=party_cmap, ax=ax, legend=False, dodge=True, size=4, marker="s")
    sns.stripplot(df_p, x=f"{kernel}_alpha_{q}", hue="party_close", hue_order=parties_full, palette=party_cmap, ax=ax, legend=False, dodge=True, size=1)
    ax.set_xlabel("")
for ax, q in zip(axs[[0,2],:].flatten(), questions_sc):
    ax.grid("x")
    # sns.boxplot(df_p, x=f"{kernel}_alpha_{q}", hue="party_close", hue_order=parties_full, palette=party_cmap, ax=ax,
    #              fill=True, width=0.5, legend=False, fliersize=0)
    sns.kdeplot(df_p, x=f"{kernel}_alpha_{q}", hue="party_close", hue_order=parties_full, palette=party_cmap, ax=ax,
                 fill=False, cut=0, legend=False, common_norm=False)
    ax.set_xlabel("")
    ax.set_ylim(0,)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_title(q)
ax.set_xlim(0.0,0.5)

# %%
