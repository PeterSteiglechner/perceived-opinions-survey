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

LRcuts = [0,0.33,0.67,1.]

# %%
df_p = pd.read_csv("processed_data/2026-06-19_data_processed_participant_withAllIssueWeights_justParties.csv")
df_diff = pd.read_csv("processed_data/2026-06-19_data_processed_differences_withAllIssueWeights_justParties.csv")
# %%
for k_func in ["exp", "linear"]:
    func=s if k_func=="exp" else linear
    wave = 1
    fig, axs = plt.subplot_mosaic([["a", "b"], ["curve", "curve"]], height_ratios=[1,2], figsize=(16/2.54, 10/2.54))
    for ax, col in zip([axs["a"], axs["b"]], [f"{k_func}_param1", f"{k_func}_param2"]): 
        sns.histplot(df_p, x=col, hue="wave", ax=ax)
    sampleids = df_p.sample(100)["id"].values
    xx = np.linspace(0,1)
    for n,x in df_p.loc[(df_p["id"].isin(sampleids)) & (df_p['wave']==wave)].iterrows():
        params = (x[f"{k_func}_param1"], x[f"{k_func}_param2"] )
        axs["curve"].plot(xx, func(xx, params[0], params[1]), lw=0.5, alpha=0.4)
        axs["curve"].set_ylim(0,1)
        axs["curve"].set_xlim(0,1)

#%%
# k= "linear"
# fig, ax = plt.subplots(1,1)
# sampleids = df_p.loc[df_p.wave==1].sample(10)["id"].values
# for id in sampleids:
#     current_data = df_diff.loc[(df_diff.wave==wave) & (df_diff['id']==id)]
#     current_partic_data = df_p.loc[(df_p.wave==wave) & (df_p['id']==id)]
#     alphas = current_partic_data[[f"{k}_alpha_{q}" for q in questions_sc]].values.reshape(1,6)
#     deltas = current_data[[f'deltaX_{q}' for q in questions_sc]].values.T
#     d =np.dot(alphas, deltas)[0].astype(float)
#     predicted = func(d, params[0], params[1])
#     pixel_d =  current_data["pixel_dist"]
#     predicted = pd.DataFrame({"predicted":predicted, "true":pixel_d})
#     sns.scatterplot(predicted, x="true", y="predicted", s=3)
#     ax.plot([0,1], [0,1], "--", color="grey") 
#     ax.set_aspect("equal")


#%%
df_p["best_kernel"].value_counts()

#%%
k="corrS"
# --------- WAVE 1 vs 2 --------------
alpha_cols = [f"{k}_alpha_{q}" for q in questions_sc]
fig, ax = plt.subplots(1,1, figsize=(18/2.54, 9/2.54))
sns.barplot(df_p[alpha_cols+["wave"]].melt(id_vars="wave", ).reset_index().replace(dict(zip(alpha_cols, [q.replace("_", "\n") for q in questions_sc]))), x="variable", y="value", hue="wave", palette="Set1", estimator='mean')
fig.autofmt_xdate(rotation=20, ha="center")
ax.set_xlabel("")
ax.hlines(1/6, ax.get_xlim()[0], ax.get_xlim()[1], linestyles="--", colors="grey")
ax.set_ylabel(f"weight ({k})")
#%%
# --------- By Party --------------
alpha_cols = [f"{k}_alpha_{q}" for q in questions_sc]
fig, ax = plt.subplots(1,1, figsize=(18/2.54, 9/2.54))
ppp = parties + ["No party"]
waves = [1]
aa = df_p.loc[df_p.wave.isin(waves), alpha_cols+["party_close"]].melt(id_vars="party_close", ).reset_index().replace(dict(zip(alpha_cols, questions_sc)))
aa["value"] = aa["value"]
sns.barplot(aa, x="variable", y="value", hue="party_close", hue_order=ppp, palette=party_cmap, err_kws={'linewidth': 0.6}, alpha=0.8, estimator='mean', errorbar=('ci', 95),)
# sns.stripplot(aa, x="variable", y="value", hue="party_close", hue_order=ppp, palette=party_cmap, alpha=0.8, size=2, dodge=True, legend=False)
ax.set_xticklabels([labelMap_nl[l.get_text()] for l in ax.get_xticklabels()])
fig.autofmt_xdate(rotation=20, ha="center")
ax.set_xlabel("")
ax.hlines(1/6, -0.5, len(questions_sc)-0.5, linestyles="--", colors="grey")
ax.set_ylabel(f"issue weight \n({k} kernel, {f'wave {waves[0]}' if len(waves)==1 else 'both waves'})")
handles, labels = ax.get_legend_handles_labels()
c = df_p.loc[df_p.wave.isin(waves), ["party_close"]].value_counts()
labels = [f'{l} ($n={c[l]}$)' for l in labels]
ax.legend(handles, labels, ncols=1, handlelength=2, columnspacing=0.5,  frameon=False, bbox_to_anchor=(1.01,1))
ax.set_ylim(0,0.32 if not "corr" in k else 0.5)
ax.set_xlim(-0.5,len(questions_sc)-0.5)
plt.savefig("issue_weights_by_party.png", dpi=600)
#%%
# --------- By Party Dist --------------
alpha_cols = [f"{k}_alpha_{q}" for q in questions_sc]
fig, ax = plt.subplots(1,1, figsize=(18/2.54, 9/2.54))
ppp = parties + ["No party"]
waves = [1,2]
aa = df_p.loc[df_p.wave.isin(waves), alpha_cols+["party_close"]].melt(id_vars="party_close", ).reset_index().replace(dict(zip(alpha_cols, questions_sc)), )
aa["value"] = aa["value"]
sns.boxplot(aa, x="variable", y="value", hue="party_close", hue_order=ppp, palette=party_cmap, fliersize=0, showcaps=False,  medianprops={"linewidth": 3, "color":"k"}, notch=True, saturation=0.3)
sns.stripplot(aa, x="variable", y="value", hue="party_close", hue_order=ppp, palette=party_cmap, alpha=0.8, size=2, dodge=True, legend=False)
fig.autofmt_xdate(rotation=20, ha="center")
ax.set_xlabel("")
ax.hlines(1/6, -0.5, len(questions_sc)-0.5, linestyles="--", colors="grey")
ax.set_ylabel(f"issue weight \n({k} kernel)")
handles, labels = ax.get_legend_handles_labels()
c = df_p.loc[df_p.wave.isin(waves), ["party_close"]].value_counts()
labels = [f'{l}\n$n={c[l]}$' for l in labels]
ax.legend(handles, labels, ncols=1, handlelength=2, columnspacing=0.5,  frameon=False, bbox_to_anchor=(1.01,1))
ax.set_ylim(0,0.32 if not "corr" in k else 0.9)
ax.set_xlim(-0.5,len(questions_sc)-0.5)

#%%
# --------- By LR --------------
cmapLR = {'left':'#d8b365', 'moderate':"#A1A1A1", 'right':'#5ab4ac'}
alpha_cols = [f"{k}_alpha_{q}" for q in questions_sc]
fig, ax = plt.subplots(1,1, figsize=(18/2.54, 9/2.54))
waves = [1,2]
df_p["lr_label"]= pd.cut(df_p.lr, bins=LRcuts, labels=cmapLR.keys())
aa = df_p.loc[df_p.wave.isin(waves), alpha_cols+["lr_label"]].melt(id_vars="lr_label", ).reset_index().replace(dict(zip(alpha_cols, questions_sc)))
aa["value"] = aa["value"]
sns.barplot(aa, x="variable", y="value", hue="lr_label", hue_order=cmapLR.keys(), palette=cmapLR, err_kws={'linewidth': 0.6}, alpha=0.8, estimator='mean', errorbar=('ci', 95),)
# sns.stripplot(aa, x="variable", y="value", hue="party_close", hue_order=ppp, palette=party_cmap, alpha=0.8, size=2, dodge=True, legend=False)
ax.set_xticklabels([labelMap_nl[l.get_text()] for l in ax.get_xticklabels()])
fig.autofmt_xdate(rotation=20, ha="center")
ax.set_xlabel("")
ax.hlines(1/6, -0.5, len(questions_sc)-0.5, linestyles="--", colors="grey")
ax.set_ylabel(f"issue weight \n({k} kernel, {f'wave {waves[0]}' if len(waves)==1 else 'both waves'})")
handles, labels = ax.get_legend_handles_labels()
c = df_p.loc[df_p.wave.isin(waves), ["lr_label"]].value_counts()
labels = [f'{l} ($n={c[l]}$)' for l in labels]
ax.legend(handles, labels, ncols=3, handlelength=2, columnspacing=0.5,  frameon=False)
ax.set_ylim(0,0.32 if not "corr" in k else 0.5)
ax.set_xlim(-0.5,len(questions_sc)-0.5)
plt.savefig("issue_weights_by_lr.png", dpi=600)
# %%


# %%
sns.histplot(df_p, x= f"{k}_alpha_climate_concern", y = f"{k}_alpha_rights_indep_integration")
plt.figure()
sns.histplot(df_p, x= f"{k}_alpha_east_germans", y = f"{k}_alpha_regulate_internet")
# %%
df_p["maxAlpha"] = df_p[alpha_cols].max(axis=1)
plt.figure()
sns.histplot(df_p, x="maxAlpha", hue="party_close", palette=party_cmap, kde=True, bins=np.linspace(0,1), stat="proportion", common_norm=False, kde_kws={"cut":0})
# %%

fig, axs = plt.subplots(5,3, sharex="row", sharey=False, figsize=(16/2.54, 11/2.54), gridspec_kw={"height_ratios":[0.7,1,0.2,0.7,1]})
for ax in axs[2,:]:
    ax.axis("off")
for ax, q in zip(axs[[1,4],:].flatten(), questions_sc):
    ax.grid("x")
    sns.stripplot(df_p.groupby("party_close")[f"{k}_alpha_{q}"].mean().reset_index(),  x=f"{k}_alpha_{q}", hue="party_close", hue_order=parties_full, palette=party_cmap, ax=ax, legend=False, dodge=True, size=4, marker="s")
    sns.stripplot(df_p, x=f"{k}_alpha_{q}", hue="party_close", hue_order=parties_full, palette=party_cmap, ax=ax, legend=False, dodge=True, size=1)
    ax.set_xlabel(q)
for ax, q in zip(axs[[0,3],:].flatten(), questions_sc):
    ax.grid("x")
    # sns.boxplot(df_p, x=f"{k}_alpha_{q}", hue="party_close", hue_order=parties_full, palette=party_cmap, ax=ax,
    #              fill=True, width=0.5, legend=False, fliersize=0)
    sns.kdeplot(df_p, x=f"{k}_alpha_{q}", hue="party_close", hue_order=parties_full, palette=party_cmap, ax=ax,
                 fill=False, cut=0, legend=False, common_norm=False)
    ax.set_title("")
    ax.set_ylim(0,)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("")
    #ax.set_xlim(0.0,0.5 if not "corr" in k else 1.0)
fig.suptitle(f"weights: {k}")
# %%


# %%

alpha_cols = [f"{k}_alpha_{q}" for q in questions_sc]
fig, ax = plt.subplots(1,1, figsize=(18/2.54, 9/2.54))
waves = [1]
aaa = df_p.copy()
# aaa = df_p[alpha_cols].divide(df_p["sumCorrAlpha"], axis=0).join(df_p, lsuffix="_relative")
# alpha_cols = [f"corr_alpha_{q}_relative" for q in questions_sc]
aa = aaa.loc[aaa.wave.isin(waves), alpha_cols+["party_close"]].melt(id_vars="party_close", ).reset_index().replace(dict(zip(alpha_cols, questions_sc)))
aa["value"] = aa["value"]

sns.stripplot(aa, x="variable", y="value", hue="party_close", hue_order=parties+["No party"], palette=party_cmap, alpha=0.8, size=1, dodge=True, legend=False, jitter=2/len(parties+["No party"]))
sns.barplot(aa, x="variable", y="value", hue="party_close", hue_order=parties+["No party"], palette=party_cmap, err_kws={'linewidth': 1}, alpha=0.8, estimator='mean', errorbar=('ci', 95), fill=False)
ax.set_xticklabels([labelMap_nl[l.get_text()] for l in ax.get_xticklabels()])
fig.autofmt_xdate(rotation=20, ha="center")
ax.set_xlabel("")
# ax.hlines(1/6, -0.5, len(questions_sc)-0.5, linestyles="--", colors="grey")
ax.set_ylabel(f"issue weight \n(correlation, {f'wave {waves[0]}' if len(waves)==1 else 'both waves'})")
handles, labels = ax.get_legend_handles_labels()
c = df_p.loc[df_p.wave.isin(waves), ["party_close"]].value_counts()
labels = [f'{l} ($n={c[l]}$)' for l in labels]
ax.legend(handles, labels, ncols=3, handlelength=2, columnspacing=0.5,  frameon=False, fontsize=8)
ax.grid(axis="y")
ax.set_ylim(min(df_p[alpha_cols].min()),1.2)
ax.set_xlim(-0.5,len(questions_sc)-0.5)
# plt.savefig("issue_weights_by_lr.png", dpi=600)


#%%
cmapLR = {'left':'#d8b365', 'moderate':"#A1A1A1", 'right':'#5ab4ac'}
fig, ax = plt.subplots(1,1, figsize=(18/2.54, 9/2.54))
waves = [1]
df_p["lr_label"]= pd.cut(df_p.lr, bins=LRcuts, labels=cmapLR.keys())
aaa = df_p.copy()
# aaa = df_p[alpha_cols].divide(df_p["sumCorrAlpha"], axis=0).join(df_p, lsuffix="_relative")
# alpha_cols = [f"corr_alpha_{q}_relative" for q in questions_sc]
aa = aaa.loc[aaa.wave.isin(waves), alpha_cols+["lr_label"]].melt(id_vars="lr_label", ).reset_index().replace(dict(zip(alpha_cols, questions_sc)))
aa["value"] = aa["value"]
sns.stripplot(aa, x="variable", y="value", hue="lr_label", hue_order=cmapLR.keys(), palette=cmapLR, alpha=0.8, size=1, dodge=True, legend=False, jitter=1/len(aa["lr_label"].unique()))
sns.barplot(aa, x="variable", y="value", hue="lr_label", hue_order=cmapLR.keys(), palette=cmapLR, err_kws={'linewidth': 3}, alpha=0.8, estimator='mean', errorbar=('ci', 95), fill=False)
ax.set_xticklabels([labelMap_nl[l.get_text()] for l in ax.get_xticklabels()])
fig.autofmt_xdate(rotation=20, ha="center")
ax.set_xlabel("")
# ax.hlines(1/6, -0.5, len(questions_sc)-0.5, linestyles="--", colors="grey")
ax.set_ylabel(f"issue weight \n(correlation, {f'wave {waves[0]}' if len(waves)==1 else 'both waves'})")
handles, labels = ax.get_legend_handles_labels()
c = df_p.loc[df_p.wave.isin(waves), ["lr_label"]].value_counts()
labels = [f'{l} ($n={c[l]}$)' for l in labels]
ax.legend(handles, labels, ncols=3, handlelength=2, columnspacing=0.5,  frameon=False, fontsize=8)
ax.grid(axis="y")
ax.set_ylim(min(df_p[alpha_cols].min()),1.2)
ax.set_xlim(-0.5,len(questions_sc)-0.5)
# plt.savefig("issue_weights_by_lr.png", dpi=600)


# %%
alphas = df_p.loc[(df_p.id==330717073703941) & (df_p.wave==2), [f"corrP_alpha_{q}" for q in questions_sc]]  
alphas/alphas.sum(axis=1).values[0]

# %%

# %%



# %%
for q in questions_sc:
    print(q, df_p[[f"w_{q}", f"{k}_alpha_{q}"]].corr().iloc[0,1])
# %%
