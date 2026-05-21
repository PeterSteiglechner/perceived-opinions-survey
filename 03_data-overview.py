# %%
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

# %% [markdown]
# # Variable Overview

# %% [markdown]
# ### Exlcusion Criteria

# %%
df_p_full = pd.read_csv("processed_data/2026-05-13_data_processed_participant.csv")
print("size of original data: ", len(df_p_full), " including wave 1 and 2: ", df_p_full["wave"].value_counts().to_dict())
df_p = df_p_full.loc[(~df_p_full.excl_double) & (~df_p_full.excl_NA) & (~df_p_full.excl_time)].copy()
print(f"Nr of distinct participants: {len(df_p["id"].unique())}")
print(f"    after excluding (i) participants with NA opinions ({sum(df_p_full.excl_NA)}) and (ii) participants with less than 1/3 of the median completion time ({sum(df_p_full.excl_time)}) and (iii) double entries (already excluded in preprocessing step)")
print("size of updated data: ", len(df_p), " including wave 1 and 2: ", df_p["wave"].value_counts().to_dict())
ids_w1 = df_p.loc[df_p.wave==1, "id"].unique()
ids_w2 = df_p.loc[df_p.wave==2, "id"].unique()
inds_bothwaves = df_p["id"].value_counts().reset_index().query("count==2")["id"].tolist()
print("Nr of participants who completed both waves: ", len(inds_bothwaves))
df_p_bothwaves = pd.read_csv("processed_data/2026-05-13_data_processed_participant_pivot.csv").query(f"id in {inds_bothwaves}")
print(len(df_p_bothwaves))
# print(df_p.shape, df_p.columns[50:60])

# %%
df_diff = pd.read_csv("processed_data/2026-05-13_data_processed_differences.csv")
df_diff = df_diff.loc[((df_diff["id"].isin(ids_w1)) & (df_diff["wave"]==1)) | ((df_diff["id"].isin(ids_w2)) & (df_diff["wave"]==2))]
print("Full Size of pairwise data (in wave 1 and wave 2): ")
print(df_diff["wave"].value_counts().to_dict())
print(df_diff.shape, df_diff.columns)
print(len(df_diff.loc[df_diff.wave==2,"id"].unique()))

# %% [markdown]
# ## Data 

# %%
plt.figure(figsize=(12/2.54, 5/2.54))
sns.histplot((pd.to_datetime(df_p_bothwaves["wave2_t_completed"]) - pd.to_datetime(df_p_bothwaves['wave1_t_completed'])).dt.days, bins=np.arange(0.5,100.5))
plt.xlabel("time between wave 1 and wave 2 [days]" )

# %%
fig, ax = plt.subplots(1,1, sharex=False)
t = 'time_total'
vmax = np.percentile(df_p.assign(t_div_60=df_p[t] / 60)['t_div_60'].values, 95) * 3
sns.histplot(df_p.assign(t_div_60=df_p[t] / 60), x='t_div_60', hue="wave", bins = np.arange(0,vmax,3),  alpha=0.5)
ax.set_title(t)
ax.text(0.95,0.95,
        "\n".join([f"{np.sum(df_p.loc[df_p.wave==w, t].values / 60 < vmax)/df_p.loc[df_p.wave==w, t].count()*100:.1f}% with <{vmax:.1f}min (wave {w})" for w in [1,2]]),
        ha="right", va="top", transform=ax.transAxes, fontsize=7)
ax.set_xlabel("time in min")
fig.tight_layout()

print(f"median ({t}): {df_p[t].div(60).median():.2f} (25%-perc: {df_p[t].div(60).describe()["25%"]:.2f}, 75%-perc: {df_p[t].div(60).describe()["75%"]:.2f})")

times = ['time_trainingGame', 'time_training', 'time_spam', 'time_spam18dots', 'time_pairwise', 'time_pairwise18pairs']
fig, axs = plt.subplots(2,3, sharex=False)
for ax, t in zip(axs.flatten(), times):
    vmax = np.percentile(df_p[t].dropna().values, 95) * 3 / 60
    sns.histplot(df_p.assign(t_div_60=df_p[t] / 60), x='t_div_60', hue="wave", bins = np.arange(0,vmax,0.5),  alpha=0.5, ax=ax, legend=False)
    ax.set_title(t)
    ax.text(0.95,0.95,
            "\n".join([f"{np.sum(df_p.loc[df_p.wave==w, t].values / 60 < vmax)/df_p.loc[df_p.wave==w, t].count()*100:.1f}% with <{vmax:.1f}min (wave {w})" for w in [1,2]]),
            ha="right", va="top", transform=ax.transAxes, fontsize=7)
    ax.set_xlabel("time in min")
fig.tight_layout()

display(df_p[times].div(60).describe())
display(df_p[times].div(60).describe())


# for t in [4,6]:
#     descr = df_p[time_cols(1)[t]].div(60).describe(percentiles=[0.05,0.25,0.5,0.75,0.95])
#     print(f"median ({time_cols(1)[t]}): {descr["50%"]:.2f} (25%-perc: {descr["25%"]:.2f}, 75%-perc: {descr["75%"]:.2f}), (5%-perc: {descr["5%"]:.2f}, 95%-perc: {descr["95%"]:.2f})")
#     descr = df_p[time_cols(2)[t]].div(60).describe(percentiles=[0.05,0.25,0.5,0.75,0.95])
#     print(f"median ({time_cols(2)[t]}): {descr["50%"]:.2f} (25%-perc: {descr["25%"]:.2f}, 75%-perc: {descr["75%"]:.2f})  (5%-perc: {descr["5%"]:.2f}, 95%-perc: {descr["95%"]:.2f})")


# %% [markdown]
# Note: some of the times are negative and should not be negative. This is probably because they reloaded the page and this changed the visited time.

# %% [markdown]
# ## Demographics
# 

# %% [markdown]
# ### Age Gender Region (bilendi meta-data)

# %%
demo_cols = ["gender", "age", "party_vote", "region"]  # participant-level constants
for d in demo_cols:
    fig = plt.figure(figsize=(5,2))
    ax = plt.axes()
    sns.histplot(df_p[d], ax=ax)
    fig.autofmt_xdate()

# %% [markdown]
# #### Parties / political identity

# %%
party_cmap["Miscellaneous"] = "brown"
party_cmap["Not Voting"] = "darkgrey"
fig, axs = plt.subplots(3,1, sharex=True)
for d, w, name, ax in zip(["party_vote", "party_close", "party_close"], [1,1,2], ["vote", "close_wave1", "close_wave2"], axs.flatten()):
    sns.barplot(df_p.loc[df_p.wave==w, d].value_counts()[parties_vote if "vote" in d else parties_full].reset_index(), ax=ax, y="count", x=d, hue=d, palette=party_cmap)
    fig.autofmt_xdate()
    ax.set_ylabel(name)
    ax.set_xlabel("")

# %%
fig, axs = plt.subplots(3,1, figsize=(12/2.54, 12/2.54), sharex=True, sharey=False)
for (d1, d2), ax in zip(combinations(["party_vote", "wave1_party_close", "wave2_party_close"], 2), axs):

    sns.heatmap(df_p_bothwaves[[d1,d2,"age"]].pivot_table(index=d1,columns=d2, aggfunc="count",values="age").loc[parties_full if "wave" in d1 else parties_vote, parties_full if "wave" in d2 else parties_vote], cmap="Reds", ax=ax, cbar=False,)
    # ax.set_aspect("equal")
    ax.set_title(d2)
    # ax.set_ylabel("")
axs[-1].set_xlabel("")
fig.autofmt_xdate(rotation=30)
fig.tight_layout()

# %%
for var in ["lr", "polInterest", "polFrequency", "n_contacts"]:
    fig, axs = plt.subplots(1,2, figsize=(16/2.54,6/2.54))
    sns.histplot(df_p, x=var, hue="wave", ax=axs[0], alpha=0.6)
    sns.regplot(df_p_bothwaves[["wave2_"+var, "wave1_"+var]], x="wave1_"+var, y="wave2_"+var, ax=axs[1], scatter_kws={"s":3, "alpha":0.5, "color":"grey"}, y_jitter=0.2*(var=="n_contacts"), x_jitter=0.2*(var=="n_contacts"))
    axs[1].set_aspect("equal")
    print(f"correlation wave 2 wave 1 {var}: {df_p_bothwaves[['wave2_'+var, 'wave1_'+var]].corr().values[1,0]}")


# %%
for var in ["P_tot"]+[f"P_{q}" for q in questions_sc]:
    fig, axs = plt.subplots(1,2, figsize=(16/2.54,6/2.54))
    sns.histplot(df_p, x=var, hue="wave", ax=axs[0], alpha=0.6)
    sns.regplot(df_p_bothwaves, x="wave1_"+var, y="wave2_"+var, ax=axs[1], scatter_kws={"s":3, "alpha":0.5, "color":"grey"})
    axs[1].set_aspect("equal")
    print(f"correlation wave 2 wave 1 {var}: {df_p_bothwaves[['wave2_'+var, 'wave1_'+var]].corr().values[1,0]}")
descr1P = df_p[["P_tot"]+[f"P_{q}" for  q in questions_sc]].describe()
display(descr1P)


# %%
descr1P.loc[["mean", "std", "50%"]].plot.bar()
plt.ylabel("issue polarisation")

# %% [markdown]
# ## Issue Importance

# %%
for var in [f"w_{q}" for q in questions_sc]:
    fig, axs = plt.subplots(1,2, figsize=(16/2.54,6/2.54))
    sns.histplot(df_p, x=var, hue="wave", ax=axs[0], alpha=0.6)
    sns.regplot(df_p_bothwaves, x="wave1_"+var, y="wave2_"+var, ax=axs[1], scatter_kws={"s":3, "alpha":0.5, "color":"grey"})
    axs[1].set_aspect("equal")
    print(f"correlation wave 2 wave 1 {var}: {df_p_bothwaves[['wave2_'+var, 'wave1_'+var]].corr().values[1,0]}")
    fig.tight_layout()
descr1w = df_p[[f"w_{q}" for  q in questions_sc]].describe()
display(descr1w)


# %%
descr1w.loc[["mean", "std", "50%"]].plot.bar()
plt.ylabel("issue importance")

# %%
df_p[[f"w_{q}" for  q in questions_sc]].describe()

# %%
aa.columns[1]

# %%
var = "sum_issue_importance"
fig, axs = plt.subplots(1,2, figsize=(16/2.54,6/2.54))
df_p[var] = df_p[[f"w_{q}" for  q in questions_sc]].sum(axis=1)
sns.histplot(df_p,x=var, hue="wave", ax=axs[0], alpha=0.6)
aa = df_p.pivot_table(index="id", columns="wave", values=var).rename(columns={1:"wave1", 2:"wave2"})
sns.regplot(aa, x="wave1", y="wave2", ax=axs[1], scatter_kws={"s":3, "alpha":0.5, "color":"grey"})
axs[1].set_aspect("equal")
print(f"correlation wave 2 wave 1 {var}: {aa.corr().values[1,0]}")
fig.tight_layout()
display(df_p[var].describe())


# %% [markdown]
# ## Voter Sympathy

# %%
from itertools import zip_longest

N = len(partiesVars)
ncols = 2
nrows = (N + 1) // ncols  # ceiling division

fig, axs = plt.subplots(
    nrows, ncols,
    figsize=(16/2.54, 12/2.54),
    sharex=True, sharey=True
)
axs_flat = axs.flatten()

for i, p in enumerate(partiesVars):
    ax = axs_flat[i]
    sns.histplot(
        data=df_p,
        x=f"sym_{p}",
        hue="party_close",
        ax=ax,
        alpha=0.6,
        palette=party_cmap,
        multiple="stack",
        bins=np.linspace(-0.01, 1.01, 21),
        legend=(i == N - 1),
        linewidth=0,
        hue_order=parties_full,
    )
    ax.set_title(f"towards {p} voter", x=0.95, y=0.8,
                 color=party_cmap[p], va="top", ha="right")
    ax.set_ylabel("")
    ax.set_xlabel("sympathy towards typical voter" if i >= N - ncols else "")

# Hide unused axes (last cell if N is odd)
for ax in axs_flat[N:]:
    ax.axis("off")

ax = axs_flat[N-1]
sns_legend = ax.get_legend()
handles = sns_legend.legend_handles
labels = [t.get_text() for t in sns_legend.get_texts()]
sns_legend.remove()

# Hide the last axes but keep it as a legend host
empty_ax = axs_flat[N]
empty_ax.axis("off")

empty_ax.legend(
    handles, labels,
    bbox_to_anchor=(0.05, 0.5),
    ncol=2,
    fontsize=6,
    title="party id",
    title_fontsize=6,
    loc="center left",
    borderaxespad=0,
)

display(df_p[[f"sym_{p}" for p in partiesVars]].describe())

fig.tight_layout()
plt.savefig("sympathy.png", dpi=600)

# %% [markdown]
# # Traning

# %%
for var in ["attemptsPractice"]:
    fig, axs = plt.subplots(1,2, figsize=(16/2.54,6/2.54))

    sns.histplot(df_p.replace({-999:8}), x=var, hue="wave", ax=axs[0], bins=0.5+np.arange(0,9), alpha=0.6)
    sns.regplot(df_p_bothwaves[["wave1_"+var, "wave2_"+var]].replace({-999:np.nan}), x="wave1_"+var, y="wave2_"+var, ax=axs[1], scatter_kws={"s":3, "alpha":0.5, "color":"grey"}, y_jitter=0.2*(var=="attemptsPractice"), x_jitter=0.2*(var=="attemptsPractice"))
    axs[1].set_aspect("equal")
    print(f"correlation wave 2 wave 1 {var}: {df_p_bothwaves[['wave2_'+var, 'wave1_'+var]].replace({-999:np.nan}).corr().values[1,0]}")


display(df_p.groupby("wave")["attemptsPractice"].replace({-999:np.nan}).describe())

# %%
display(df_p[[f"dist_game_{a[0]}-{b[0]}" for a,b in combinations(practice_game_dots, 2)]].describe())
fig = plt.figure(figsize=(2,2))
df_p[[f"dist_game_{a[0]}-{b[0]}" for a,b in combinations(practice_game_dots, 2)]].mean().plot.bar()
fig.autofmt_xdate()


# %%
vars = [f"dist_practice_{a[0]}-{b[0]}" for a,b in combinations(practice_training_dots, 2)]
display(df_p[vars].describe())
fig = plt.figure(figsize=(6,2))
df_p[vars].mean().plot.bar(color=plt.get_cmap("tab10").colors)
plt.xticks(plt.gca().get_xticks(), labels=[s[-3:]+" ("+s[:5]+")"  for s in vars])
fig.autofmt_xdate()


# %%
df_p.groupby("wave")["passed_practice_sanity"].value_counts().reset_index()

# %% [markdown]
# ## Mapping

# %%
vars = ["mappingEnjoy", "mappingEasier", "map_satisfaction"]
for var in vars:
    fig, axs = plt.subplots(1,2, figsize=(16/2.54,6/2.54))
    sns.histplot(df_p,x=var, hue="wave", ax=axs[0], alpha=0.6)
    sns.regplot(df_p_bothwaves, x="wave1_"+var, y="wave2_"+var, ax=axs[1], scatter_kws={"s":3, "alpha":0.5, "color":"grey"})
    axs[1].set_aspect("equal")
    print(f"correlation wave 2 wave 1 {var}: {df_p_bothwaves[['wave2_'+var, 'wave1_'+var]].corr().values[1,0]}")
    fig.tight_layout()

display(df_p[vars].describe())
fig, axs = plt.subplots(1,3, figsize=(16/2.54,6/2.54))
for (var1, var2), ax in zip(combinations(vars,2), axs.flatten()):
    sns.regplot(df_p, x=var1, y=var2, ax=ax, scatter_kws={"s":3, "alpha":0.5, "color":"grey"})


# %%
vars = ["average_pixel_dist", "average_pixel_dist_parties"]
for var in vars:
    fig, axs = plt.subplots(1,2, figsize=(16/2.54,6/2.54))
    sns.histplot(df_p, x=var, hue="wave", ax=axs[0], alpha=0.6)
    sns.regplot(df_p_bothwaves, x="wave1_"+var, y="wave2_"+var, ax=axs[1], scatter_kws={"s":3, "alpha":0.5, "color":"grey"})
    axs[1].set_aspect("equal")
    print(f"correlation wave 2 wave 1 {var}: {df_p_bothwaves[['wave2_'+var, 'wave1_'+var]].corr().values[1,0]}")
    fig.tight_layout()

display(df_p[vars].describe())


# %% [markdown]
# # Opinions

# %%
vars = [f"x_self_{q}" for q in questions_sc]
varsPrior = [f"first_x_self_{q}" for q in questions_sc]
wave = 2
for var, varp in zip(vars, varsPrior):
    fig, axs = plt.subplots(1,2, figsize=(16/2.54,6/2.54))
    sns.histplot(df_p[[varp, var]], ax=axs[0], alpha=0.6)
    sns.regplot(df_p, x=varp, y=var, ax=axs[1], scatter_kws={"s":3, "alpha":0.5, "color":"grey"})
    axs[1].set_aspect("equal")
    print(f"correlation wave {wave} {var} and {varp}: {df_p[[varp, var]].corr().values[1,0]}")
    fig.tight_layout()

print(f"number of people who changed their opinions: {dict(zip(questions_sc, ((np.abs(df_p[[var for  var in vars]].values - df_p[[varp for  varp in varsPrior]].values)>0).sum(axis=0))))}")


# %%
vars = [f"x_self_{q}" for q in questions_sc]
for var in vars:
    fig, axs = plt.subplots(1,2, figsize=(16/2.54,6/2.54))
    sns.histplot(df_p_bothwaves[["wave2_"+var, "wave1_"+var]], ax=axs[0], alpha=0.6)
    sns.regplot(df_p_bothwaves[["wave2_"+var, "wave1_"+var]], x="wave1_"+var, y="wave2_"+var, ax=axs[1], scatter_kws={"s":3, "alpha":0.5, "color":"grey"})
    axs[1].set_aspect("equal")
    print(f"correlation wave 2 wave 1 {var}: {df_p_bothwaves[['wave2_'+var, 'wave1_'+var]].corr().values[1,0]}")
    fig.tight_layout()

# display(df_p[[var for  var in vars]].describe())


# %%
fig, axs = plt.subplots(2,2, figsize=(12/2.54,7/2.54), sharex=True, sharey=True)
for ax, q in zip(axs.flatten(), questions_sc):
    sns.histplot(df_p, x = f"x_self_{q}", hue="wave", bins=11, binrange=(-1,1), ax=ax, alpha=0.6,  legend=False, stat="percent", multiple="layer", kde=False)
    ax.set_title(q)
fig.suptitle("Own opinions")
fig.tight_layout()


# %%
fig, axs = plt.subplots(2,3, figsize=(16/2.54,8/2.54), sharex=True, sharey=True)
for ax, q in zip(axs.flatten(), questions_sc):
    a = df_p[[f"x_{p}_{q}" for p in partiesVars]].melt(var_name="party", )
    a["party"]= a["party"].apply(lambda x: x.split("_")[1])
    sns.histplot(a, x="value", hue="party", palette=party_cmap, hue_order=partiesVars, bins=11, binrange=(-1,1), ax=ax, alpha=0.6, legend=False, stat="percent", multiple="layer", kde=False)
    ax.set_title(q)
fig.suptitle("What are the opinions of a typical voter of party...?")
fig.tight_layout()


# %%
fig, axs = plt.subplots(2,2, figsize=(16/2.54,14/2.54), sharex=True, sharey=True)
fig.suptitle("What are the opinions of a typical voter of party...?")
for ax, q in zip(axs.flatten(), questions_sc):
    a = df_p[[f"x_{p}_{q}" for p in partiesVars]].melt(var_name="party")
    a["party"]= a["party"].apply(lambda x: x.split("_")[1])
    sns.violinplot(a, x="value", y="party", hue="party", fill=False, inner=None, palette=party_cmap, hue_order=partiesVars, ax=ax, legend=False, cut=0, )
    sns.stripplot(a, x="value", y="party", hue="party", palette=party_cmap, hue_order=partiesVars, ax=ax, legend=False, size=1)
    ax.set_title(q)
fig.tight_layout()


# %% [markdown]
# ## Std Dev of Opinions in social circles

# %%
fig, axs = plt.subplots(2,3, figsize=(16/2.54,14/2.54), sharex=True, sharey=True)
fig.suptitle("Std Dev of Opinions in social circles")
for ax, q in zip(axs.flatten(), questions_sc):
    a = df_p[[f"std_socialCircle_ops_{q}"]+[f"party_close"]].melt(var_name="social circle", id_vars=[f"party_close"])
    sns.violinplot(a, x="value", y=f"party_close", hue=f"party_close", fill=False,  palette=party_cmap, hue_order=parties_full, ax=ax, legend=False, cut=0, inner="quart", order=parties_full)
    sns.stripplot(a, x="value", y=f"party_close", hue=f"party_close", palette=party_cmap, hue_order=parties_full, ax=ax, legend=False, size=1)
    sns.stripplot(a.groupby(f"party_close")["value"].mean().reset_index(), x="value", y=f"party_close", hue=f"party_close", palette=party_cmap, hue_order=parties_full, ax=ax, legend=False, size=5, marker="s")
    ax.set_title(q)
fig.tight_layout()


# %% [markdown]
# ### Treatment

# %%
print(f"Treatment: {df_p["treatment_wave2"].value_counts().to_dict()}")

# %%
fig, ax = plt.subplots(1,1, figsize=(16/2.54,14/2.54))
ax.set_title("Std Dev of Opinions in social circles")
a = df_p[[f"std_socialCircle_ops_{q}" for q in questions_sc]+["treatment_wave2"]].melt(var_name="question", id_vars=[f"treatment_wave2"])
a["question"]= a["question"].apply(lambda x: "_".join(x.split("_")[3:]))
a = a.dropna()
print(len(a))
sns.violinplot(a, x="value",  y="question",split=True, hue="treatment_wave2", fill=False,   ax=ax, legend=False, cut=0, inner="quart")
sns.stripplot(a, x="value",  y="question",  hue=f"treatment_wave2",  ax=ax, legend=False, size=1, dodge=True)
sns.stripplot(a.groupby([f"treatment_wave2", "question"])["value"].mean().reset_index(), x="value", y=f"question", hue=f"treatment_wave2", ax=ax, legend=True, size=5, marker="s")
fig.tight_layout()


# %%
fig, axs = plt.subplots(2,3, sharex=True, sharey=True)
w=2
fig.suptitle("ALTERNATIVE: Std Dev of Opinions in social circles")

for ax, qq in zip(axs.flatten(), questions_sc):
    sns.boxplot(df_p, x="treatment_wave2", y=f"std_socialCircle_ops_{qq}", ax=ax, fill=False, whis=[5,95], fliersize=0)
    sns.stripplot(df_p, x="treatment_wave2", y=f"std_socialCircle_ops_{qq}", ax=ax, size=1,alpha=0.3)
    sns.stripplot(df_p.groupby("treatment_wave2")[f"std_socialCircle_ops_{qq}"].mean().reset_index(), x="treatment_wave2", y=f"std_socialCircle_ops_{qq}", ax=ax, size=10, marker="s",alpha=0.4)
    ax.set_title(qq)
    ax.set_xlabel("")
    ax.set_ylabel("social circle opinion std")
axs[-1,1].set_xlabel("Treatment")
fig.tight_layout()

# %% [markdown]
# # DISTANCES

# %%
diff_cols = [f'deltaX_{q}' for q in questions_sc] + ['pixel_dist']
sns.pairplot(df_diff.sample(500)[diff_cols], plot_kws={'size':0.1}, )

# %%
fig, axs = plt.subplots(2,2, figsize=(16/2.54, 10/2.54))
sns.histplot(df_diff, x="pairwise_similarity", hue="wave", palette="Set1", hue_order=[2,1], ax=axs[0,0])

sns.histplot(df_diff, x="pixel_dist", hue="wave", palette="Set1", hue_order=[2,1], ax=axs[0,1])


sns.histplot(df_diff, x="sympathy", hue="wave", palette="Set1", hue_order=[2,1], ax=axs[1,0])

sns.histplot(df_diff, x="socialCloseness", hue="wave", palette="Set1", hue_order=[2,1], ax=axs[1,1])

fig.tight_layout()

# %%
diff_cols = [f'deltaX_{q}' for q in questions_sc] + ['pixel_dist']
mask = np.triu(np.ones_like(df_diff[diff_cols].corr()))
sns.heatmap(df_diff[diff_cols].corr(), annot=True, cmap="hot_r", vmax=1, vmin=0, mask=mask, cbar_kws={'label':"correlation"})

# %%
print(f"Correlation between Sympathy and pixel distance of a voter: {df_diff[['sympathy']+['pixel_dist']].corr().iloc[0,1]}")
sns.lmplot(df_diff, x="pixel_dist", y="sympathy", hue="party", hue_order=partiesVars, palette=party_cmap, order=1, scatter_kws={"s":1, "alpha":0.3 },  y_jitter=0.015 )
plt.ylim(-0.05,1.05)
plt.xlim(-0.05,1.05)

# %%
import statsmodels.formula.api as smf

model = smf.logit("sympathy ~ pixel_dist", data=df_diff).fit()
print(model.summary())
fig, ax = plt.subplots()

for party in partiesVars:
    df_party = df_diff[df_diff["party"] == party]
    model = smf.logit("sympathy ~ pixel_dist", data=df_party).fit(disp=0)  # disp=0 silences output

    x_range = np.linspace(df_diff["pixel_dist"].min(), df_diff["pixel_dist"].max(), 300)
    y_pred = model.predict(exog=dict(pixel_dist=x_range))

    ax.scatter(df_party["pixel_dist"], df_party["sympathy"],
               s=1, alpha=0.3, color=party_cmap[party])
    ax.plot(x_range, y_pred, lw=2, label=party, color=party_cmap[party])

ax.set_ylim(-0.05, 1.05)
ax.set_xlim(-0.05, 1.05)
ax.set_xlabel("Pixel Distance")
ax.set_ylabel("Sympathy")
ax.legend()

# %%
sns.histplot(df_diff, x="sympathy", hue="ingroupdummy", palette="Set2",)

# %%
print(f"Correlation between social Closeness and pixel distance: {df_diff[['socialCloseness']+['pixel_dist']].corr().iloc[0,1]}")
df_diff["lr_cat"] =pd.cut(df_diff.lr, np.linspace(0, 1, 4), right=False, labels=["left", "moderate", "right"])
sns.lmplot(df_diff, x="pixel_dist", y="socialCloseness", hue="lr_cat", scatter_kws={"s":1, "alpha":0.3 }, y_jitter=0.01)
plt.ylim(-0.05,1.05)

# %%

df_p["std_socialCircle_ops_east_germans"]

# %%
sns.histplot(df_diff, x="treatment_wave2", y="y")

# %%
df_p["treatment_wave2"]#.join(df_diff["treatment_wave2"]


# %%


# %%
df_diff["treatment"] = False
df_diff.loc[df_diff["id"].isin(df_p.loc[df_p.treatment_wave2==1, "id"].tolist()), "treatment"] = True 

sns.violinplot(df_diff.loc[df_diff.wave==2], x="treatment", y="sympathy")
sns.stripplot(df_diff.loc[df_diff.wave==2], x="treatment", y="sympathy",size=1)

# %%
q = questions_sc[0]
sns.violinplot(df_diff.loc[(df_diff.wave==2) & (df_diff.dot2.isin(partiesVars)) & (df_diff.dot1=="self") ], x="treatment", y=f"deltaX_{q}")
sns.stripplot(df_diff.loc[(df_diff.wave==2) & (df_diff.dot2.isin(partiesVars)) & (df_diff.dot1=="self")], x="treatment", y=f"deltaX_{q}",size=1,)

# %%
df_diff.loc[(df_diff.wave==2) & (df_diff.dot2.isin(partiesVars)) & (df_diff.dot1=="self")].groupby("treatment")[f"deltaX_{q}"].describe()

# %%
df_diff.loc[~df_diff.sympathy.isna()].groupby("treatment")["sympathy"].describe()

# %%
df_diff.loc[df_diff.wave==2, "id"].unique()

# %%



