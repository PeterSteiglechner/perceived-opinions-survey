
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from consts import *
import numpy as np
from itertools import combinations
from scipy.stats import linregress
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import json

plt.rcParams.update({"font.size":9})
plt.rcParams.update({"figure.figsize":(16/2.54, 9/2.54)})
sns.set_style("ticks")
sns.set_context("paper")


#%%

df_p = pd.read_csv("processed_data/2026-07-07_data_processed_participant_withAllIssueWeights.csv")
df_diff = pd.read_csv("processed_data/2026-07-07_data_processed_differences_withAllIssueWeights.csv")

# %%
df_diff["treatment"] = False
df_diff.loc[df_diff["id"].isin(df_p.loc[df_p.treatment_wave2==1, "id"].tolist()), "treatment"] = True 

plt.figure()
sns.violinplot(df_diff.loc[df_diff.wave==2], x="treatment", y="sympathy")
sns.stripplot(df_diff.loc[df_diff.wave==2], x="treatment", y="sympathy",size=1)


# %%
plt.figure()
q = questions_sc[0]
sns.violinplot(df_diff.loc[(df_diff.wave==2) & (df_diff.dot2.isin(partiesVars)) & (df_diff.dot1=="self") ], x="treatment", y=f"deltaX_{q}")
sns.stripplot(df_diff.loc[(df_diff.wave==2) & (df_diff.dot2.isin(partiesVars)) & (df_diff.dot1=="self")], x="treatment", y=f"deltaX_{q}",size=1,)



# %%
diff_cols = [f'deltaX_{q}' for q in questions_sc] + ['pixel_dist']
sns.pairplot(df_diff.sample(500)[diff_cols], plot_kws={'size':0.1}, )



# %%
fig, axs = plt.subplots(2,2, figsize=(7/2.54, 7/2.54))
df_diff["dottype"] = df_diff.apply(lambda x: "personal" if (("reference" in x['dot1'] or "self" in x["dot1"]) and ("reference" in x['dot2'] or "self" in x["dot2"])) else "voter", axis=1)

hue= None #"dottype"
hue_order = None #["voter", "personal"] #[2,1]
cmap =  "#1f78b4" # {"voter":"k", "personal":party_cmap["contact"]}#
mult = "stack"
sns.histplot(df_diff, x="pairwise_similarity", hue=hue, palette=cmap, color=cmap, hue_order=hue_order, ax=axs[0,0], multiple=mult, kde=True, bins=21)
axs[0,0].set_xlabel("pairwise similarity")

sns.histplot(df_diff, x="pixel_dist", hue=hue, palette=cmap, color=cmap,hue_order=hue_order, ax=axs[0,1], multiple=mult, kde=True, bins=21)
axs[0,1].set_xlabel("map distance")

sns.histplot(df_diff, x="sympathy", hue=hue, palette=cmap, color=cmap,hue_order=hue_order, ax=axs[1,0], multiple=mult, kde=True, bins=21)
axs[1,0].set_xlabel("likability [voters]")

sns.histplot(df_diff, x="socialCloseness", hue=hue, palette=cmap,color=cmap, hue_order=hue_order, ax=axs[1,1], multiple=mult, kde=True, bins=21)
axs[1,0].set_xlabel("closeness [contacts]")

for ax in axs.flatten():
    ax.set_ylabel("")
    ax.set_yticks([])
    

fig.tight_layout()




# %%
plt.figure()
diff_cols = [f'deltaX_{q}' for q in questions_sc] + ['pixel_dist']
diff_cols_names = [fr'$\Delta x$ {q}' for q in qs] + ['map dist']
mask = np.triu(np.ones_like(df_diff[diff_cols].corr()))
colsNames = dict(zip(diff_cols, diff_cols_names))
sns.heatmap(df_diff[diff_cols].corr().rename(columns=colsNames, index=colsNames), annot=True, cmap="hot_r", vmax=1, vmin=0, mask=mask, cbar_kws={'label':"correlation"})


# %%
df_diff["pairwise_dissimilarity"] = 1-df_diff["pairwise_similarity"]
sns.lmplot(df_diff, x="pixel_dist", y="pairwise_dissimilarity", scatter_kws={"marker":".","s":0.1, "alpha":0.1 }, y_jitter=0.02, height=6/2.54, aspect=1,logistic=True,)
plt.ylim(-0.05,1.05)
plt.xlim(-0.05,1.05)
plt.title(f"correlation: {df_diff[['pairwise_dissimilarity']+['pixel_dist']].corr().iloc[0,1]:.2f}", y=0.05, x=1, ha="right")
plt.xlabel("map distance")
plt.ylabel("dissimilarity rating")
plt.tight_layout()
plt.savefig("figs/correlation_pairwiseD_map.png", dpi=300)

# %%
plt.figure()
print(f"Correlation between Sympathy and pixel distance of a voter: {df_diff[['sympathy']+['pixel_dist']].corr().iloc[0,1]}")
sns.lmplot(df_diff, x="pixel_dist", y="sympathy", hue="party", hue_order=partiesVars, palette=party_cmap, order=1, scatter_kws={"s":1, "alpha":0.3 },  y_jitter=0.015, logistic=True )
plt.ylim(-0.05,1.05)
plt.xlim(-0.05,1.05)

# %%
# fig = plt.figure(figsize=(5/2.54,5/2.54))
df_diff["dislikability"] = 1 - df_diff["sympathy"]
print(f"Correlation between dislikability and map distance: {df_diff[['dislikability']+['pixel_dist']].corr().iloc[0,1]}")
sns.lmplot(df_diff.loc[df_diff.dot1=="self"], x="pixel_dist", y="dislikability", scatter_kws={"marker":".","s":1, "alpha":0.1 }, y_jitter=0.01, height=6/2.54, aspect=1, logistic=True)
plt.ylim(-0.05,1.05)
plt.title(f"correlation: {df_diff[['dislikability']+['pixel_dist']].corr().iloc[0,1]:.2f}", y=0.05, x=1, ha="right")
plt.xlabel("map distance")
plt.ylabel("voter dislikability")
plt.tight_layout()
plt.savefig("figs/correlation_sympathyD_map.png", dpi=300)

# %%

plt.figure()
sns.histplot(df_diff, x="sympathy", hue="ingroupdummy", palette="Set2",)


# %%

plt.figure(figsize=(16/2.54,9/2.54))
print(f"Correlation between social closeness and pixel distance: {df_diff[['socialCloseness']+['pixel_dist']].corr().iloc[0,1]}")
df_diff["lr_cat"] =pd.cut(df_diff.lr, np.linspace(0, 1, 4), right=False, labels=["left", "moderate", "right"])
sns.lmplot(df_diff, x="pixel_dist", y="socialCloseness", hue="wave", scatter_kws={"s":1, "alpha":0.3 }, y_jitter=0.01, height=3)
plt.ylim(-0.05,1.05)
#%%


fig, ax = plt.subplots(1,1)
sympathy_matrix = df_diff.loc[df_diff.wave==2].pivot_table(columns="party", index="dot2", values="sympathy", aggfunc="mean")
sns.heatmap(sympathy_matrix.loc[partiesVars, partiesVars], annot=True, cmap="hot_r", cbar_kws={"label":"sympathy"}, ax=ax)
ax.set_aspect("equal")
ax.set_ylabel("evaluated party")
ax.set_xlabel("party participants feel closest to")
fig.autofmt_xdate()
#%%


fig, (ax1,ax2) = plt.subplots(1,2, sharex=True, sharey=True, figsize=(10,4))
sympathy_matrix = df_diff.loc[df_diff.wave==2].pivot_table(columns="party", index="dot2", values="sympathy", aggfunc="mean")
sns.heatmap(sympathy_matrix.loc[partiesVars, partiesVars], annot=True, cmap="hot_r", cbar_kws={"label":"sympathy"}, ax=ax1)
dist_matrix = df_diff.loc[df_diff.wave==2].pivot_table(columns="party", index="dot2", values="pixel_dist", aggfunc="mean")
sns.heatmap(dist_matrix.loc[partiesVars, partiesVars], annot=True, cmap="hot", cbar_kws={"label":"pixel distance"}, ax=ax2)
ax.set_aspect("equal")
ax.set_ylabel("evaluated party")
ax.set_xlabel("party participants feel closest to")
fig.autofmt_xdate()

#%%

res = []
for p in partiesVars:
    for p_own in partiesVars:
        a = df_diff.loc[(df_diff.wave==2) & (df_diff.treatment_wave2==False) ].query(f"party=='{p_own}' and dot2=='{p}'")[[f"std_socialCircle_ops_{q}" for q in questions_sc]+["sympathy", "dot2", "party"]]
        a["overall_std"] = a[[f"std_socialCircle_ops_{q}" for q in questions_sc]].mean(axis=1)
        res.append([p_own, p, a[["overall_std", "sympathy"]].corr().iloc[0,1]])
res = pd.DataFrame(res, columns=["party", "dot2", "r_std_sym"])#.describe()

#%%


#%%


fig, ax = plt.subplots(1,1)
sympathy_matrix = res.pivot_table(columns="party", index="dot2", values="r_std_sym")
sns.heatmap(sympathy_matrix.loc[partiesVars, partiesVars], annot=True, cmap="coolwarm", vmin=-1, vmax=1, cbar_kws={"label":"correlation: sympathy - std social circle"}, ax=ax, fmt=".2f")
ax.set_aspect("equal")
ax.set_ylabel("evaluated party")
ax.set_xlabel("party participants feel closest to")
fig.autofmt_xdate()

#%%

res = []
for p in partiesVars:
    for p_own in partiesVars:
        a = df_diff.loc[(df_diff.wave==1)].query(f"party=='{p_own}' and dot2=='{p}'")[[f"std_socialCircle_ops_{q}" for q in questions_sc]+["pixel_dist", "dot2", "party"]]
        a["overall_std"] = a[[f"std_socialCircle_ops_{q}" for q in questions_sc]].mean(axis=1)
        res.append([p_own, p, a[["overall_std", "pixel_dist"]].corr().iloc[0,1]])
res = pd.DataFrame(res, columns=["party", "dot2", "r_std_dist"])#.describe()

#%%


fig, ax = plt.subplots(1,1)
corr_matrix = res.pivot_table(columns="party", index="dot2", values="r_std_dist")
sns.heatmap(corr_matrix.loc[partiesVars, partiesVars], annot=True, cmap="coolwarm", vmin=-1, vmax=1, cbar_kws={"label":"correlation: distance - std social circle"}, ax=ax, fmt=".2f")
ax.set_aspect("equal")
ax.set_ylabel("evaluated party")
ax.set_xlabel("party participants feel closest to")
fig.autofmt_xdate()


#%%
a = df_diff.loc[(df_diff.wave==2) & (df_diff.treatment_wave2==False) ].query(f"party=='AfD' and dot2=='GreenParty'")[[f"std_socialCircle_ops_{q}" for q in questions_sc]+["sympathy", "dot2", "party"]].dropna()
a["overall_std"] = a[[f"std_socialCircle_ops_{q}" for q in questions_sc]].mean(axis=1)
sns.lmplot(a, x="overall_std", y="sympathy", height=3)
plt.title("How AfD like Greens?")
a = df_diff.loc[(df_diff.wave==2) & (df_diff.treatment_wave2==False) ].query(f"party=='GreenParty' and dot2=='AfD'")[[f"std_socialCircle_ops_{q}" for q in questions_sc]+["sympathy", "dot2", "party"]].dropna()
a["overall_std"] = a[[f"std_socialCircle_ops_{q}" for q in questions_sc]].mean(axis=1)
sns.lmplot(a, x="overall_std", y="sympathy", height=3)
plt.title("How Greens like AfD?")


#%%

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6, 3))

var = "sympathy"
p1 = r'GreenParty'
p2 = r'AfD'
condition = "wave==2 and treatment_wave2==False"

cols = [f"std_socialCircle_ops_{q}" for q in questions_sc] + [var, "dot2", "party"]

def plot_with_slope(ax, party, dot2, title):
    a = df_diff.query(condition).query(f"party=='{party}' and dot2=='{dot2}' and dot1 =='self'")[cols].dropna()
    a["overall_std"] = a[[f"std_socialCircle_ops_{q}" for q in questions_sc]].mean(axis=1)
    sns.regplot(data=a, x="overall_std", y=var, scatter_kws=dict(s=5, alpha=0.8), ax=ax)
    ax.set_title(title)

    res = linregress(a["overall_std"], a[var])
    p = res.pvalue
    ax.text(
        0.05, 0.95,
        f"slope = {res.slope:.3f}\n$R^2$ = {res.rvalue**2:.3f}\np = {p:.3g}"+(( "***" if p<0.001 else ("**" if p<0.01 else "*")) if p<0.05 else ""),
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )    
    return res

res1 = plot_with_slope(ax1, p1, p2, f"How {p1} affiliates like {p2}?")
res2 = plot_with_slope(ax2, p2, p1, f"How {p2} affiliates like {p1}?")

fig.tight_layout()


var = "sympathy"
wave = 2
a = df_diff.loc[(df_diff.wave==wave)].query(f"dot1 in {['self']} and dot2 in {partiesVars} and party in {partiesVars}")[[f"std_socialCircle_ops_{q}" for q in questions_sc]+[var, "dot2", "party"]]
a["overall_std"] = a[[f"std_socialCircle_ops_{q}" for q in questions_sc]].mean(axis=1)

def slope_and_stars(g):
    res = linregress(g["overall_std"], g[var])
    slope, p = res.slope, res.pvalue
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    else:
        stars = ""
    return pd.Series({"slope": slope, "p": p, "stars": stars})

results = a.groupby(["dot2", "party"]).apply(slope_and_stars)

corr_table = results["slope"].unstack("party").loc[partiesVars, partiesVars]
stars_table = results["stars"].unstack("party").loc[partiesVars, partiesVars]

# build combined annotation strings: slope value + stars
annot_labels = corr_table.round(1).astype(str) + stars_table

fig, ax = plt.subplots(1, 1)
sns.heatmap(
    corr_table, cmap="coolwarm", vmin=-1, vmax=1,
    cbar_kws={"label": rf"Correlations between {var} and $\overline{{\sigma_{{sc}}}}$"},
    ax=ax, annot=annot_labels, fmt="", annot_kws={"fontsize":7}
)
ax.set_aspect("equal")
fig.autofmt_xdate()


# corr_table = a.groupby(["dot2", "party"]).apply(lambda g:  linregress(g["overall_std"], g[var]).slope).unstack("party").loc[partiesVars, partiesVars]
# sns.heatmap(corr_table, cmap="coolwarm", vmin=-1,vmax=1, cbar_kws={"label":f"Correlations between {var} and $\overline{'{\sigma_{sc}}'}$"}, ax=ax, annot=True)

# ax.set_aspect("equal")
# fig.autofmt_xdate()


#%%
from scipy.stats import linregress

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6, 3))

var = "pixel_dist"
p1 = r'GreenParty'
p2 = r'AfD'
condition = "(wave==1) or (wave==2 and treatment_wave2==False)"

cols = [f"std_socialCircle_ops_{q}" for q in questions_sc] + [var, "dot2", "party"] + ["id", "wave"]

def plot_with_slope(ax, party, dot2, title):
    a = df_diff.query(condition).query(f"party=='{party}' and dot1=='{'self'}' and dot2=='{dot2}'")[cols].dropna()
    print(len(a["id"].unique()))
    a["overall_std"] = a[[f"std_socialCircle_ops_{q}" for q in questions_sc]].mean(axis=1)
    sns.regplot(data=a, x="overall_std", y=var, scatter_kws=dict(s=2, alpha=0.4), ax=ax)
    ax.set_title(title)

    res = linregress(a["overall_std"], a[var])
    p = res.pvalue
    ax.text(
        0.05, 0.95,
        f"slope = {res.slope:.3f}\n$R^2$ = {res.rvalue**2:.3f}\np = {p:.3g}"+(( "***" if p<0.001 else ("**" if p<0.01 else "*")) if p<0.05 else ""),
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )    
    return a, res

res1 = plot_with_slope(ax1, p1, p2, f"How {p1} affiliates position {p2}?")
res2 = plot_with_slope(ax2, p2, p1, f"How {p2} affiliates position {p1}?")

fig.tight_layout()




# TOOL ANALYSIS


#%% 
# Reliability
#%%
dists_parties = df_diff.loc[((df_diff.dot1=="self") | (df_diff.dot1.isin(partiesVars))) & (df_diff.dot2.isin(partiesVars)) & (df_diff.treatment_wave2==0), ["id", "wave", "dot1", "dot2", "pixel_dist"]].pivot_table(index=["id", "dot1", "dot2"], columns="wave", values="pixel_dist", ).rename(columns={1:"wave 1", 2:"wave 2"}).reset_index().dropna(subset=["wave 1", "wave 2"])
reliability_corrs = dists_parties[["dot1", "dot2", "wave 1", "wave 2"]].groupby(["dot1", "dot2"]).corr().reset_index().query("wave=='wave 2'" ).set_index(["dot1", "dot2"])["wave 1"]
reliability_corrs

d = dists_parties.loc[dists_parties.dot1=="self"]
print(f"correlation wave1_mapDistance -- wave2_mapDistance of pairs including self and a typical voter of party A: {pearsonr(d["wave 1"], d["wave 2"]).statistic:.3f}")
d = dists_parties
print(f"correlation wave1_mapDistance -- wave2_mapDistance of pairs including self or a typical voter of party A and a typical voter of party B: {pearsonr(d["wave 1"], d["wave 2"]).statistic:.3f}")

fig, ax = plt.subplots(figsize=(4.,3.))
d = dists_parties.loc[dists_parties.dot1=="self"].dropna(subset=["wave 1","wave 2"])
d = dists_parties.dropna(subset=["wave 1","wave 2"])
hb = ax.hexbin(d["wave 1"], d["wave 2"], gridsize=20 if len(d.dot1.unique())==1 and d.dot1.unique()==["self"]  else 40 , 
               cmap="Reds", mincnt=1)
lims = [0, dists_parties[["wave 1","wave 2"]].max().max()]
ax.plot(lims, lims, ls="--", c="k", lw=1)
ax.set_xlabel("wave 1 map distance")
ax.set_ylabel("wave 2 map distance")
ax.set_aspect("equal")
ax.set_xticks([0,0.5,1])
ax.set_yticks([0,0.5,1])
ax.set_title("(self--voters)" if len(d.dot1.unique())==1 and d.dot1.unique()==["self"] else "(self--voters or voters--voters)" , x=1, ha="right", fontsize=8)
ax.text(0.5,0.95,f"r={pearsonr(d['wave 1'], d['wave 2']).statistic:.3f}", va="top", ha="center", transform=ax.transAxes)
plt.colorbar(hb, label="count", shrink=0.6)
plt.tight_layout()
plt.savefig(f"figs/reliability_{'self2Voter' if len(d.dot1.unique())==1 and d.dot1.unique()==['self'] else '_self2Voter2Voter'}.png", dpi=600)


fig, ax = plt.subplots(figsize=(5,5))
sns.scatterplot(data=dists_parties.dropna(subset=["wave 1","wave 2"]), 
                 x="wave 1", y="wave 2", alpha=0.1, size=1, ax=ax, hue="dot2")
lims = [0, dists_parties[["wave 1","wave 2"]].max().max()]
ax.plot(lims, lims, ls="--", c="gray", label="identity")
ax.set_xlabel("Pixel distance (Wave 1)")
ax.set_ylabel("Pixel distance (Wave 2)")
ax.set_title("Test-retest reliability of pixel_dist")
ax.legend()



d = dists_parties.copy()
d["abs_diff"] = (d["wave 2"] - d["wave 1"]).abs()
mad = d["abs_diff"].mean()
scale_range = dists_parties[["wave 1","wave 2"]].max().max()
print(f"Mean absolute difference: {mad:.1f} px ({mad/scale_range:.1%} of max scale)")


fig, ax = plt.subplots(figsize=(5,5))
hb = ax.hexbin(sub_valid["wave 1"], sub_valid["wave 2"], gridsize=40, 
               cmap="viridis", mincnt=1)
lims = [0, dists_parties[["wave 1","wave 2"]].max().max()]
ax.plot(lims, lims, ls="--", c="white", lw=1)
plt.colorbar(hb, label="count")