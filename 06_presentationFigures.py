#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from consts import *
import numpy as np
import scipy.stats as stats
from itertools import combinations
from matplotlib.patches import Patch
plt.rcParams.update({"font.size":9})
plt.rcParams.update({"figure.figsize":(16/2.54, 9/2.54)})
sns.set_style("ticks")
sns.set_context("notebook")
pd.set_option('display.float_format', '{:.3f}'.format)
#%%

df_p = pd.read_csv("processed_data/2026-06-19_data_processed_participant_withAllIssueWeights.csv")
df_diff = pd.read_csv("processed_data/2026-06-19_data_processed_differences_withAllIssueWeights.csv")


# %%
# 1. Demographics
wavecondition = "wave in [1]" 
print("Age: ", df_p.query(wavecondition)["age"].describe())
print("Gender: ", df_p.query(wavecondition)["gender"].value_counts().sort_values(ascending=False)/df_p.query("wave==1")["region"].count())
print("Region: ", df_p.query(wavecondition)["region"].value_counts().sort_values(ascending=False)/df_p.query("wave==1")["region"].count())

# %%
# 2. Party Identity/Vote
partyPrefsVote = df_p.query(wavecondition)["party_vote"].value_counts().sort_values(ascending=False)
partyPrefsClose = df_p.query(wavecondition)["party_close"].value_counts().sort_values(ascending=False)
partyPrefs = (
    partyPrefsVote.to_frame("vote")
    .join(partyPrefsClose.to_frame("close"), how="outer")
    .fillna(0).astype(int)
    .sort_values("vote", ascending=False)
)
print("Party Vote/Close: ", partyPrefs)
print("Party Vote/Close Fraction: ",partyPrefs/ partyPrefs.sum(axis=0))
# %%
# 3. Opinions
op_cols = [f"x_self_{q}" for q in questions_sc]
fig, axes = plt.subplots(2, 3, figsize=(7, 3.5), sharey=False, sharex=True)
axes = axes.flatten()

for ax, col, label in zip(axes, op_cols, questions_sc):
    sns.histplot(
        df_p.query(wavecondition)[col].dropna(),
        ax=ax,
        bins=11,
        binrange=(-1, 1),
        # kde=True,
        stat="density",      # makes KDE and bars scale together properly
        color="cornflowerblue",
        alpha=0.5,
    )
    ax.set_title(label)
    ax.set_xlim(-1, 1) 
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="x")

fig.tight_layout(pad=0.6)
plt.savefig("figs/ownOps.png")
# %%
fig, axs = plt.subplots(2, 3, figsize=(8, 3.5), sharey=False, sharex=True)
axes = axs.flatten()

for ax, q in zip(axes, questions_sc):
    op_cols_p = [f"x_{r}_{q}" for r in partiesVars]
    sns.kdeplot(
        df_p.query(wavecondition)[op_cols_p].dropna().melt().replace(dict(zip(op_cols_p, partiesVars))),
        x="value",
        hue="variable",
        palette=party_cmap,
        ax=ax,
        fill=True,
        alpha=0.1,
        lw=2,
        legend=False,   # suppress all in-plot legends
    )

    ax.set_title(q)
    ax.set_xlim(-1, 1)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="x")

# Build legend handles manually from the last axis
handles = [
    Patch(facecolor=party_cmap[p], alpha=0.4, label=p)
    for p in partiesVars
]
fig.legend(
    handles, partiesVars,
    loc="center left",
    bbox_to_anchor=(0.82, 0.5),   # just outside the right edge
    frameon=False,
)

fig.tight_layout(pad=0.6)
fig.subplots_adjust(right=0.82)  # make room for the legend
plt.savefig("figs/partyOps.png")
#%%
ops = df_p.melt(id_vars=["id", "wave"], value_vars=[f"x_self_{q}" for q in questions_sc], value_name="opinion", var_name="question").replace(dict(zip([f"x_self_{q}" for q in questions_sc], questions_sc)))
stds = df_p.melt(id_vars=["id", "wave"], value_vars=[f"std_socialCircle_ops_{q}" for q in questions_sc], value_name="std social circle opinions", var_name="question").replace(dict(zip([f"std_socialCircle_ops_{q}" for q in questions_sc], questions_sc)))
ops_std = ops.merge(stds, on=["id", "wave", "question"])
ops_std["|opinion|"] = abs(ops_std["opinion"])


def p_to_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

g = sns.lmplot(
    ops_std,
    y="std social circle opinions",
    x="|opinion|",
    hue="question",
    height=3.5,
    aspect=1.2,
    x_jitter=0.02,
    scatter_kws={"s": 1, "alpha": 0.15,},
    line_kws={"linewidth": 1.3},
    legend=False,  # we'll draw a compact one manually
)

ax = g.axes[0, 0]
palette = sns.color_palette(n_colors=len(questions_sc))

slopes, stars_list = [], []
for question in questions_sc:
    subset = ops_std[ops_std["question"] == question].dropna(
        subset=["std social circle opinions", "|opinion|"]
    )
    slope, _, _, p, _ = stats.linregress(
        subset["std social circle opinions"],
        subset["|opinion|"]
    )
    slopes.append(slope)
    stars_list.append(p_to_stars(p))
qmap = dict(zip(questions_sc, ["climate concern", "gay marriage", "migrant rights", "inequality", "digital regulation", "east germans"]))
handles = [
    plt.Line2D([0], [0], color=palette[i], linewidth=2,
               label=f"{qmap[q]}     β={slope:.2f}{stars}")
    for i, (q, slope, stars) in enumerate(zip(questions_sc, slopes, stars_list))
]

ax.legend(handles=handles, fontsize=7, frameon=False,
          loc="upper right", handlelength=1.2,
          handletextpad=0.4, labelspacing=0.3)

# ax.set_title(f"Issue weight vs. social circle opinion spread", fontsize=9, pad=8)
ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)  # reference line at 0

g.figure.tight_layout()
g.figure.savefig("figs/corr_absOpinion_stdSC.png", dpi=150, bbox_inches="tight")

# sns.lmplot(ops_std, hue="question", x="abs_opinion", y="std", scatter_kws={"s":1, "alpha":0.3})

# %%

# 5. Social Circle Opinions
stdSC_cols = [f"std_socialCircle_ops_{q}" for q in questions_sc]
fig, axes = plt.subplots(2, 3, figsize=(7, 3.5), sharey=True, sharex=True)
axes = axes.flatten()
df_p["treatment_wave2_str"] = df_p["treatment_wave2"].map({1.0:"T", 0.0:"C", np.nan: "w1"})
for ax, col, q in zip(axes, stdSC_cols, questions_sc):
    sns.histplot(
        df_p.query(wavecondition)[[col, "treatment_wave2_str"]].dropna(),
        x=col,
        ax=ax,
        bins=11,
        binrange=(0, 1.),
        kde=True,
        stat="density",      # makes KDE and bars scale together properly
        # color="orange",
        hue="treatment_wave2_str",
        palette={"T":"magenta", "C":"brown", "w1":"orange"},
        alpha=0.3,
        legend=False
    )
    mean_T = df_p.query(wavecondition+" and treatment_wave2_str=='T'")[col].dropna().mean()
    mean_C = df_p.query(wavecondition+" and treatment_wave2_str=='C'")[col].dropna().mean()
    mean_w1 = df_p.query(wavecondition+" and treatment_wave2_str=='w1'")[col].dropna().mean()
    for meanval, col, y, label in zip([mean_T, mean_C, mean_w1], ["magenta", "gold", "orange"], [0.8,0.65, 1], ["T", "C", "{w1}"]):
        ax.vlines(
            meanval,
            0,1.8,
            color=col,
            linestyles="--"
        )
        s = '{SD}'
        ax.text(meanval-0.05, 2.5, rf"   $\overline{s}_{label}$:"+f" {meanval:.2f}", ha='left', va='top', color=col)
    

    ax.set_title(q)
    ax.set_xlim(0, 1.) 
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="x")
    # ax.text( df_p.query(wavecondition)[col].dropna().mean(), 2.8, f" mean: {df_p.query(wavecondition)[col].dropna().mean():.2f}", ha='left', va='top')

fig.suptitle("social circle std")
fig.tight_layout(pad=0.6)
plt.savefig("figs/socialvarOps.png")
# %%
# 7. VIF and Correlation Number

#%%
# 8. Plot a few maps 

#%%
# 9. Pairwise Similarity vs. Political Distance (Map)
fig, axes = plt.subplots(1,2, figsize=(5, 2.), sharey=False, sharex=True)
for ax, col in zip(axes, ["pairwise_similarity", "pixel_dist"]):
    sns.histplot(df_diff.query(wavecondition)[col], color="purple" if "sim" in col else "steelblue", bins=11, binrange=[-0.01,1.01], ax=ax, kde=True, linewidth=3, alpha=0.3)
    ax.set_yticks([])
    ax.set_ylabel(r"# responses" if "sim" in col else "")
    ax.set_xlabel("pairwise similarity" if "sim" in col else "map distance")
plt.tight_layout()
plt.savefig("figs/dists_pairwise_mapDist.png", dpi=600)
# %%
# 10. Party -- Self

fig, axes = plt.subplots(1,3, figsize=(7, 2.), sharey=False, sharex=True)
for ax, col in zip(axes, ["pairwise_similarity", "pixel_dist", "sympathy"]):
    sns.histplot(df_diff.query(f"wave==2 and dot1=='self' and dot2 in {partiesVars}")[col], color="purple" if "sim" in col else ("steelblue" if "pix" in col else "tomato"), bins=11, binrange=[-0.01,1.01], ax=ax, kde=True, linewidth=3, alpha=0.3)
    ax.set_yticks([])
    ax.set_ylabel("responses" if "sim" in col else "")
    ax.set_xlabel("pairwise similarity" if "sim" in col else ("map distance" if "pix" in col else "likability"))
axes[-1].set_title("only 'self' vs. voter evaluations", y=1.0, x=0.96, va="bottom", ha="right")
plt.tight_layout()
plt.savefig("figs/dists_pairwise_mapDist_sym.png", dpi=600)
# %%
# 11. Correlations
wavecondition = "wave in [1,2]"
corr_cols = [f"corrP_alpha_{q}" for q in questions_sc]
fig, axes = plt.subplots(2, 3, figsize=(8, 3.5), sharey=True, sharex=True)
axes = axes.flatten()
for ax, col, q in zip(axes, corr_cols, questions_sc):
    sns.histplot(
        df_p.query(wavecondition)[[col, "treatment_wave2_str"]],
        x = col,
        ax=ax,
        bins=14,
        binrange=(-0.2, 1.2),
        hue="treatment_wave2_str",
        kde=True,
        stat="density",      # makes KDE and bars scale together properly
        palette={"T":"tomato", "C":"k", "w1":"grey"},
        alpha=0.5,
        legend=False,
    )
    mean_T = df_p.query(wavecondition+" and treatment_wave2_str=='T'")[col].dropna().mean()
    mean_C = df_p.query(wavecondition+" and treatment_wave2_str=='C'")[col].dropna().mean()
    mean_w1 = df_p.query(wavecondition+" and treatment_wave2_str=='w1'")[col].dropna().mean()

    for meanval, col, y, label in zip([mean_T, mean_C, mean_w1], ["tomato", "k", "grey"], [0.97,0.75, 0.55], ["T", "C", "{w1}"]):
        ax.vlines(
            meanval,
            0,1,
            color=col,
            linestyles="--"
        )
        s = r'{r_{d,\Delta X}}'
        ax.text(meanval, ax.get_ylim()[1]*y, rf"   $\overline{s}({label})$:"+f" {meanval:.2f}", ha='left', va='top', color=col)
    ax.set_title(q)
    ax.set_xlim(-0.2, 1.2) 
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="x")
    # ax.text( df_p.query(wavecondition+" and treatment_wave2==0")[col].dropna().median(), 2., f" median: {df_p.query(wavecondition+' and treatment_wave2==0')[col].dropna().median():.2f}", ha='left', va='top')
fig.suptitle("Correlation `map distance' with `opinion distance'")
fig.tight_layout(pad=0.6)
plt.savefig("figs/corrP.png")


#%%
# 12. Correlations over Social Circle Variance 
wavecondition="wave in [1,2]"
corrs =df_p.query(wavecondition)[["id", "wave", "treatment_wave2_str"] +corr_cols].melt(id_vars=["id", "wave", "treatment_wave2_str"], value_name="issue weight (correlation)", var_name="question").replace(dict(zip(corr_cols, questions_sc)))
stds = df_p.query(wavecondition)[["id", "wave", "treatment_wave2_str"] +stdSC_cols].melt(id_vars=["id", "wave", "treatment_wave2_str"], value_name="std social circle opinions", var_name="question").replace(dict(zip(stdSC_cols, questions_sc)))
corrStd_df = corrs.set_index(["id", "wave","treatment_wave2_str", "question"]).join(stds.set_index(["id", "wave", "question", "treatment_wave2_str"])).reset_index()

def p_to_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

g = sns.lmplot(
    corrStd_df,
    x="std social circle opinions",
    y="issue weight (correlation)",
    hue="question",
    height=3.5,
    aspect=1.2,
    scatter_kws={"s": 1, "alpha": 0.15},
    line_kws={"linewidth": 1.3},
    legend=False,  # we'll draw a compact one manually
)

ax = g.axes[0, 0]
palette = sns.color_palette(n_colors=len(questions_sc))

slopes, stars_list = [], []
for question in questions_sc:
    subset = corrStd_df[corrStd_df["question"] == question].dropna(
        subset=["std social circle opinions", "issue weight (correlation)"]
    )
    slope, _, _, p, _ = stats.linregress(
        subset["std social circle opinions"],
        subset["issue weight (correlation)"]
    )
    slopes.append(slope)
    stars_list.append(p_to_stars(p))
qmap = dict(zip(questions_sc, ["climate concern", "gay marriage", "migrant rights", "inequality", "digital regulation", "east germans"]))
handles = [
    plt.Line2D([0], [0], color=palette[i], linewidth=2,
               label=f"{qmap[q]}     β={slope:.2f}{stars}")
    for i, (q, slope, stars) in enumerate(zip(questions_sc, slopes, stars_list))
]

ax.legend(handles=handles, fontsize=7, frameon=False,
          loc="upper right", handlelength=1.2,
          handletextpad=0.4, labelspacing=0.3)

# ax.set_title(f"Issue weight vs. social circle opinion spread", fontsize=9, pad=8)
ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)  # reference line at 0

g.figure.tight_layout()
g.figure.savefig("figs/corrP_stdSC.png", dpi=150, bbox_inches="tight")

# %%
# 13. Change in weights/corr/stdSC over two waves

k = "std_socialCircle_ops"
kname = r"SD"
# k="linear_alpha"
# kname = r"\alpha_{d,\Delta X}"
# k = "corrP_alpha"
# kname = r"r_{d,\Delta X}"
palette = {0.:"#444", 1.:"magenta"}
fig, axes = plt.subplots(5, 3, figsize=(7, 4.5), sharey="row", sharex=True, height_ratios=[1,2,0.2,1,2])
for ax in axes[2,:].flatten():
    ax.axis("off")
for ax, ax2, q in zip(axes[[1,4], :].flatten(), axes[[0,3],:].flatten(), questions_sc): 
    ax.axvline(0,color="k")
    ax2.axvline(0,color="k")
    alpha_change = df_p.pivot_table(index="wave", columns="id", values=f"{k}_{q}").diff(axis=0).query("wave==2").T.rename(columns={2:"diff"})
    alpha_change = alpha_change.join(df_p.query("wave==2")[["id", "treatment_wave2"]].set_index("id"),)
    sns.histplot(alpha_change, x="diff", hue="treatment_wave2", ax=ax, legend=False, palette=palette)
    sns.boxplot(alpha_change, x="diff", hue="treatment_wave2", ax=ax2, legend=False, fliersize=False, palette=palette)
    mean_diff = alpha_change.groupby("treatment_wave2")["diff"].mean().reset_index()
    sns.stripplot(x="diff", y=None, hue="treatment_wave2", data=mean_diff, ax=ax2, legend=False, dodge=True, s=4, marker="s", linewidth=1, edgecolor="k", palette=palette)
    ax.set_xlabel(fr"$\Delta {kname}{'{(w2-w1)}'}$")
    ax2.set_title(f"{q}")
    ax.set_xlim(-0.5, 0.5)

legend_handles = [
    Patch(facecolor=palette[0.], label="Control"),
    Patch(facecolor=palette[1.], label="Treatment"),
]

axes[-1,-1].legend(
    handles=legend_handles,
    labels=["Control", "Treatment"],
    fontsize=7,
    frameon=False,
    loc="upper right",
    handlelength=1.2,
    handletextpad=0.4,
    labelspacing=0.3,
)
fig.tight_layout(h_pad=0)
plt.savefig(f"figs/{k}_Change_wave2.png", dpi=600)

# %%
# 14. Relation stdSC to corrP/issue weights
k = "corrP_alpha"
kname = r"r_{d,\Delta X}"
# k = "exp_alpha"
# kname = r"\alpha_{q}"
fig, axes = plt.subplots(2, 3, figsize=(7, 4.5), sharey=True, sharex=True, )
# for ax in axes[2,:].flatten():
#     ax.axis("off")
palette = {0.:"k", 1.:"magenta"}

    

for ax,  q in zip(axes.flatten(),  questions_sc): 
    alpha_change = df_p.pivot_table(index="wave", columns="id", values=f"{k}_{q}").diff(axis=0).query("wave==2").T.rename(columns={2:"diff"})
    std_change = df_p.pivot_table(index="wave", columns="id", values=f"std_socialCircle_ops_{q}").diff(axis=0).query("wave==2").T.rename(columns={2:"std_diff"})
    std_change = std_change.join(df_p.query("wave==2")[["id", "treatment_wave2"]].set_index("id"),)
    ddd = std_change.join(alpha_change)
    for trt, color in palette.items():
        subset = ddd[ddd["treatment_wave2"] == trt]

        sns.regplot(
            data=subset,
            x="std_diff",
            y="diff",
            ax=ax,
            scatter=False,
            color=color,
            label=None,
            line_kws={"linestyle":"--" if trt==1 else "-"}
        )

    sns.scatterplot(
        data=ddd,
        x="std_diff",
        y="diff",
        hue="treatment_wave2",
        ax=ax,
        legend=False,
        palette=palette,
        alpha=0.2,
        size=1,
    )
    ax.set_title(f"{q}")
    ax.set_xlabel(""); ax.set_ylabel("")
    # sns.scatterplot(ddd, y="diff", x="std_diff", hue="treatment_wave2", ax=ax, legend=False, palette=palette, size=2, alpha=0.2, lw=0.1, edgecolor="w")
    # sns.scatterplot(ddd, y="diff", x="std_diff", hue="treatment_wave2", ax=ax, legend=False, palette=palette, size=2, alpha=0.2, lw=0.1, edgecolor="w")
    # sns.boxplot(alpha_change, x="diff", hue="treatment_wave2", ax=ax2, legend=False, fliersize=False, palette=palette)
    # mean_diff = alpha_change.groupby("treatment_wave2")["diff"].mean().reset_index()
    # sns.stripplot(x="diff", y=None, hue="treatment_wave2", data=mean_diff, ax=ax2, legend=False, dodge=True, s=4, marker="s", linewidth=1, edgecolor="k", palette=palette)

    slopes, stars_list = [], []
    for trt, group in enumerate(["Control", "Treatment"]):
        subset = ddd[ddd["treatment_wave2"] == trt].dropna(subset=["diff", "std_diff"])
        slope, _, _, p, _ = stats.linregress(
            subset["std_diff"],
            subset["diff"]
        )
        slopes.append(slope)
        stars_list.append(p_to_stars(p))


    legend_handles = [
        plt.Line2D([0], [0], color=palette[i], linewidth=2,
                label=f"{group}     β={slope:.2f}{stars}")
        for i, (group, slope, stars) in enumerate(zip(["Control", "Treatment"], slopes, stars_list))
    ]

    ax.legend(
        handles=legend_handles,
        # labels=["Control", "Treatment"],
        fontsize=7,
        frameon=False,
        loc="upper right",
        handlelength=1.2,
        handletextpad=0.4,
        labelspacing=0.3,
    )


for ax in axes[-1,:]:
    ax.set_xlabel(fr"$\Delta \sigma_{'{sc}'}{'{(w2-w1)}'}$")
for ax in axes[:,0]:
    ax.set_ylabel(fr"$\Delta {kname}{'{(w2-w1)}'}$")

# legend_handles = [
#     Patch(facecolor=palette[0.], label="Control"),
#     Patch(facecolor=palette[1.], label="Treatment"),
# ]



for ax in axs.flatten():
    ax.axvline(0,color="k", zorder=-1)
    ax.axhline(0,color="k", zorder=-1)

fig.tight_layout(h_pad=0)
plt.savefig(f"figs/corr_varChange_{k}_change.png", dpi=600)


# %%
