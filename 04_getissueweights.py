# %% 
# ---
# title: extracting issue weights (fits) and correlations from map 
# author: Peter Steiglechner
# date: 19.06.2026
# ---

import numpy as np
from numpy.linalg import lstsq, svd
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# %% Configuration

questions_sc = [
    "climate_concern",
    "gay_marriage",
    "rights_indep_integration",
    "econ_inequality",
    "regulate_internet",
    "east_germans"
]
delta_cols = [f"deltaX_{q}" for q in questions_sc]

n_issues = len(delta_cols)

parties = ["Left Party", "BSW", "Green Party", "SPD", "FDP", "CDU/CSU", "AfD",
           "No party", "Other party", "Refuse to say/No answer"]
partiesVars = ["LeftParty", "BSW", "GreenParty", "SPD", "FDP", "CDU/CSU", "AfD"]


# %% # Load Data

df_partic = pd.read_csv("processed_data/2026-06-19_data_processed_participant.csv")

inds_bothwaves = (
    df_partic.groupby("id")
    .size()
    .reset_index(name="n")
    .query("n == 2")["id"]
    .values
)

df_diff = pd.read_csv("processed_data/2026-06-19_data_processed_differences.csv")

print(f"df_partic rows:        {len(df_partic)}")
print(f"df_partic unique ids:  {df_partic['id'].nunique()}")
print(f"df_diff unique ids:    {df_diff['id'].nunique()}")


# %% # Analysis Theory
# 
# ## Inferred variables
#
# The key variables we are interested in are how much weight participants implicitly
# assign to belief differences on each of the six issues when placing individuals
# on the political map. We infer these weights, α_{n,q}, from the participant n's
# responses by comparing the map distances d_{n,ij} of individuals i and j with
# their belief differences |x_{n,i,q} - x_{n,j,q}| on all issues q as reported
# by the participant. In particular, we model the map distances between dots i and j
# for participant n as a function of their belief differences on issues q:
#
#   d_{n,ij} = f_n( g_n( |x_{n,i,q} - x_{n,j,q}| ) )
#
# g_n aggregates belief differences on the six items q into a normalised
# 'overall belief distance', using a weighted L1-norm (Manhattan distance):
#
#   g_n(...) = sum_q  α_{n,q} * |x_{n,i,q} - x_{n,j,q}|
#
# f_n is a linear or exponential kernel translating overall belief distance
# into a pixel distance d_{n,ij} (normalised to [0,1]):
#
#   Linear:      f_n(x) = a_n + b_n * x
#   Exponential: f_n(x) = 1 - exp(-b_n * x^c_n)


# %% # Helper functions

def weighted_l1(deltas: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Weighted L1 norm (g_n). deltas: (n_pairs, n_issues), alpha: (n_issues,)"""
    return deltas @ alpha


def linear_kernel(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * x


def exp_kernel(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """a = reach or scaling (b_n), b = sharpness (c_n)"""
    return 1 - np.exp(-a * (x ** b))


def make_loss(deltas: np.ndarray, pixel_dist: np.ndarray,
              kernel: str = "linear", lam: float = 0.0):
    """
    Returns a loss function (SSE + optional entropy penalty) over params:
      params[:n_issues]    = alpha (issue weights, should sum to 1)
      params[n_issues]     = param1 (a: intercept or reach)
      params[n_issues + 1] = param2 (b: slope or exponent)
    """
    def loss(params):
        alpha = params[:n_issues]
        a     = params[n_issues]
        b     = params[n_issues + 1]

        g = weighted_l1(deltas, alpha)

        if kernel == "linear":
            pred = linear_kernel(g, a, b)
        else:
            pred = exp_kernel(g, a, b)

        sse             = np.sum((pixel_dist - pred) ** 2)
        entropy_penalty = -lam * np.sum(np.log(alpha + 1e-6))
        return sse + entropy_penalty

    return loss


def compute_diagnostics(deltaX: np.ndarray) -> dict:
    """
    Compute VIF per column and condition number of the (scaled) matrix.
    Mirrors the R compute_diagnostics() function.
    deltaX: nr of participants x nr of issues
    """

    N, M = deltaX.shape
    vifs = {}

    for j, q in enumerate(questions_sc):
        y = deltaX[:, j]
        X = np.delete(deltaX, j, axis=1)

        # R² of regressing column j on all others
        X_  = np.column_stack([np.ones(N), X])
        coef, *_ = lstsq(X_, y, rcond=None)
        y_hat = X_ @ coef
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vifs[f"deltaX_{q}"] = 1 / (1 - r2) if r2 < 1 else np.inf

    # Condition number on columns with nonzero variance
    col_sd  = deltaX.std(axis=0)
    deltaX_ok  = deltaX[:, col_sd > 0]

    if deltaX_ok.shape[1] < 2:
        cond_number = np.nan
    else:
        mat_scaled = (deltaX_ok - deltaX_ok.mean(axis=0)) / deltaX_ok.std(axis=0)
        sv          = svd(mat_scaled, compute_uv=False)
        min_sv      = sv[sv > np.finfo(float).eps * sv.max()].min()
        cond_number = sv.max() / min_sv

    return {"vif": vifs, "condition_number": cond_number}




# %% Fit participant (diagnostics only — mirrors active R code)

def fit_participant(df_p: pd.DataFrame,
                    kernel: str = "linear",
                    n_starts: int = 10,
                    lam: float = 0.0,
                    verbose: bool = False) -> pd.DataFrame:
    deltas     = df_p[delta_cols].to_numpy(dtype=float)
    pixel_dist = df_p["pixel_dist"].to_numpy(dtype=float)
    loss       = make_loss(deltas, pixel_dist, kernel=kernel, lam=lam)
    diag       = compute_diagnostics(deltas)
    if kernel == "linear":
        bounds = [(0, 1)] * n_issues + [(-0.2,1), (0, 2)]
    else:
        bounds = [(0, 1)] * n_issues + [(0.1, 10), (0, 10)]
    constraints = [{"type": "eq", "fun": lambda p: p[:n_issues].sum() - 1}]

    all_results = []  # collect one DataFrame per valid start

    for i in range(n_starts):
        if i == 0:
            alpha0 = np.full(n_issues, 1 / n_issues)
        else:
            raw    = np.random.rand(n_issues)
            alpha0 = raw / raw.sum()
        if kernel == "linear":
            p1 = np.random.uniform(0, 0.5)   if i > 0 else 0.0
            p2 = np.random.uniform(0.0, 2.0) if i > 0 else 1.0
        else:
            p1 = np.random.uniform(0, 2)  if i > 0 else 1.0
            p2 = np.random.uniform(0.5, 3) if i > 0 else 1.0
        x0 = np.concatenate([alpha0, [p1, p2]])

        try:
            fit = minimize(
                loss, x0,
                method      = "SLSQP",
                bounds      = bounds,
                constraints = constraints,
                options     = {"ftol": 1e-9, "maxiter": 10_000}
            )
            alpha_candidate = fit.x[:n_issues]
            alpha_sum_ok    = abs(alpha_candidate.sum() - 1) < 1e-3
            alpha_valid     = np.all(alpha_candidate >= 0)

            if alpha_sum_ok and alpha_valid:
                params = fit.x
                df_start = pd.DataFrame({
                    "issue":            delta_cols,
                    "alpha":            params[:n_issues],
                    "vif":              [diag["vif"][c] for c in delta_cols],
                    "condition_number": diag["condition_number"],
                    "kernel":           kernel,
                    "converged":        fit.success,
                    "sse":              fit.fun,
                    "param1":           params[n_issues],
                    "param2":           params[n_issues + 1],
                    "param1_name":      f"{kernel}Kernel_param1",
                    "param2_name":      f"{kernel}Kernel_param2",
                    "i_start":          i,           # <-- start index
                    "is_best":          False,       # <-- flag best start later
                })
                all_results.append((fit.fun, df_start))

        except Exception as e:
            if verbose:
                print(f"  start {i:2d} failed: {e}")
            continue

    if not all_results:
        # Fallback
        print("WARNING: all starts failed — returning equal-weight fallback")
        params = np.concatenate([
            [np.nan] * n_issues, #np.full(n_issues, 1 / n_issues),
            [np.nan, np.nan] #if kernel == "linear" else [1.0, 4.0]
        ])
        df_fallback = pd.DataFrame({
            "issue":            delta_cols,
            "alpha":            params[:n_issues],
            "vif":              [diag["vif"][c] for c in delta_cols],
            "condition_number": diag["condition_number"],
            "kernel":           kernel,
            "converged":        False,
            "sse":              loss(params),
            "param1":           params[n_issues],
            "param2":           params[n_issues + 1],
            "param1_name":      f"{kernel}Kernel_param1",
            "param2_name":      f"{kernel}Kernel_param2",
            "i_start":          -1,
            "is_best":          True,
        })
        return df_fallback

    # Mark the best start
    best_idx = min(range(len(all_results)), key=lambda k: all_results[k][0])
    all_results[best_idx][1]["is_best"] = True

    return pd.concat([df for _, df in all_results], ignore_index=True)
print("function fit_participant check")


# %% Run for all participants × both kernels

np.random.seed(2)

records = []
for (pid, wave), group in df_diff.groupby(["id", "wave"]):
    for kernel in ["linear", "exp"]:
        res = fit_participant(group, kernel=kernel, lam=0.0, n_starts=10)
        res["id"]   = pid
        res["wave"] = wave
        records.append(res)

results = pd.concat(records, ignore_index=True)
print("... fitting done.")



#%%
# store results
results.to_csv("processed_data/fits_allweights_vif_10starts.csv")
# %% Analyse VIF

vif_summary = (
    results.loc[results.is_best]
    .groupby(["id", "wave"])
    .agg(
        max_vif  =("vif", "max"),
        mean_vif =("vif", "mean"),
        any_high =("vif", lambda x: (x > 10).any())
    )
    .reset_index()
)

valid_ids = vif_summary.loc[vif_summary.max_vif<10, ["wave", "id"]]
valid_ids["valid"] = True
print(vif_summary.head())
resultsV = results.merge(valid_ids, on=["wave", "id"], how="left")
(vif_summary.max_vif<10).value_counts()

#%%
sns.boxplot(resultsV.loc[resultsV.valid & resultsV.is_best], x="issue", y="alpha", hue="wave", palette="Set1", fliersize=0, saturation=1)
# sns.barplot(resultsV.loc[resultsV.is_best & resultsV.valid ], x="issue", y="alpha", hue="wave", alpha=0.1, palette="Set1")
sns.stripplot(resultsV.loc[resultsV.valid & resultsV.is_best], x="issue", y="alpha", hue="wave", marker="o", size=1, dodge=True, palette="Set1", edgecolor="w", linewidth=0.05)
sns.stripplot(resultsV.loc[resultsV.valid & resultsV.is_best].groupby("issue")["alpha"].mean().reset_index(), x="issue", y="alpha", marker="s", size=10, palette="Set1")

plt.ylim(-0.02, 0.4)
resultsV.loc[resultsV.valid & resultsV.is_best].groupby("issue")["alpha"].mean()

# %% Collect alphas — wide format (one row per participant × kernel)


alphas_wide = (
    results.loc[results.is_best]
    .pivot_table(index=["id", "wave", "kernel"], columns="issue", values="alpha")
    .reset_index()
)
alphas_wide.columns.name = None
alphas_wide.columns = [
    c if c in ["id", "wave", "kernel"] else f"alpha_{c}"
    for c in alphas_wide.columns
]

# Convergence summary
print(results.loc[results.is_best].groupby(["kernel", "converged"]).size().unstack(fill_value=0))


# %% Attach alphas to dataframes
# (also requires the optimisation block to be active)

alphas_join = (
    alphas_wide
    .merge(
        results.loc[results.is_best][["id", "wave", "kernel", "sse", "converged",
                 "param1", "param2"]].drop_duplicates(),
        on=["id", "wave", "kernel"]
    )
)

# Pivot kernels wide
alphas_join = alphas_join.pivot_table(
    index=["id", "wave"],
    columns="kernel",
    values=[c for c in alphas_join.columns if c not in ["id", "wave", "kernel"]],
    aggfunc="first"
).reset_index()
alphas_join.columns = [
    f"{b}_{a}" if b else a
    for a, b in alphas_join.columns
]
alphas_join.columns = alphas_join.columns.str.replace("alpha_deltaX_", "alpha_")

# Pick best kernel per participant × wave
alphas_join["best_kernel"] = np.where(
    alphas_join["linear_sse"] < alphas_join["exp_sse"],
    "linear", "exp"
)

# Join back
df_diff_with_alphas   = df_diff.merge(alphas_join,   on=["id", "wave"], how="left")
df_partic_with_alphas = df_partic.merge(alphas_join, on=["id", "wave"], how="left")


#%%



corrS_by_group = (
    df_diff
    .groupby(["id", "wave"])
    [[f"pixel_dist"] + [f"deltaX_{q}" for q in questions_sc]]
    .apply(lambda g: g.corr(method='spearman')["pixel_dist"][[f"deltaX_{q}" for q in questions_sc]])
    .reset_index()
    .rename(columns={f"deltaX_{q}": f"corrS_alpha_{q}" for q in questions_sc})
)

df_diff2 = df_diff_with_alphas.merge(corrS_by_group, on=["id", "wave"], how="left").copy()
df_diff2["sumCorrSAlpha"] = df_diff2[[f"corrS_alpha_{q}" for q in questions_sc]].sum(axis=1)

corrP_by_group = (
    df_diff
    .groupby(["id", "wave"])
    [[f"pixel_dist"] + [f"deltaX_{q}" for q in questions_sc]]
    .apply(lambda g: g.corr(method='pearson')["pixel_dist"][[f"deltaX_{q}" for q in questions_sc]])
    .reset_index()
    .rename(columns={f"deltaX_{q}": f"corrP_alpha_{q}" for q in questions_sc})
)

df_diff2 = df_diff2.merge(corrP_by_group, on=["id", "wave"], how="left").copy()
df_diff2["sumCorrPAlpha"] = df_diff2[[f"corrP_alpha_{q}" for q in questions_sc]].sum(axis=1)

# %%
df_p2 = df_partic_with_alphas.merge(corrS_by_group, on=["id", "wave"], how="left")
df_p2["sumCorrSAlpha"] = df_p2[[f"corrS_alpha_{q}" for q in questions_sc]].sum(axis=1).copy()
df_p2 = df_p2.merge(corrP_by_group, on=["id", "wave"], how="left")
df_p2["sumCorrPAlpha"] = df_p2[[f"corrP_alpha_{q}" for q in questions_sc]].sum(axis=1).copy()



#%%
df_p2.to_csv(
    "processed_data/2026-06-19_data_processed_participant_withAllIssueWeights.csv",
    index=False)
df_diff2.to_csv(
    "processed_data/2026-06-19_data_processed_differences_withAllIssueWeights.csv",
    index=False)


# %% # Visualisation
# ## Average alpha weights
# (requires optimisation to be active; commented out to match R)

# alpha_plot = (
#     alphas_wide
#     .melt(id_vars=["id", "wave", "kernel"],
#           var_name="issue", value_name="alpha")
#     .assign(issue=lambda d: d["issue"].str.replace("alpha_", ""))
# )
#
# alpha_summary = (
#     alpha_plot
#     .groupby(["kernel", "issue"])
#     .agg(mean=("alpha", "mean"), se=("alpha", lambda x: x.std() / np.sqrt(len(x))))
#     .reset_index()
# )
#
# fig, ax = plt.subplots(figsize=(8, 5))
# width = 0.35
# kernels = alpha_summary["kernel"].unique()
# issues  = alpha_summary["issue"].unique()
# x = np.arange(len(issues))
#
# for i, k in enumerate(kernels):
#     sub = alpha_summary.query("kernel == @k").set_index("issue").reindex(issues)
#     ax.barh(x + i * width, sub["mean"], width,
#             xerr=1.96 * sub["se"], label=k)
#
# ax.set_yticks(x + width / 2)
# ax.set_yticklabels(issues)
# ax.set_xlabel("Mean alpha")
# ax.set_title("Average inferred issue weights")
# ax.legend()
# plt.tight_layout()
# plt.show()


# %% # Predict distances
# (requires optimisation to be active; function shown for completeness)

def predict_distances(df_p: pd.DataFrame, fit_row: pd.Series) -> pd.DataFrame:
    alpha_cols = [c for c in fit_row.index if c.startswith("alpha_")]
    alpha      = fit_row[alpha_cols].to_numpy(dtype=float)
    deltas     = df_p[delta_cols].to_numpy(dtype=float)
    g          = deltas @ alpha

    if fit_row["kernel"] == "linear":
        pred = fit_row["param1"] + fit_row["param2"] * g
    else:
        pred = 1 - np.exp(-fit_row["param1"] * (g ** fit_row["param2"]))

    return pd.DataFrame({"observed": df_p["pixel_dist"].values, "predicted": pred})


# participant = "330717073703941"
# wave        = 2
# kernel      = "linear"
#
# one_fit = alphas_wide.query(
#     "kernel == @kernel and id == @participant and wave == @wave"
# ).iloc[0]
#
# one_df  = df_diff.query("id == @participant and wave == @wave")
# pred_df = predict_distances(one_df, one_fit)
#
# fig, ax = plt.subplots()
# ax.scatter(pred_df["observed"], pred_df["predicted"], alpha=0.6)
# ax.axline((0, 0), slope=1, color="red", linestyle="--")
# ax.set_aspect("equal")
# ax.set_xlabel("Observed pixel distance")
# ax.set_ylabel(f"Predicted distance ({kernel} kernel)")
# ax.set_title(f"Observed vs predicted\n(id {participant}, wave {wave}, {kernel})")
# plt.tight_layout()
# plt.show()


# %% # Analyse initial-condition dependence of weights
examples = results.loc[results.vif<5, ["id", "wave", "param1_name"]].sample(10)
x = np.linspace(0,1)
id, wave, kernel = (331246904848564, 1, "linearKernel_param1") #examples.iloc[0]
kernel2 = ("expKernel_param1" if "linear" in kernel else "linearKernel_param1") 
example = results.loc[(results["id"]==id) & (results["wave"]==wave) & (results["param1_name"] == kernel)]
for i in range(50):
    a,b = example.loc[example.i_start==i, ["param1", "param2"]].iloc[0]
    print(a,b)
    plt.plot(x, linear_kernel(x, a,b) if "linear" in kernel else exp_kernel(x, a, b))


exampleFull = results.loc[(results["id"]==id) & (results["wave"]==wave) & (results["param1_name"].isin([kernel, kernel2]))]

fig = plt.figure()
sns.barplot(exampleFull, x="issue", y="alpha", hue="kernel")
sns.swarmplot(exampleFull, x="issue", y="alpha", size=3, hue="kernel")
fig.autofmt_xdate()


# %%
# Pick a few participants
sample_ids = df_diff["id"].unique()[:5]

fig, axes = plt.subplots(len(sample_ids), 2, figsize=(10, 3 * len(sample_ids)))

for i, pid in enumerate(sample_ids):
    for j, wave in enumerate([1, 2]):
        ax = axes[i, j]
        grp = df_diff.loc[(df_diff["id"] == pid) & (df_diff["wave"] == wave)]
        if grp.empty:
            ax.set_visible(False)
            continue

        deltas     = grp[delta_cols].to_numpy(dtype=float)
        pixel_dist = grp["pixel_dist"].to_numpy(dtype=float)

        fitted_row = results.loc[
            (results["id"] == pid) & (results["wave"] == wave) &
            (results["kernel"] == "exp") & results["is_best"]
        ]
        if fitted_row.empty:
            ax.set_visible(False)
            continue

        alpha_fit  = fitted_row.set_index("issue")["alpha"].reindex(delta_cols).to_numpy()
        p1         = fitted_row["param1"].iloc[0]
        p2         = fitted_row["param2"].iloc[0]

        # Equal weights baseline
        equal_alpha = np.full(n_issues, 1 / n_issues)
        pred_equal  = exp_kernel(weighted_l1(deltas, equal_alpha), p1, p2)
        pred_fit    = exp_kernel(weighted_l1(deltas, alpha_fit),   p1, p2)

        ax.scatter(pred_equal, pixel_dist, label="equal α",  alpha=0.6, s=20)
        ax.scatter(pred_fit,   pixel_dist, label="fitted α", alpha=0.6, s=20, marker="x")
        ax.plot([0,1],[0,1], "k--", lw=1)  # identity line

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("predicted pixel_dist")
        ax.set_ylabel("actual pixel_dist")
        ax.set_title(f"id={pid}, wave={wave}")
        ax.legend(fontsize=7)

plt.tight_layout()
plt.show()
# %%
