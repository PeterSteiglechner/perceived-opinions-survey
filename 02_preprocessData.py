# %%
import numpy as np
import json
import os

from consts import *
import pandas as pd
from itertools import combinations
import warnings
from pandas.errors import PerformanceWarning

warnings.filterwarnings("ignore", category=PerformanceWarning)

df = pd.read_csv("processed_data/2026-05-13_allBilendiData.csv")
# %%

resdf = df[
    ["bilendi_id", "wave"]
    + ["excl_double", "excl_NA", "excl_training", "excl_time", "excl_noMetaData", "excl_wave1Only"]
].copy().rename(columns={"bilendi_id":"id"})
assert(resdf[["excl_double", "excl_NA", "excl_time"]].sum().sum()==0)

duplicates = resdf[resdf.duplicated(subset=["id", "wave"], keep=False)]
assert(len(duplicates)==0)
# print(
#     f"removed {np.sum(duplicates['excl_double'].astype(int))} data points from "
#     f"{len(duplicates['id'].unique())} participants. I keep their first complete entry!"
# )
# resdf_unique = resdf.loc[~resdf.excl_double.astype(bool)]


# ----------------------------------------------
# -------    TIME
# ----------------------------------------------

resdf["t_completed"] = pd.to_datetime(df["player.t_on_success"], unit="s")
df["player.t_firstDotMoved"] = df["player.t_firstDotMoved"].replace({-1: np.nan})
df["player.t_after_first_pair"] = df["player.t_after_first_pair"].replace({-1: np.nan})

tcols = {
    "time_total": ("t_on_ownOpinion", "t_on_success"),
    "time_trainingGame": ("t_on_practiceGame", "t_on_practice"),
    "time_training": ("t_on_practice", "t_on_map"),
    "time_spam": ("t_on_map", "t_on_satisfaction"),
    "time_spam18dots": ("t_firstDotMoved", "t_on_satisfaction"),
    "time_pairwise": ("t_on_pairwise", "t_on_toc4"),
    "time_pairwise18pairs": ("t_after_first_pair", "t_on_toc4"),
}

for tcol, (t0, t1) in tcols.items():
    resdf[tcol] = (df[f"player.{t1}"] - df[f"player.{t0}"]).copy()  # in s

# ----------------------------------------------
# -------    META
# ----------------------------------------------

resdf["gender"] = df["S1"].map({"Männlich": "m", "Weiblich": "f", "Divers": "d"})
resdf["age"] = df["S2"]
resdf["party_vote"] = pd.Categorical(df["party_vote"], categories=parties_vote)
resdf["region"] = pd.Categorical(df["S3"])

# ----------------------------------------------
# -------    Identity
# ----------------------------------------------

resdf["party_close"] = pd.Categorical(df["player.feel_closest_party"], parties_full)
resdf["lr"] = (df["player.lrscale"] + MAX_DIPOLE_SLIDER) / (2 * MAX_DIPOLE_SLIDER)
resdf["polInterest"] = df["player.political_interest"] / MAX_DEFAULTSLIDER
resdf["polFrequency"] = df["player.political_discussion"] / MAX_DEFAULTSLIDER


# ----------------------------------------------
# -------    Opinions, Party Opinions, References
# ----------------------------------------------

resdf["n_contacts"] = np.sum(
    (~(df[[f"player.reference{k}" for k in range(1, MAX_NCONTACS + 1)]]).isna()), axis=1
)

opinion_cols = {
    q: [f"own__{q}", f"own2__{q}"]
    + [f"reference{k}__{q}" for k in range(1, MAX_NCONTACS + 1)]
    + [f"{p}__{q}" for p in partiesVars]
    for q in questions_sc
}
for q in questions_sc:
    for opCol in opinion_cols[q]:
        df[f"player.{opCol}"] = df[f"player.{opCol}"].replace({-999: np.nan})


def get_opinionStd_social_circle(x, q):
    return np.std(
        x[[f"player.reference{k}__{q}" for k in range(1, int(x["player.n_references"]) + 1)]]
        / MAX_OPINIONSLIDER, 
    )


for q, qq in zip(questions_sc, qs):
    resdf[f"std_socialCircle_ops_{q}"] = df.apply(
        get_opinionStd_social_circle, args=(q,), axis=1
    )


for q, qq in zip(questions_sc, qs):
    resdf[f"x_{'self'}_{q}"] = (df[f"player.own2__{q}"]) / MAX_OPINIONSLIDER
    resdf[f"first_x_{'self'}_{q}"] = (df[f"player.own__{q}"]) / MAX_OPINIONSLIDER

resdf["zeroVar_hint"] = df["player.nudged_for_variation"].astype(bool)

for q, qq in zip(questions_sc, qs):
    for p, pp in zip(partiesVars, parties_full):
        resdf[f"x_{p}_{q}"] = (df[f"player.{p}__{q}"]) / MAX_OPINIONSLIDER


resdf["treatment_wave2"] = np.nan
resdf.loc[resdf.wave == 2, "treatment_wave2"] = df.loc[
    df.wave == 2, "player.treatmentCondition"
].astype(int)


# ----------------------------------------------
# -------    Training
# ----------------------------------------------
def get_dist(x, a, b, v):
    pos = json.loads(x["player." + v])
    a_pos = [
        np.array([p["x"], p["y"]]) for p in pos if p["varname"].replace(" ", "") == a
    ]
    b_pos = [
        np.array([p["x"], p["y"]]) for p in pos if p["varname"].replace(" ", "") == b
    ]
    if len(a_pos) and len(b_pos):
        return np.linalg.norm(a_pos[0] - b_pos[0]) / np.sqrt(
            MAX_PIXELPOS**2 + MAX_PIXELPOS**2
        )
    else:
        return np.nan


for v1, v2 in combinations(practice_game_dots, 2):
    resdf[f"dist_game_{v1[0]}-{v2[0]}"] = df.apply(
        get_dist, args=(v1, v2, "positionsGame"), axis=1
    )

for v1, v2 in combinations(practice_training_dots, 2):
    resdf[f"dist_practice_{v1[0]}-{v2[0]}"] = df.apply(
        get_dist, args=(v1, v2, "positionsTest"), axis=1
    )

resdf["attemptsPractice"] = df["player.attemptPractice"].astype(int)
resdf["passed_practice_sanity"] = np.nan
resdf.loc[df["player.isTrainingPassed"].astype(bool), "passed_practice_sanity"] = df.loc[df["player.isTrainingPassed"].astype(bool), "player.practice_sanity_check_correct"]

resdf.loc[resdf.excl_training, "attemptsPractice"] = -999
resdf["attemptsPractice"] = pd.Categorical(
    resdf["attemptsPractice"], categories=list(range(MAX_PRACTICEATTEMPTS + 1)) + [-999]
)  # -999 not passed

# ----------------------------------------------
# -------    SPAM
# ----------------------------------------------
# include average distance
resdf["average_pixel_dist"] = df.apply(
    lambda x: np.mean(
        [get_dist(x, a, b, "positions") for a, b in combinations(peeps, 2) if not np.isnan(get_dist(x, a, b, "positions"))]
    ),
    axis=1,
)

resdf["average_pixel_dist_parties"] = df.apply(
    lambda x: np.mean(
        [
            get_dist(x, a, b, "positions")
            for a, b in combinations(peeps, 2)
            if (not "reference" in a) and (not "reference" in b)
        ]
    ),
    axis=1,
)


# ----------------------------------------------
# -------    Satisfaction and Evaluation
# ----------------------------------------------
resdf[f"map_satisfaction"] = df[f"player.satisfaction"] / MAX_DEFAULTSLIDER

resdf[f"mappingEasier"] = df[f"player.mappingEasier"] / MAX_DIPOLE_SLIDER

resdf[f"mappingEnjoy"] = df[f"player.mappingEnjoy"] / MAX_DIPOLE_SLIDER


# ----------------------------------------------
# -------    Polarisation
# ----------------------------------------------
for q in questions_sc:
    resdf[f"P_{q}"] = df[f"player.how_polarised_{q}"] / MAX_DEFAULTSLIDER
resdf[f"P_tot"] = df[f"player.how_polarised"] / MAX_DEFAULTSLIDER


# ----------------------------------------------
# -------    Issue importance
# ----------------------------------------------
for q in questions_sc:
    resdf[f"w_{q}"] = df[f"player.importance_{q}"] / MAX_DEFAULTSLIDER

# ----------------------------------------------
# -------    Sympathy
# ----------------------------------------------
for p in partiesVars:
    resdf[f"sym_{p}"] = np.nan
    resdf.loc[resdf.wave == 2, f"sym_{p}"] = (
        df.loc[df.wave == 2, f"player.{p}_sympathy"] + MAX_DIPOLE_SLIDER
    ) / (2 * MAX_DIPOLE_SLIDER)

# %%
columns = resdf.columns.to_list()


demo_cols = ["gender", "age", "party_vote", "region"]  # participant-level constants
excl_cols = ["excl_double", "excl_NA", "excl_training", "excl_time", "excl_wave1Only"]

value_cols = [
    c for c in resdf.columns if c not in ["id", "wave"] + demo_cols + excl_cols
]

df_wide = resdf.pivot(index="id", columns="wave", values=value_cols)
df_wide.columns = [f"wave{wave}_{var}" for var, wave in df_wide.columns]
df_wide = df_wide.reset_index()

# Join both demographics and exclusion flags — all are participant-level constants
static_cols = demo_cols + excl_cols
static_once = resdf.drop_duplicates("id").set_index("id")[
    static_cols
]
df_wide = df_wide.join(static_once, on="id")

if not os.path.isdir("processed_data/"):
    os.mkdir("processed_data/")
df_wide.to_csv(
    "processed_data/2026-06-19_data_processed_participant_pivot.csv", index=False
)
df_wide.to_excel(
    "processed_data/2026-06-19_data_processed_participant_pivot.xlsx", index=False
)
resdf.to_csv(
    "processed_data/2026-06-19_data_processed_participant.csv", index=False
)

resdf.to_excel(
    "processed_data/2026-06-19_data_processed_participant.xlsx", index=False
)


# %%


# %%
# ----------------------------------------------
# -------    DX
# ----------------------------------------------


# Helpers
def find_pair_index(pairs, a, b):
    for i, p in enumerate(pairs):
        pp = [p[0].replace(" ", ""), p[1].replace(" ", "")]
        if pp == [a, b] or pp == [b, a]:
            return i
    return None


def get_op(data_id_wave, p):
    return (
        data_id_wave[
            [f"player.{'own2' if p == 'self' else p}__{q}" for q in questions_sc]
        ]
    ).values / MAX_OPINIONSLIDER


dx = []
for w in [1, 2]:
    dfw = resdf.loc[resdf.wave == w]
    assert len(dfw["id"]) == len(dfw["id"].unique())
    for counter, (i, x) in enumerate(list(dfw.iterrows())):
        id = x["id"]
        data_id_wave = (
            df.loc[~df.excl_double.astype(bool)]
            .query(f"wave=={w} & bilendi_id=={id}")
            .iloc[0]
        )
        if counter % 100 == 0:
            print(f"{counter}", end="...")
        
        def get_opinionStd_social_circle(x, q):
                return np.std(
                    x[[f"player.reference{k}__{q}" for k in range(1, int(x["player.n_references"]) + 1)]]
                    / MAX_OPINIONSLIDER, 
                )
        std_socialCircle_ops = {"std_socialCircle_ops_": get_opinionStd_social_circle(data_id_wave, q) for q in questions_sc}
        
        for a, b in combinations(peeps, 2):
            row = {}
            row["id"] = id
            row["wave"] = w
            row["age"] = data_id_wave["S2"]
            row["gender"] = {"Männlich": "m", "Weiblich": "f", "Divers": "d", np.nan: np.nan}[data_id_wave["S1"]]
            party = data_id_wave["player.feel_closest_party"].replace(" ", "")
            row["party"] = party
            row["lr"] = (data_id_wave["player.lrscale"]/MAX_DIPOLE_SLIDER + 1) / 2
            row["dot1"] = a
            row["dot2"] = b
            assert b != "self"  #
            row["ingroupdummy"] = (a == "self") and (b == party)

            minpair = json.loads(data_id_wave["player.min_pair"])
            minpair = [minpair[0].replace(" ", ""), minpair[1].replace(" ", "")]
            row["minPixelDistance_pair_dummy"] = ([a, b] == minpair) or ([b, a] == minpair)
            maxpair = json.loads(data_id_wave["player.max_pair"])
            maxpair = [maxpair[0].replace(" ", ""), maxpair[1].replace(" ", "")]
            row["maxPixelDistance_pair_dummy"] = ([a, b] == maxpair) or ([b, a] == maxpair)

            dot1_ops = get_op(data_id_wave, a)
            dot2_ops =  get_op(data_id_wave, b)
            row.update(dict(zip([f"dot1_{q}" for q in questions_sc], dot1_ops)))
            row.update(dict(zip([f"dot2_{q}" for q in questions_sc], dot2_ops)))
            row.update(
                dict(
                    zip(
                        [f"deltaX_{q}" for q in questions_sc],
                        np.abs(dot1_ops - dot2_ops),
                    )
                )
            )

            assert type(data_id_wave["player.positions"]) == str
            dist = get_dist(data_id_wave, a, b, "positions")
            row["pixel_dist"] = dist

            pairs = json.loads(data_id_wave["player.pairSequence"])
            match = find_pair_index(pairs, a, b)

            if match is None:
                row["pairwise_similarity"] = np.nan
            else:
                row["pairwise_similarity"] = (
                    data_id_wave[f"player.similarityPair{match + 1}"]
                    / MAX_DEFAULTSLIDER
                )

            if (
                w == 2 and a == "self" and (b in partiesVars)
            ):
                row["sympathy"] = (data_id_wave[f"player.{b}_sympathy"]/MAX_DIPOLE_SLIDER + 1)/2
            else:  # not measured in wave 1
                row["sympathy"] = np.nan
                
            if a == "self" and "reference" in b:
                row["socialCloseness"] = (
                    data_id_wave[f"player.{b}_socialCloseness"] / MAX_DEFAULTSLIDER
                )
            else:
                row["socialCloseness"] = np.nan

            row.update(std_socialCircle_ops)

            dx.append(row)
print(".done")
# %%
dxdf = pd.DataFrame(dx)
dxdf["treatment_wave2"] = False
dxdf.loc[(dxdf.wave==2) & (dxdf["id"].isin(resdf.loc[resdf.treatment_wave2==1, "id"].tolist())), "treatment_wave2"] = True 

dxdf_vlaidrows = dxdf.dropna(subset=["pixel_dist"])
if not os.path.isdir("processed_data/"):
    os.mkdir("processed_data/")
dxdf_vlaidrows.to_csv("processed_data/2026-06-19_data_processed_differences.csv", index=False)
dxdf_vlaidrows.to_excel("processed_data/2026-06-19_data_processed_differences.xlsx", index=False)

# %%

print(resdf.columns)
print(dxdf.columns)

# %%


# %%
