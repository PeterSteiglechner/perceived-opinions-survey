# %%
import numpy as np
import os
import pandas as pd
import seaborn as sns
from consts import *

pre = "player."

WAVE_CONFIG = {
    1: {
        "a": ("wave1/data/survey_wave1_2026-01-20.csv", "1/14/2026  10:13 AM"),
        "b": ("wave1/data/survey_wave1_2026-02-20.csv", "1/20/2026  10:00 AM"),
    },
    2: {
        "a": ("wave2/data/survey_wave2_2026-02-20.csv", "2026-02-03 08:30"),
        "b": ("wave2/data/survey_wave2_2026-05-13.csv", "2026-02-20 23:00"),
    },
}

# ----------------------------------------------
# -------    Load Prescreener MetaData
# ----------------------------------------------
meta = pd.read_spss(
    "wave1/data/DE_155853 Complexity Science Hub Residents of Germany_final/"
    "DE_ 155853 Complexity Science Hub Residents of Germany.sav"
)
meta["bilendi_id"] = meta.ID
meta["S4"] = meta["S4"].cat.reorder_categories(CATEGORY_ORDER)
meta["party_vote"] = meta["S4"].map(VOTING_PREF_MAP)
print(f"Meta data: len {len(meta)}")

# ----------------------------------------------
# -------    Load otree data
# ----------------------------------------------
df_dict = {}
for wave, cfg in WAVE_CONFIG.items():
    aname, starttime_1 = cfg["a"]
    bname, starttime_2 = cfg["b"]
    nan_vals = {"test": np.nan, "Test": np.nan, "noID": np.nan}
    data = []
    for filename, starttime in zip([aname, bname], [starttime_1, starttime_2]):
        dd = (
            pd.read_csv(filename, low_memory=False)
            .replace(nan_vals)
            .dropna(subset=[pre + "bilendi_id"])
        )
        print(f"wave: {wave}, (n={len(dd)})", end="; ")
        dd["participant.time_started_utc"] = pd.to_datetime(
            dd["participant.time_started_utc"]
        )
        dd[pre + "recontact"] = dd[pre + "recontact"].astype(bool)
        dd[pre + "completed"] = dd[pre + "completed"].map(
            {1: True, 0: False, np.nan: False}
        )
        dd[pre + "bilendi_id"] = dd[pre + "bilendi_id"].astype(int).astype(str)

        if wave == 1:
            dd[pre + "allpositionsTest"] = np.nan

        dd = dd.loc[dd["participant.time_started_utc"] >= pd.to_datetime(starttime)]
        dd_completed = dd.loc[dd[pre + "completed"]]
        print(f" len after filtering: n={len(dd)}, and completed {len(dd_completed)}")
        data.append(dd_completed)

    df = pd.concat(data).reset_index(drop=True)
    df = df.rename(columns={pre + "bilendi_id": "bilendi_id"})
    print(f"combined: n={len(df)}, unique IDs={df.bilendi_id.nunique()}")

    df_merged = pd.merge(meta, df, on="bilendi_id", how="right")
    df_merged["wave"] = wave
    df_dict[wave] = df_merged

df = pd.concat(df_dict.values()).reset_index(drop=True).copy()
print("Total by wave:\n", df["wave"].value_counts())

# ----------------------------------------------
# DROP: (1) Duplicates, (2) Speeders, (3) NA in ops
# ----------------------------------------------
df["excl_double"] = False
for wave in [1, 2]:
    mask = df.wave == wave
    dupes = df.loc[mask, "bilendi_id"].duplicated(keep="first")
    df.loc[mask, "excl_double"] = dupes.values

ownops = pd.isna(
    df[[pre + f"own2__{q}" for q in questions_sc]].replace({-999.0: np.nan})
).all(axis=1)
df["excl_NA"] = ownops

df["excl_training"] = (df[pre + "attemptPractice"] >= 5) & ~df[
    pre + "isTrainingPassed"
].astype(bool)

tot_time = (df["player.t_on_success"] - df["player.t_on_ownOpinion"]) / 60
df["excl_time"] = (tot_time < tot_time.median() / 3).values

df["excluded"] = df["excl_time"] | df["excl_NA"] | df["excl_double"]

valid_w1 = set(df.loc[(df.wave == 1) & ~df["excluded"], "bilendi_id"])
valid_w2 = set(df.loc[(df.wave == 2) & ~df["excluded"], "bilendi_id"])
df["excl_wave1Only"] = df["bilendi_id"].isin(
    {p for p in valid_w1 if p not in valid_w2}
)  # & ~df["excluded"]


print("Total:", df["wave"].value_counts().to_dict())
print(
    "Exclusion counts:",
)
display(
    df[["excl_double", "excl_time", "excl_NA", "wave"]]
    .value_counts()
    .reset_index()
    .sort_values("wave")
)
display(df[["excluded", "wave"]].value_counts().reset_index().sort_values("wave"))

print(
    "valid in wave1 but not in wave 2: ",
    df[["wave", "excl_wave1Only"]].value_counts().to_dict(),
)
if not os.path.isdir("processed_data/"):
    os.mkdir("processed_data/")
df.to_csv("processed_data/2026-05-13_allBilendiData.csv", index=False)

# %%
