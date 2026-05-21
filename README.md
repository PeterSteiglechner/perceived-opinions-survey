# Bilendi Survey Data Processing Pipeline

## Overview

This pipeline processes two-wave survey data from a Bilendi panel study on political opinion mapping among residents of Germany (Complexity Science Hub project). Participants completed a **Spatial Arrangement Mapping (SpAM)** task, placing "dots" representing themselves, social contacts, and party voters on a 2D map to reflect perceived political similarity. A subset also completed pairwise similarity ratings.

Data was collected in two waves (Wave 1: Jan–Feb 2026; Wave 2: Feb–May 2026).

---

## Pipeline Scripts

| Script | Description |
|---|---|
| `consts.py` | Shared constants: issue names, party lists, color maps, slider maxima |
| `01_createdatafile.py` | Merges raw oTree exports with Bilendi prescreener metadata; applies initial exclusions; saves combined CSV |
| `02_preprocessData.py` | Cleans and reshapes data into three analysis-ready CSVs (participant-level long format; participant-level wide/pivot format; pairwise long format) |
| `02b_plot_map.py` | Visualizes individual SpAM maps and overlays opinion coloring per issue |
| `03_data-overview.py` | Descriptive plots and sanity checks across all key variables |

---

## Input Data

### Raw oTree/Bilendi exports
Two CSV files per wave, corresponding to two recruitment batches:

| Wave | Batch | File | Start time filter |
|---|---|---|---|
| 1 | a | `wave1/data/survey_wave1_2026-01-20.csv` | 2026-01-14 10:13 |
| 1 | b | `wave1/data/survey_wave1_2026-02-20.csv` | 2026-01-20 10:00 |
| 2 | a | `wave2/data/survey_wave2_2026-02-20.csv` | 2026-02-03 08:30 |
| 2 | b | `wave2/data/survey_wave2_2026-05-13.csv` | 2026-02-20 23:00 |

> Start time filters are applied to exclude test/pilot responses that fall before the official launch of each batch.

### Prescreener metadata
SPSS file from Bilendi containing age, gender, region, and party vote for all recruited participants:

```wave1/data/DE_155853 Complexity Science Hub Residents of Germany_final/DE_ 155853 Complexity Science Hub Residents of Germany.sav```


Merged onto the oTree data via `bilendi_id`.

---

## Output Files (`processed_data/`)

### `2026-05-13_allBilendiData.csv`
Intermediate file produced by `01_createdatafile.py`. One row per participant × wave, combining both waves and both batches, merged with prescreener metadata. Includes exclusion flags but does **not** yet drop excluded participants (except for duplicates).

---

### `2026-05-13_data_processed_participant.csv`
Produced by `02_preprocessData.py`. **Long format**: one row per participant × wave. Duplicates are dropped; only the first complete entry per participant × wave is retained.

#### Fixed columns (participant-level, not wave-suffixed)

| Column | Type | Description |
|---|---|---|
| `id` | str | Participant identifier (`bilendi_id`) |
| `wave` | int | Survey wave (1 or 2) |
| `gender` | str | `m` / `f` / `d` (mapped from German labels) |
| `age` | int | Age in years |
| `party_vote` | categorical | Party voted for in the last federal election (from prescreener) |
| `region` | categorical | German federal state (from prescreener) |
| `excl_double` | bool | Duplicate entry — only first complete entry retained |
| `excl_NA` | bool | All own opinions missing |
| `excl_training` | bool | Failed training after ≥5 attempts |
| `excl_time` | bool | Total task time < 1/3 of sample median |
| `excl_wave1Only` | bool | Completed wave 1 but not wave 2 |

#### Wave-varying columns

##### Timing (in seconds unless noted)

| Column | Description |
|---|---|
| `t_completed` | Datetime of successful completion |
| `time_total` | Consent to completion |
| `time_trainingGame` | Practice game phase |
| `time_training` | Practice/training phase |
| `time_spam` | Full SpAM task including instructions |
| `time_spam18dots` | SpAM time from first dot moved |
| `time_pairwise` | Full pairwise task including instructions |
| `time_pairwise18pairs` | Pairwise time after first pair |

##### Political Identity

| Column | Range | Description |
|---|---|---|
| `party_close` | categorical | Party participant feels closest to |
| `lr` | [0, 1] | Left–right self-placement (0 = far left, 1 = far right) |
| `polInterest` | [0, 1] | Political interest |
| `polFrequency` | [0, 1] | Frequency of political discussion |

##### Social Contacts

| Column | Description |
|---|---|
| `n_contacts` | Number of non-missing social contacts entered (max 10) |

##### Opinions on Policy Issues

For each issue `q` in `questions_sc`:

| Column | Range | Description |
|---|---|---|
| `first_x_self_{q}` | [-1, 1] | Participant's initial opinion (before entering contact/voter opinions) |
| `x_self_{q}` | [-1, 1] | Participant's final opinion |
| `std_socialCircle_ops_{q}` | ≥ 0 | Std. deviation of opinions across social contacts (opinion diversity in social circle) |
| `x_{party}_{q}` | [-1, 1] | Participant's perceived opinion of a typical voter of each party on this issue |

##### SpAM Task

| Column | Range | Description |
|---|---|---|
| `average_pixel_dist` | [0, 1] | Mean pairwise pixel distance across all 18 dots (normalised by max diagonal) |
| `average_pixel_dist_parties` | [0, 1] | Same, restricted to self + party dots only |
| `zeroVar_hint` | bool | Whether participant was nudged for placing all dots at the same position |

##### Satisfaction & Evaluation

| Column | Range | Description |
|---|---|---|
| `map_satisfaction` | [0, 1] | Satisfaction with the mapping task |
| `mappingEasier` | [0, 1] | Perceived ease: 0 = pairwise easier, 1 = mapping easier (bipolar, re-centred) |
| `mappingEnjoy` | [0, 1] | Enjoyment: 0 = pairwise more enjoyable, 1 = mapping more enjoyable (bipolar, re-centred) |

##### Polarisation Perceptions

| Column | Range | Description |
|---|---|---|
| `P_{q}` | [0, 1] | Perceived polarisation on issue `q` |
| `P_tot` | [0, 1] | Overall perceived polarisation (not issue-specific) |

##### Issue Importance

| Column | Range | Description |
|---|---|---|
| `w_{q}` | [0, 1] | Self-rated importance of issue `q` |

##### Party Sympathy *(Wave 2 only)*

| Column | Range | Description |
|---|---|---|
| `sym_{party}` | [0, 1] | Sympathy toward typical voter of each party (NaN in wave 1) |

##### Training Performance

| Column | Description |
|---|---|
| `attemptsPractice` | Attempts to pass training (categorical; -999 = never passed) |
| `dist_game_{a}-{b}` | Normalised pixel distance between practice game dot pairs (food items) |
| `dist_practice_{a}-{b}` | Normalised pixel distance between practice training dot pairs (self / friend / coworker / relative) |
| `passed_practice_sanity` | Whether participant passed the post-training comprehension check on first attempt (NaN if they never reached it) |

##### Treatment

| Column | Description |
|---|---|
| `treatment_wave2` | Bool: received treatment condition in wave 2 (prompted to choose contacts with diverse opinions) |

---

### `2026-05-13_data_processed_participant_pivot.csv`
Produced by `02_preprocessData.py`. **Wide/pivot format**: one row per participant (`id`). Wave-specific variables are prefixed `wave1_` or `wave2_`. Contains the same variables as the long-format file above, with demographic and exclusion columns appearing once (not wave-prefixed).

---

### `2026-05-13_data_processed_differences.csv`
Produced by `02_preprocessData.py`. **Long format**: one row per participant × wave × dot-pair. With 18 dots (`self`, `reference1`–`reference10`, 7 party dots), each participant contributes up to 153 rows per wave.

#### Identifiers & Demographics

| Column | Description |
|---|---|
| `id` | Participant identifier (`bilendi_id`) |
| `wave` | Survey wave (1 or 2) |
| `age` | Participant age |
| `party` | Party participant feels closest to |
| `lr` | Left–right self-placement, normalised to [0, 1] |

#### Dot-Pair Definition

| Column | Description |
|---|---|
| `dot1` | First dot (e.g. `self`, `reference3`, `GreenParty`) |
| `dot2` | Second dot |
| `ingroupdummy` | True if pair is `(self, closest_party)` |
| `minDistance_pair_dummy` | True if this pair was the minimum-distance pair on the map |
| `maxDistance_pair_dummy` | True if this pair was the maximum-distance pair on the map |

#### Opinion Positions

For each issue `q` in `questions_sc`:

| Column | Range | Description |
|---|---|---|
| `dot1_{q}` | [-1, 1] | Dot 1's opinion on issue `q` |
| `dot2_{q}` | [-1, 1] | Dot 2's opinion on issue `q` |
| `deltaX_{q}` | [0, 2] | Absolute opinion difference between dot 1 and dot 2 on issue `q` |

#### Map Placement

| Column | Range | Description |
|---|---|---|
| `pixel_dist` | [0, 1] | Normalised pixel distance between the two dots on the SpAM map |

#### Pairwise Similarity Rating

| Column | Range | Description |
|---|---|---|
| `similarity` | [0, 1] | Explicit similarity rating for this pair (NaN if pair was not shown in pairwise task) |

#### Affective Measures

| Column | Range | Condition |
|---|---|---|
| `sympathy` | [0, 1] | Sympathy toward party dot2 (only for `self`–party pairs in wave 2; NaN otherwise) |
| `socialCloseness` | [0, 1] | Perceived social closeness to contact dot2 (only for `self`–`reference*` pairs; NaN otherwise) |

---

## Reference Tables

### Issues (`questions_sc`)

| Variable name (`q`) | Short code | Topic |
|---|---|---|
| `climate_concern` | `cli` | Climate change concern |
| `gay_marriage` | `gay` | Same-sex marriage |
| `rights_indep_integration` | `mig` | Migrant rights / integration |
| `econ_inequality` | `equ` | Economic inequality |
| `regulate_internet` | `dig` | Internet / digital regulation |
| `east_germans` | `ddr` | East German identity / recognition |

### Parties

| Display name | Variable name | Color |
|---|---|---|
| Left Party | `LeftParty` | `#bd3075` |
| BSW | `BSW` | `#691940` |
| Green Party | `GreenParty` | `#7cbb15` |
| SPD | `SPD` | `#d71f1f` |
| FDP | `FDP` | `#ffcc00` |
| CDU/CSU | `CDU/CSU` | `#121212` |
| AfD | `AfD` | `#009de0` |

### Dots / Entities (SpAM map)

| Name | Type | Description |
|---|---|---|
| `self` | Self | The participant |
| `reference1`–`reference10` | Social contact | Up to 10 contacts named by the participant |
| `LeftParty`, `BSW`, `GreenParty`, `SPD`, `FDP`, `CDU/CSU`, `AfD` | Party | Typical voter of each of the 7 German parties |

### Scale Conventions

All scores are normalised to **[0, 1]**, opinions are normalised to **[-1, 1]** using the slider maxima defined in `consts.py`:

| Constant | Value | Used for |
|---|---|---|
| `MAX_OPINIONSLIDER` | 100 | Opinion positions |
| `MAX_DEFAULTSLIDER` | 100 | Importance, satisfaction, polarisation, social closeness |
| `MAX_DIPOLE_SLIDER` | 50 | Bipolar scales (LR, sympathy, mappingEasier/Enjoy); re-centred as `(x + 50) / 100` |
| `MAX_PIXELPOS` | 550 | Map pixel space; distances normalised by `√(550² + 550²)` |

---

## Exclusion Criteria

Applied in `01_createdatafile.py` and carried forward as flags (participants are **not** dropped in the intermediate file):

| Flag | Criterion |
|---|---|
| `excl_double` | More than one complete entry in the same wave — only first entry kept |
| `excl_NA` | All own opinion responses are missing |
| `excl_training` | ≥ 5 failed training attempts and training never passed |
| `excl_time` | Total task time < 1/3 of the sample median |
| `excl_wave1Only` | Valid in wave 1 but no valid wave 2 entry |

For most analyses, filter to rows where `excl_double == False`, `excl_NA == False`, and `excl_time == False`. For longitudinal analyses, additionally filter on `excl_wave1Only == False`.