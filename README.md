# Bilendi Survey Data Processing Pipeline

## Overview

This pipeline processes two-wave survey data from a Bilendi panel study on political opinion mapping among residents of Germany (Complexity Science Hub project). Participants completed a **Spatial Arrangement Mapping (SpAM)** task, placing "dots" representing themselves, social contacts, and party voters on a 2D map to reflect perceived political similarity. A subset also completed pairwise similarity ratings.

Data was collected in two waves (Wave 1: Jan–Feb 2026; Wave 2: Feb–May 2026).

---

## Pipeline Scripts

| Script | Description |
|---|---|
| `consts.py` | Shared constants: issue names, party lists, color maps, slider maxima |
| `01_createdatafile.py` | Merges raw oTree exports (both batches, both waves) with Bilendi prescreener metadata; applies initial exclusion flags; saves the combined raw-ish CSV |
| `02_preprocessData.py` | Cleans and reshapes data into three analysis-ready CSVs: participant-level long format, participant-level wide/pivot format, and pairwise ("differences") long format |
| `03_data-overview.py` | Descriptive plots and sanity checks: completion times, demographics, party identity, opinions, social-circle opinion spread, training performance, mapping-task evaluation |
| *(not included here)* | An intermediate step fits per-participant issue weights (`alpha`) from the SpAM positions using several kernels (correlation, linear, exponential, logistic) and writes the `*_withAllIssueWeights.csv` files consumed by scripts 05–08 |
| `05_prereganalysis_refactored.qmd` | Preregistered confirmatory analysis (R/Quarto). Tests hypotheses H1–H17 on tool validity/reliability, issue-weight determinants (party, LR, age), social-circle variance effects, and the wave-2 treatment intervention |
| `06_ToolAnalysis.py` | Test–retest reliability of map distances across waves, and validity checks: map distance vs. opinion difference, pairwise similarity, and voter (dis)likeability |
| `06_makingSenseOfIssueWeights.py` | Visualizes individual SpAM maps colored by opinion per issue alongside fitted issue-weight (alpha) parameters; plots fitted distance-kernel curves (linear/exponential/logistic) |
| `06_anlaysisIssueWeights.py` | Compares inferred issue weights across waves, party affiliation, and left-right placement (with significance annotations); correlates inferred weights with self-reported issue importance |
| `07_resin.py` | Relational Structure (ResIN) network analysis: builds phi-correlation networks of discretized opinion items, lays them out via PCA-rotated spring layout, and plots networks by party and social-circle context |
| `08_socialLens_analysis.py` | Explores how social-circle opinion variance relates to issue weights, likeability, and map distance to party voters; examines the wave-2 treatment's effect on social-circle variance ("social lens" analyses) |

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
Produced by `01_createdatafile.py`. One row per participant × wave, combining both waves and both batches, merged with prescreener metadata. Includes exclusion flags but does **not** yet drop excluded participants (except duplicates).

---

### `2026-06-19_data_processed_participant.csv`
Produced by `02_preprocessData.py`. **Long format**: one row per participant × wave, restricted to non-excluded rows.

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
| `excl_wave1Only` | bool | Valid in wave 1 but no valid wave 2 entry |

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
| `time_between_waves` | Days between wave 1 and wave 2 completion (for participants with both waves) |

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
| `n_contacts` | Number of reported social contacts (max 10) |

##### Opinions on Policy Issues

For each issue `q` in `questions_sc`:

| Column | Range | Description |
|---|---|---|
| `first_x_self_{q}` | [-1, 1] | Participant's initial opinion (before entering contact/voter opinions) |
| `x_self_{q}` | [-1, 1] | Participant's final opinion |
| `std_socialCircle_ops_{q}` | ≥ 0 | Std. deviation of opinions across social contacts (opinion diversity in social circle) |
| `x_{party}_{q}` | [-1, 1] | Participant's perceived opinion of a typical voter of each party on this issue |
| `zeroVar_hint` | bool | Whether participant was nudged for reporting the same opinion for all social contacts |

##### SpAM Task

| Column | Range | Description |
|---|---|---|
| `average_pixel_dist` | [0, 1] | Mean pairwise pixel distance across all 18 dots (normalised by max diagonal) |
| `average_pixel_dist_parties` | [0, 1] | Same, restricted to self + party dots only |

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
| `passed_practice_sanity` | Whether participant passed the post-training comprehension check on first attempt (NaN if never reached) |

##### Treatment

| Column | Description |
|---|---|
| `treatment_wave2` | Bool: received treatment condition in wave 2 (prompted to choose contacts with diverse opinions) |

---

### `2026-06-19_data_processed_participant_pivot.csv`
Produced by `02_preprocessData.py`. **Wide/pivot format**: one row per participant (`id`). Wave-specific variables are prefixed `wave1_` / `wave2_`; demographic and exclusion columns appear once (not wave-prefixed).

---

### `2026-06-19_data_processed_differences.csv`
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
| `minPixelDistance_pair_dummy` | True if this pair was the minimum-distance pair on the map |
| `maxPixelDistance_pair_dummy` | True if this pair was the maximum-distance pair on the map |

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

#### Social cirlce Opinion Variance

| Column | Range | Description |
|---|---|---|
| `std_socialCircle_ops_{q}` | ≥ 0 | Std. dev. of social-circle opinions on issue `q` (joined from participant-level data) |


#### Pairwise Similarity Rating

| Column | Range | Description |
|---|---|---|
| `pairwise_similarity` | [0, 1] | Explicit similarity rating for this pair (NaN if pair was not shown in pairwise task) |

#### Affective Measures

| Column | Range | Condition |
|---|---|---|
| `sympathy` | [0, 1] | Sympathy toward party dot2 (only for `self`–party pairs in wave 2; NaN otherwise) |
| `socialCloseness` | [0, 1] | Perceived social closeness to contact dot2 (only for `self`–`reference*` pairs; NaN otherwise) |
| `treatment_wave2` | bool | Whether the participant was in the treatment condition (wave 2) |

---

### `2026-07-07_data_processed_participant_withAllIssueWeights.csv` / `2026-07-07_data_processed_differences_withAllIssueWeights.csv`
Extended versions of the participant-level and pairwise CSVs above, augmented with per-participant, per-issue weights (`alpha`) inferred from the SpAM map by fitting several distance kernels to the relationship between pixel distance and opinion differences. Columns follow the pattern `{kernel}_alpha_{fitmode}_{issue}`, where `kernel` ∈ `{corrP, corrS, exp, linear, logistic}` and `fitmode` indicates whether all 18 dots or only party dots (and 'self' dot) were used for fitting. Fitted kernel parameters are stored as `{kernel}_{fitmode}_param1/2`. Consumed by scripts `05`–`08`.

---


## Overview of Analysis Files

### 05_prereganalysis_refactored.qmd

This is the confirmatory, preregistered analysis, written in R/Quarto so it can render to a report with inline statistics. It loads the two *_withAllIssueWeights.csv files and works through hypotheses H1–H17 in order. Aim 1 (H1–H5) validates the SpAM tool itself: whether the in-group party is placed closer than out-groups, whether map distance predicts likeability and pairwise similarity, whether the closest/farthest pairs get the most extreme similarity ratings, and whether inferred issue weights are stable within a person across waves relative to between people. Aim 2 (H6–H8) tests whether party affiliation, left-right placement, and age predict the six issue weights, run both jointly (with an issue interaction term) and separately per issue, using lmer for correlation-based weights and clustered-SE lm_robust for the constrained linear/exponential fits. Aim 3 (H9–H13) asks whether social-circle opinion variance predicts issue weights, reported importance, likeability, and perceived polarisation, and culminates in H13, a model comparison (MSE/R²/AIC/BIC) of several ways of aggregating opinion differences into predicted map distance (uniform weights, population-average weights, individually fitted weights, self-reported weights). Aim 4 (H14–H17, plus "new" H16/H17 variants) evaluates the wave-2 diversity-nudge treatment: whether it actually increased social-circle variance (H14), and whether that variance change moderates the relationship between opinion differences and perceived distance/likeability (H15–H17), including a two-step ITT-then-mechanism version of H16/H17. Several inline red-text annotations document corrections made after preregistration (e.g., switching variance to SD, fixing sign errors, removing the treatment term from H16).

### 06_ToolAnalysis.py
This script is the extention to Aim 1 of the preregistration, focused specifically on establishing that the SpAM map distances are a reliable and valid measurement instrument. It first computes test–retest reliability by correlating self–party and party–party pixel distances between wave 1 and wave 2 for control-condition participants, visualized as scatter and hexbin plots against the identity line, and reports the mean absolute difference between waves. It then checks convergent validity from three angles: correlating map distance with the six per-issue opinion differences, regressing pairwise dissimilarity ratings on map distance (logistic fit), and regressing voter dislikeability on map distance — each with an accompanying correlation coefficient printed and plotted. It also confirms that participants place their affiliated party's voter dot closer than others, and closes with exploratory distributional comparisons of map distance vs. dissimilarity vs. dislikeability vs. social closeness, plus heatmaps of average dislikeability and distance between every pair of parties' affiliates and voters.


### 06_makingSenseOfIssueWeights.py
This script is a diagnostic/illustrative tool for understanding individual participants' fitted issue weights rather than a hypothesis test. For a single chosen participant and wave, it redraws their actual SpAM map (colored by opinion per issue) side by side with the fitted correlation-based weight (alpha) for each issue, and saves both a combined multi-panel figure and individual per-issue map images (including cropped "focus" versions) to figs/maps/. It supports multiple weighting/kernel definitions (correlation, linear, exponential) and displays the corresponding fitted kernel parameters and functional form as annotated text. The final cell visualizes, across a random sample of participants, the distribution of fitted kernel parameters (histograms) and overlays the resulting distance-vs-belief-difference curves, giving a sense of how much the kernel shape varies across the sample for each fitting approach (exponential, linear, logistic).

### 06_anlaysisIssueWeights.py
This script produces the main descriptive and inferential figures on how inferred issue weights vary across waves and identity groups, largely mirroring Aim 2 of the preregistration but with richer visualization and custom significance-bracket annotations. It defines helper functions for pairwise Welch/Mann-Whitney tests with Bonferroni correction and for drawing significance brackets above grouped box/bar plots. It then compares correlation-based issue weights (corrP_alpha) across waves, across party affiliation (party_close, both as bar/box plots and as combined KDE+strip "raincloud" plots per issue), and across left-right position bins, each time annotating statistically significant pairwise differences. It also checks how well the inferred weights align with self-reported issue importance via per-issue regression plots, looks at correlations between issue weights themselves, and examines the maximum issue weight by party as a rough measure of single-issue voting.


### 07_resin.py
This script implements Response Item Network (ResIN) analysis on the discretized (Likert-binned) opinion data, an alternative, more exploratory way of visualizing the belief space compared to the SpAM map itself. It dummy-codes each participant's opinion on each issue into ordinal levels, then builds a network where nodes are (issue, level) pairs and edges are phi-correlations between co-occurring response patterns (excluding same-issue edges), optionally filtering to statistically significant edges. Network layouts are computed via spring embedding and then rotated with PCA so the dominant axis of disagreement lies along the x-axis; layouts fitted on one wave can be reused as a fixed reference for subsequent subgroup plots. The script visualizes these networks for the full sample, split by party affiliation, and for individual participants' social circles, using distinct marker shapes for response levels, node size scaled by response count, and edge width/opacity scaled by correlation strength/significance. In a final section, it projects each participant's own and party dots' opinions into this ResIN space and computes a "delta ResIN" distance, comparing it against the deltaX opinion differences, map distance, sympathy, and pairwise similarity as an alternative distance metric.

### 08_socialLens_analysis.py

This script explores the "social lens" idea underlying Aim 3/Aim 4: that the diversity of opinions in a participant's social circle shapes how they perceive and weigh political differences. It first visualizes how social-circle opinion standard deviation changes between waves 1 and 2, split by treatment vs. control, for each issue, to sanity-check the diversity-nudge manipulation ahead of the formal H14 test. It then checks whether social-circle variance is confounded with a participant's own opinion extremity, and separately regresses inferred issue weights (corrP_alpha) against social-circle standard deviation per issue, both in levels and in wave-to-wave change (again split by treatment condition), to see whether growing social-circle diversity on an issue is associated with that issue losing perceived importance. It also compares the distribution of correlation-based weights across control/treatment/wave-1 groups, and closes with example scatter plots and full pairwise heatmaps showing how average social-circle SD relates to voter likeability and map distance to voters, both for a single party pair (Greens vs. AfD) and across all party combinations.


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

### Dots (SpAM map)

| Name | Type | Description |
|---|---|---|
| `self` | Self | The participant |
| `reference1`–`reference10` | Social contact | Up to 10 contacts named by the participant |
| `LeftParty`, `BSW`, `GreenParty`, `SPD`, `FDP`, `CDU/CSU`, `AfD` | Party | Typical voter of each of the 7 German parties |

### Scale Conventions

All scores are normalised to **[0, 1]**, opinions are normalised to **[-1, 1]**, using the slider maxima defined in `consts.py`:

| Constant | Value | Used for |
|---|---|---|
| `MAX_OPINIONSLIDER` | 100 | Opinion positions |
| `MAX_DEFAULTSLIDER` | 100 | Importance, satisfaction, polarisation, social closeness |
| `MAX_DIPOLE_SLIDER` | 50 | Bipolar scales (LR, sympathy, mappingEasier/Enjoy); re-centred as `(x + 50) / 100` |
| `MAX_PIXELPOS` | 550 | Map pixel space; distances normalised by `√(550² + 550²)` |

---

## Exclusion Criteria

Applied in `01_createdatafile.py` and carried forward as flags:

| Flag | Criterion |
|---|---|
| `excl_double` | More than one complete entry in the same wave — only first entry kept |
| `excl_NA` | All own opinion responses are missing |
| `excl_training` | ≥ 5 failed training attempts and training never passed |
| `excl_noMetaData` | No matching prescreener metadata for the `bilendi_id` |
| `excl_time` | Total task time < 1/3 of the sample median |
| `excl_wave1Only` | Valid in wave 1 but no valid wave 2 entry |

The combined `excluded` flag (`excl_time | excl_NA | excl_double`) is used to drop rows before saving `2026-05-13_allBilendiData.csv`. For most downstream analyses, this filtering is already applied; for longitudinal analyses, additionally filter on `excl_wave1Only == False`.
```