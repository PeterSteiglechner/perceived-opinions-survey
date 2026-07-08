questions_sc = [
    "climate_concern",
    "gay_marriage",
    "rights_indep_integration",
    "econ_inequality",
    "regulate_internet",
    "east_germans",
]
qs = ["cli", "gay", "mig", "equ", "dig", "ddr"]
qs_keys = [
    "climate",
    "gay rights",
    "migrant integration",
    "econ equality",
    "digital regulation",
    "east germany",
]


labelMap = {
    "climate_concern": "climate concern",
    "gay_marriage": "gay marriage",
    "rights_indep_integration": "migrant rights",
    "econ_inequality": "economic inequality",
    "regulate_internet": "digital regulation", 
    "east_germans": "recognition East Germans",
}
labelMap_nl = {
    "climate_concern": "climate\nconcern",
    "gay_marriage": "gay\nmarriage",
    "rights_indep_integration": "migrant\nrights",
    "econ_inequality": "economic\ninequality",
    "regulate_internet": "digital\nregulation", 
    "east_germans": "East German\nrecognition",
}


parties = ["Left Party", "BSW", "Green Party", "SPD", "FDP", "CDU/CSU", "AfD"]
parties_full = parties + ["No party", "Other party", "Refuse to say/No answer"]

partiesVars = [p.replace(" ", "") for p in parties]

references = [f"reference{k}" for k in range(1, 11)]
peeps = ["self"] + references + partiesVars

party_cmap = {
    "self": "#FF17FF",
    "Green Party": "#7cbb15",
    "GreenParty": "#7cbb15",
    "Bündnis 90/Die Grünen": "#7cbb15",
    "AfD": "#009de0",
    "LeftParty": "#bd3075",
    "Left Party": "#bd3075",
    "Die Linke": "#bd3075",
    "FDP": "#ffcc00",
    "CDU/CSU": "#121212",
    "SPD": "#d71f1f",
    "BSW": "#691940",
    "contact": "#ff6600",
}
party_cmap["No party"] = "darkgrey"
party_cmap["Other party"] = "grey"
party_cmap["Refuse to say/No answer"] = "lightgrey"
party_cmap["miscellaneous"] = "brown"
party_cmap["not voting"] = "darkgrey"

lrlabels = lambda lr: "left" if lr < -15 else ("right" if lr > 15 else "neutral")

practice_game_dots = ["tomatosalad", "spaghetticarbo", "pizza"]
practice_training_dots = ["self", "friend", "coworker", "relative"]
MAX_OPINIONSLIDER = 100
MAX_SIMSLIDER = 100
MAX_DEFAULTSLIDER = 100
MAX_NCONTACS = 10
MAX_PIXELPOS = 550
MAX_PRACTICEATTEMPTS = 5
MAX_DIPOLE_SLIDER = 50


parties_vote = [
    "Left Party",
    "Green Party",
    "SPD",
    "CDU/CSU",
    "AfD",
    "Miscellaneous",
    "Not Voting",
]

VOTING_PREF_MAP = {
    "Die Linke": "Left Party",
    "Bündnis 90 / Die Grünen": "Green Party",
    "SPD": "SPD",
    "CDU/CSU": "CDU/CSU",
    "AFD": "AfD",
    "Sonstige (z.B. FDP, BSW etc.)": "Miscellaneous",
    "Ich würde nicht an der Wahl teilnehmen": "Not Voting",
}

CATEGORY_ORDER = list(VOTING_PREF_MAP.keys())


colorsOrig = [
    (
        party_cmap["contact"]
        if "reference" in p
        else (
            party_cmap[p[5:].replace(" ", "")]
            if "voter" in p
            else party_cmap[p.replace(" ", "")]
        )
    )
    for p in peeps
]

cmapWave = {1: "#7fc97f", 2: "#beaed4"}
cmapQuestions = dict(zip(questions_sc, ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33"]))
cmapTreatment = {False: "#386cb0", True: "#f0027f"}
cmapTreatment[0] = cmapTreatment[False]
cmapTreatment[1] = cmapTreatment[True]
