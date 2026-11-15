"""Feature definitions shared by training and serving.

Keeping this in one place means the API and the training script can never
disagree about column order or names. If I change a feature here, both sides
pick it up.
"""

# The 8 mix inputs, in the exact order the model expects them.
# Names map to the raw UCI columns but are cleaned up for a JSON API.
FEATURE_NAMES = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
]

# Units for each feature, used in the API schema and docs.
# All mix components are kg per cubic metre of concrete. Age is in days.
FEATURE_UNITS = {
    "cement": "kg/m^3",
    "blast_furnace_slag": "kg/m^3",
    "fly_ash": "kg/m^3",
    "water": "kg/m^3",
    "superplasticizer": "kg/m^3",
    "coarse_aggregate": "kg/m^3",
    "fine_aggregate": "kg/m^3",
    "age": "days",
}

TARGET_NAME = "compressive_strength"  # MPa

# The raw UCI spreadsheet uses long descriptive headers. This maps them to our
# clean names. The raw order matches FEATURE_NAMES + target.
RAW_COLUMN_MAP = {
    "Cement (component 1)(kg in a m^3 mixture)": "cement",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": "blast_furnace_slag",
    "Fly Ash (component 3)(kg in a m^3 mixture)": "fly_ash",
    "Water  (component 4)(kg in a m^3 mixture)": "water",
    "Superplasticizer (component 5)(kg in a m^3 mixture)": "superplasticizer",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": "coarse_aggregate",
    "Fine Aggregate (component 7)(kg in a m^3 mixture)": "fine_aggregate",
    "Age (day)": "age",
    "Concrete compressive strength(MPa, megapascals) ": TARGET_NAME,
}
