"""Load and clean the concrete dataset."""

from pathlib import Path

import pandas as pd

from .features import FEATURE_NAMES, RAW_COLUMN_MAP, TARGET_NAME

# Default location of the raw UCI spreadsheet, relative to the repo root.
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Concrete_Data.xls"


def load_dataset(path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the UCI concrete spreadsheet and return a clean DataFrame.

    Columns come back as the 8 feature names plus the target, all lower case.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find the dataset at {path}. "
            "Download it from the UCI repository first (see README)."
        )

    # xlrd reads the legacy .xls format the UCI file ships in.
    df = pd.read_excel(path, engine="xlrd")
    df = df.rename(columns=RAW_COLUMN_MAP)

    missing = [c for c in FEATURE_NAMES + [TARGET_NAME] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns after renaming: {missing}. "
            f"Got columns: {list(df.columns)}"
        )

    # Keep only the columns we care about, in the right order.
    df = df[FEATURE_NAMES + [TARGET_NAME]].copy()

    # The raw data has no nulls, but drop any just in case a bad copy slips in.
    df = df.dropna().reset_index(drop=True)
    return df
