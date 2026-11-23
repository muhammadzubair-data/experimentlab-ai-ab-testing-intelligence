"""Create sample_data/messy_upload_example.csv: real-world export problems for the upload validator."""
from pathlib import Path

import numpy as np
import pandas as pd

from experimentlab.core import generate_scenario

OUT = Path(__file__).resolve().parents[1] / "sample_data" / "messy_upload_example.csv"


def build(seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = generate_scenario(seed=seed, n_total=3000, days=10).astype(object)
    d["converted"] = d["converted"].map({1: "yes", 0: "no"})
    d["variant"] = d["variant"].map({"A": "control", "B": "new_checkout"})
    d["timestamp"] = pd.to_datetime(d["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    d = d.rename(columns={"variant": "Group", "converted": "Outcome", "user_id": "visitor_id", "timestamp": "Date",
                          "device": "Device"})
    d["session_note"] = [f"note-{i}" for i in range(len(d))]          # high-cardinality: not a segment
    blank = d.sample(4, random_state=1).index
    d.loc[blank, "Group"] = " "
    bad = d.drop(blank).sample(5, random_state=2).index
    d.loc[bad, "Outcome"] = ["maybe", "2", "", "n/a", "-1"]
    d.loc[d.sample(3, random_state=3).index, "Date"] = "not a date"
    dup = d.sample(7, random_state=4)                                     # exact duplicates
    rep = d[d.Group.isin(["control", "new_checkout"])].sample(6, random_state=5).copy()
    rep["Outcome"] = "yes"                                                 # same user, same arm, extra row
    rep["session_note"] = "repeat"
    cross = d[d.Group == "control"].sample(4, random_state=6).copy()
    cross["Group"] = "new_checkout"                                        # same user in both arms
    cross["session_note"] = "cross"
    return pd.concat([d, dup, rep, cross]).sample(frac=1, random_state=7).reset_index(drop=True)


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    build().to_csv(OUT, index=False)
    print(OUT)
