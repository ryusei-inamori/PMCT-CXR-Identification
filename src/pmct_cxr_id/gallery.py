from __future__ import annotations
import pandas as pd

REQUIRED_QUERY = {"query_id", "identity_id", "image_path", "pmct_date"}
REQUIRED_GALLERY = {"identity_id", "image_path", "exam_date", "is_distractor"}


def load_queries(path):
    df = pd.read_csv(path, dtype={"query_id": str, "identity_id": str})
    missing = REQUIRED_QUERY - set(df.columns)
    if missing:
        raise ValueError(f"Missing query columns: {sorted(missing)}")
    df["pmct_date"] = pd.to_datetime(df["pmct_date"], errors="raise")
    return df


def load_gallery(path):
    df = pd.read_csv(path, dtype={"identity_id": str})
    missing = REQUIRED_GALLERY - set(df.columns)
    if missing:
        raise ValueError(f"Missing gallery columns: {sorted(missing)}")
    df["is_distractor"] = df["is_distractor"].astype(int).astype(bool)
    df["exam_date"] = pd.to_datetime(df["exam_date"], errors="coerce")
    return df


def select_category(gallery: pd.DataFrame, queries: pd.DataFrame, category: str) -> pd.DataFrame:
    if category not in {"oldest", "nearest", "all"}:
        raise ValueError(category)
    distractors = gallery[gallery.is_distractor].copy()
    study = gallery[~gallery.is_distractor].copy()
    if category == "all":
        out = pd.concat([study, distractors], ignore_index=True)
        out["selection_rule"] = "all"
        return out

    pmct_map = queries.drop_duplicates("identity_id").set_index("identity_id")["pmct_date"]
    selected = []
    for identity, group in study.groupby("identity_id", sort=True):
        group = group.dropna(subset=["exam_date"]).copy()
        if group.empty:
            continue
        dates = sorted(group.exam_date.unique())
        if category == "oldest":
            chosen = dates[0]
        else:
            if identity not in pmct_map:
                continue
            pmct = pmct_map.loc[identity]
            chosen = min(dates, key=lambda d: (abs(pd.Timestamp(d) - pmct), pd.Timestamp(d)))
        keep = group[group.exam_date == chosen].copy()
        keep["selection_rule"] = category
        keep["selected_exam_date"] = pd.Timestamp(chosen)
        selected.append(keep)
    out = pd.concat(selected + [distractors], ignore_index=True) if selected else distractors.copy()
    if "selection_rule" not in out:
        out["selection_rule"] = category
    out.loc[out.is_distractor, "selection_rule"] = "distractor"
    return out
