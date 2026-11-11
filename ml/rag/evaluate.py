"""Offline evaluation of retrieval over the player-season corpus.

    python -m ml.rag.evaluate            # writes ml/reports/RAG_REPORT.md + rag_metrics.json

Runs the same embedding model and the same filters as serving, with exact
(brute-force) cosine search in numpy instead of Postgres, so it needs no
database. pgvector's cosine distance ranks identically.

1. **Comparables** (held-out season; evaluated, *not served*). For every player-season in season
   ``last - 1`` that has an outcome in ``last``, retrieve the k most similar
   past profiles at the same position from seasons up to ``last - 2`` (whose
   own outcomes are all known by ``last - 1``, so nothing leaks from the test
   season). Predict next-season PPR per game as this season's PPG plus the
   comparables' average change. If similarity search finds genuinely similar
   players, this beats both "no change" and the same rule with *random*
   same-position comparables. It didn't beat a plain position average, so
   serving (``analysis_core/retrieval.py``) doesn't use it.

2. **Injury history ranking** (what serving uses). For players with several recent seasons where
   one season clearly had the most injury-report listings, check whether the
   injury query ranks that season first (hit@1), against "most recent season
   first" and random order.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_core.retrieval import INJURY_QUERY, FastEmbedEmbeddings
from analysis_core.season import current_season

from ..train import RAW_DIR, REPORT_DIR
from .documents import build_documents

SEED = 7
MIN_GAMES = 4
COMPARABLES_K = 5


def doc_frame(docs) -> pd.DataFrame:
    return pd.DataFrame([{**d.metadata, "text": d.page_content} for d in docs])


def normalize(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def mae(a, b) -> float:
    return round(float(np.mean(np.abs(np.asarray(a) - np.asarray(b)))), 3)


def evaluate_comparables(df: pd.DataFrame, vecs: np.ndarray, last: int, k: int = COMPARABLES_K,
                         random_draws: int = 20) -> dict:
    rng = np.random.default_rng(SEED)
    q_mask = ((df["season"] == last - 1) & (df["games"] >= MIN_GAMES)
              & df["has_next"] & (df["next_games"].fillna(0) >= MIN_GAMES))
    c_mask = ((df["season"] <= last - 2) & (df["games"] >= MIN_GAMES)
              & df["has_next"] & (df["next_games"].fillna(0) >= MIN_GAMES))
    cand = df[c_mask]
    cand_vecs = vecs[c_mask.to_numpy()]
    cand_delta = (cand["next_ppg"] - cand["ppg"]).to_numpy()

    rows = []
    for i in np.flatnonzero(q_mask.to_numpy()):
        q = df.iloc[i]
        ok = ((cand["position"] == q["position"]) & (cand["player_id"] != q["player_id"])).to_numpy()
        idx = np.flatnonzero(ok)
        sims = cand_vecs[idx] @ vecs[i]
        top = idx[np.argsort(-sims)[:k]]
        rand = np.mean([cand_delta[rng.choice(idx, size=k, replace=False)].mean()
                        for _ in range(random_draws)])
        rows.append({
            "position": q["position"], "ppg": q["ppg"], "actual": q["next_ppg"],
            "similar": q["ppg"] + cand_delta[top].mean(),
            "random": q["ppg"] + rand,
            "position_mean": q["ppg"] + cand_delta[idx].mean(),
        })
    r = pd.DataFrame(rows)

    def scores(sub):
        return {
            "n": int(len(sub)),
            "no_change": mae(sub["ppg"], sub["actual"]),
            "random_comparables": mae(sub["random"], sub["actual"]),
            "position_average_change": mae(sub["position_mean"], sub["actual"]),
            "similar_comparables": mae(sub["similar"], sub["actual"]),
        }

    out = {"overall": scores(r)}
    for pos in ("QB", "RB", "WR", "TE"):
        out[pos] = scores(r[r["position"] == pos])
    out["relevant"] = scores(r[r["ppg"] >= 10])
    return out


def evaluate_injury_ranking(df: pd.DataFrame, vecs: np.ndarray, query_vec: np.ndarray,
                            last: int, window: int = 3) -> dict:
    rng = np.random.default_rng(SEED)
    recent = df[(df["season"] > last - window) & (df["season"] <= last)]
    hits = {"injury_query": 0, "most_recent_first": 0, "random_order": 0.0}
    n = 0
    for _, g in recent.groupby("player_id"):
        if len(g) < 2:
            continue
        listed = g["weeks_listed"].to_numpy()
        top = np.sort(listed)[::-1]
        # Only players where one season clearly stands out.
        if top[0] < 2 or top[0] - top[1] < 2:
            continue
        n += 1
        target = g.index[int(np.argmax(listed))]
        sims = vecs[g.index.to_numpy()] @ query_vec
        hits["injury_query"] += int(g.index[int(np.argmax(sims))] == target)
        hits["most_recent_first"] += int(g["season"].idxmax() == target)
        hits["random_order"] += 1 / len(g)
    return {"n": n, **{f"hit_at_1/{k}": round(v / n, 3) if n else None for k, v in hits.items()}}


def run(raw_dir: Path, first: int, last: int, report_dir: Path, embeddings=None) -> dict:
    docs = build_documents(raw_dir, first, last)
    df = doc_frame(docs)
    embeddings = embeddings or FastEmbedEmbeddings()
    vecs = normalize(np.array(embeddings.embed_documents(df["profile"].tolist())))
    query_vec = normalize(np.array([embeddings.embed_query(INJURY_QUERY)]))[0]

    report = {
        "generated_on": date.today().isoformat(),
        "embedding_model": getattr(embeddings, "model_name", type(embeddings).__name__),
        "documents": len(docs),
        "seasons": [first, last],
        "comparables": {"k": COMPARABLES_K, "test_target_season": last,
                        "mae": evaluate_comparables(df, vecs, last)},
        "injury_ranking": evaluate_injury_ranking(df, vecs, query_vec, last),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "rag_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (report_dir / "RAG_REPORT.md").write_text(render(report), encoding="utf-8")
    return report


def render(r: dict) -> str:
    c = r["comparables"]
    lines = [
        "# Retrieval evaluation",
        "",
        f"Generated by `python -m ml.rag.evaluate` on {r['generated_on']}. "
        f"Embedding model: `{r['embedding_model']}`. {r['documents']} player-season documents, "
        f"seasons {r['seasons'][0]}–{r['seasons'][1]}.",
        "",
    ]
    i = r["injury_ranking"]
    lines += [
        "## Injury history (served): is the most injury-affected season ranked first?",
        "",
        f"{i['n']} players with 2+ of the last 3 seasons, one with at least 2 more "
        "injury-report listings than any other.",
        "",
        "| Ranking | Hit@1 |",
        "|---|---|",
        f"| Injury query (retrieval) | **{i['hit_at_1/injury_query']}** |",
        f"| Most recent season first | {i['hit_at_1/most_recent_first']} |",
        f"| Random order | {i['hit_at_1/random_order']} |",
        "",
        f"## Comparables (evaluated, not served): predicting the held-out "
        f"{c['test_target_season']} season",
        "",
        f"Prediction = this season's PPR per game + the average next-season change of {c['k']} "
        "comparables. MAE in PPR points per game (lower is better). Similarity search over "
        "these profiles did not beat a plain position average, so serving does not use it.",
        "",
        "| Slice | n | No change | Random comparables | Position average | Similar comparables (retrieval) |",
        "|---|---|---|---|---|---|",
    ]
    for key in ("overall", "QB", "RB", "WR", "TE", "relevant"):
        m = c["mae"][key]
        lines.append(f"| {key} | {m['n']} | {m['no_change']} | {m['random_comparables']} | "
                     f"{m['position_average_change']} | {m['similar_comparables']} |")
    lines += [
        "",
        "`relevant` = at least 10 PPR points per game in the query season.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--first", type=int, default=2012)
    parser.add_argument("--last", type=int, default=current_season() - 1)
    parser.add_argument("--reports", type=Path, default=REPORT_DIR)
    args = parser.parse_args(argv)
    report = run(args.raw, args.first, args.last, args.reports)
    print(json.dumps(report["comparables"]["mae"]["overall"]))
    print(json.dumps(report["injury_ranking"]))


if __name__ == "__main__":
    main()
