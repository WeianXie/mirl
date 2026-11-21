"""Compare self-evolving HealthBench runs: proposer novelty, categories, accuracy.

Reads, per run, the experiment log dir written by slurm_b200_healthbench.sbatch:
  <run>/gen/server_candidates_*.jsonl   one row per proposed candidate
  <run>/gen/server_embeddings_*.jsonl   fp16 embeddings (requests + accepted tasks)
  <run>/gen/server_reports_*.jsonl      trainer rubric score per rollout per task
  <run>/gen/server_served_*.jsonl       every /sample hand-out
  <run>/gen/server_metrics_*.jsonl      60 s gen-server snapshots
  <run>/gen/server_accepted_*.jsonl     accepted entries (task text)
  <run>/trainer_metrics_*.jsonl         verl file logger (train + val per step)

Writes to --out_dir:
  report.html            self-contained charts + plain-language explanations
  summary.json           headline numbers per run
  timeseries.csv         tidy (run, step, metric, value) for any plotting tool
  candidates.csv         every candidate with its metrics (text truncated)
  tasks.csv              every accepted task joined with its trainer feedback
  trainer_metrics.csv    tidy trainer metrics (run, step, metric, value)

Gen-server events are mapped to trainer steps by wall clock (the trainer log
carries a ts per step). Without trainer metrics (a gen-only run, e.g.
slurm_cpu_healthbench_proposer.sbatch) the x-axis is batches of 32 accepted tasks.

Run inside the training SIF (needs numpy, pandas, sklearn):
  apptainer exec --bind /work:/work --bind /scratch:/scratch $SIF python -I \\
    scripts/self_evolving/report_novelty.py \\
    --run control=$LOGS/hbpro_qwen35_9b_control --run bon4=$LOGS/hbpro_qwen35_9b_control_bon4 \\
    --out_dir $LOGS/report_control_vs_bon4
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import math
import os
import re
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

TRAIN_KEYS = [  # trainer metrics plotted (substring match, first hit wins per key)
    ("train_score", r"^critic/score/mean$"),
    ("train_reward", r"^critic/rewards/mean$"),
    ("response_length", r"^response_length/mean$"),
    ("entropy", r"^actor/entropy$"),
    ("pg_loss", r"^actor/pg_loss$"),
    ("grad_norm", r"^actor/grad_norm$"),
    ("step_time_s", r"^timing_s/step$"),
]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _jsonl(paths):
    rows = []
    for p in sorted(paths):
        with open(p) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _unix(row):
    if isinstance(row.get("unix"), (int, float)):
        return float(row["unix"])
    ts = row.get("ts")
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return float("nan")


def load_run(name: str, path: str) -> dict:
    gen = os.path.join(path, "gen") if os.path.isdir(os.path.join(path, "gen")) else path
    g = lambda pat: glob.glob(os.path.join(gen, pat))  # noqa: E731
    run = {"name": name, "path": path}
    run["cands"] = pd.DataFrame(_jsonl(g("server_candidates_*.jsonl")))
    run["reports"] = pd.DataFrame(_jsonl(g("server_reports_*.jsonl")))
    run["served"] = pd.DataFrame(_jsonl(g("server_served_*.jsonl")))
    run["snaps"] = pd.DataFrame(_jsonl(g("server_metrics_*.jsonl")))
    acc = _jsonl(g("server_accepted_*.jsonl"))
    run["task_text"] = {
        r.get("question_id"): ((r.get("entry") or {}).get("extra_info") or {}).get("question", "")
        for r in acc}
    embs = {"request": {}, "task": {}}
    for r in _jsonl(g("server_embeddings_*.jsonl")):
        if r.get("emb"):
            key = r.get("qid") if r.get("kind") == "task" else r.get("cand_id")
            embs.setdefault(r.get("kind"), {})[key] = np.frombuffer(
                base64.b64decode(r["emb"]), dtype=np.float16).astype(np.float32)
    run["embs"] = embs
    tm = _jsonl(glob.glob(os.path.join(path, "trainer_metrics_*.jsonl")))
    # A resumed job re-logs steps; keep the last record per step.
    by_step = {}
    for r in tm:
        if isinstance(r.get("step"), int):
            by_step[r["step"]] = r
    run["trainer"] = [by_step[k] for k in sorted(by_step)]
    for df in (run["cands"], run["reports"], run["served"], run["snaps"]):
        if len(df):
            df["unix"] = [_unix(r) for r in df.to_dict("records")]
    return run


# ---------------------------------------------------------------------------
# step mapping
# ---------------------------------------------------------------------------
GEN_ONLY_BATCH = 32  # = data.train_batch_size, so gen-only bucket k ~ trainer step k


def step_mapper(run: dict):
    """unix -> trainer step in progress at that time (1-based). Without a
    timestamped trainer log (gen-only run): the batch of GEN_ONLY_BATCH accepted
    tasks in progress, so arms that generate at different speeds are compared at
    equal novelty-history size, not equal wall clock. 10-minute buckets only if
    nothing was accepted."""
    ts = [r.get("ts") for r in run["trainer"] if isinstance(r.get("ts"), (int, float))]
    steps = [r["step"] for r in run["trainer"] if isinstance(r.get("ts"), (int, float))]
    if ts:
        ts_arr, st_arr = np.asarray(ts), np.asarray(steps)
        last = int(st_arr.max())

        def f(u):
            i = np.searchsorted(ts_arr, u, side="left")
            return int(st_arr[i]) if i < len(st_arr) else last + 1
        return f, "trainer step"
    c = run["cands"]
    if len(c) and "outcome" in c:
        acc_unix = np.sort(c.loc[c["outcome"] == "accepted", "unix"].to_numpy(dtype=float))
        if len(acc_unix):
            # side="left": an iteration's own acceptances (same proposal unix) don't count.
            return ((lambda u: int(np.searchsorted(acc_unix, u, side="left")) // GEN_ONLY_BATCH + 1),
                    f"batch of {GEN_ONLY_BATCH} accepted tasks")
    starts = [df["unix"].min() for df in (run["cands"], run["reports"]) if len(df)]
    t0 = min(starts) if starts else 0.0
    return (lambda u: int((u - t0) // 600)), "10-min bucket"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _entropy(values) -> float | None:
    c = Counter(v for v in values if isinstance(v, str) and v)
    n = sum(c.values())
    if not n:
        return None
    return -sum((k / n) * math.log(k / n) for k in c.values())


def _f(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(x)


def build(runs: list[dict], k_clusters: int):
    tidy, summary, task_tables, cand_tables, trainer_tables = [], {}, [], [], []
    axis_label = None
    for run in runs:
        name, c = run["name"], run["cands"].copy()
        smap, axis_label = step_mapper(run)
        if len(c):
            c["step"] = [smap(u) for u in c["unix"]]
        acc = c[c.get("outcome", pd.Series(dtype=str)) == "accepted"].copy() if len(c) else c
        rep = run["reports"].copy()
        if len(rep):
            rep["step"] = [smap(u) for u in rep["unix"]]
        # per-task feedback
        fb = pd.DataFrame()
        if len(rep):
            fb = rep.groupby("question_id").agg(
                n_reports=("accuracy", "size"), mean_score=("accuracy", "mean"),
                min_score=("accuracy", "min"), max_score=("accuracy", "max"),
                first_step=("step", "min")).reset_index()
            fb["spread"] = (fb["max_score"] - fb["min_score"]) > 1e-6
        if len(acc):
            acc = acc.merge(fb, how="left", left_on="qid", right_on="question_id") if len(fb) else acc
            acc["task_text"] = acc["qid"].map(run["task_text"]).fillna("")
        acc["run"] = name
        task_tables.append(acc)
        if len(c):
            c["run"] = name
            cand_tables.append(c)

        def add(step, metric, value):
            v = _f(value) if not isinstance(value, str) else value
            if v is not None:
                tidy.append({"run": name, "step": int(step), "metric": metric, "value": v})

        # ---- proposer / generation metrics per step ----
        if len(c):
            for step, grp in c.groupby("step"):
                a = grp[grp["outcome"] == "accepted"]
                add(step, "candidates", len(grp))
                add(step, "accepted", len(a))
                add(step, "yield_accepted_per_candidate", len(a) / max(1, len(grp)))
                att = grp[grp["outcome"].isin(["accepted", "gen_error", "sft_failed"])]
                add(step, "generator_fail_rate", (att["outcome"] != "accepted").mean() if len(att) else None)
                add(step, "seed_leak_rate", (grp["outcome"] == "seed_leak").mean())
                for col, metric in [
                    ("novelty_at_proposal", "novelty_all_candidates"),
                    ("hist_max_cos", "nn_cos_to_history_all_candidates"),
                    ("ref_max_cos", "nn_cos_to_benchmark_all_candidates"),
                ]:
                    if col in grp:
                        add(step, metric, grp[col].mean())
                for col, metric in [
                    ("novelty_at_pick", "novelty_selected"),
                    ("hist_max_cos", "nn_cos_to_history_selected"),
                    ("hist_knn_cos", "knn10_cos_to_history_selected"),
                    ("intra_max_cos", "intra_batch_max_cos_selected"),
                    ("task_novelty", "task_novelty"),
                    ("task_hist_max_cos", "task_nn_cos_to_history"),
                    ("task_ref_max_cos", "task_nn_cos_to_benchmark"),
                    ("req_task_cos", "request_to_task_cos"),
                    ("task_chars", "task_chars"),
                    ("n_rubric", "rubric_items"),
                ]:
                    if col in a and len(a):
                        add(step, metric, a[col].mean())
                for col in ("task_obs_task_type", "task_obs_specialty", "use_case", "language"):
                    if col in a and len(a):
                        add(step, f"entropy_{col}", _entropy(a[col]))
        # cumulative category coverage
        if len(acc) and "task_obs_task_type" in acc:
            seen = set()
            for step, grp in acc.sort_values("unix").groupby("step"):
                seen |= set(zip(grp.get("task_obs_task_type", []), grp.get("task_obs_specialty", [])))
                add(step, "cumulative_type_x_specialty_cells", len(seen))
        # ---- trainer feedback on generated tasks (the proposer's "accuracy") ----
        if len(rep):
            for step, grp in rep.groupby("step"):
                add(step, "solver_score_on_generated", grp["accuracy"].mean())
                per_q = grp.groupby("question_id")["accuracy"]
                multi = per_q.size() >= 2
                if multi.any():
                    spread = (per_q.max() - per_q.min())[multi] > 1e-6
                    add(step, "learnable_task_frac", spread.mean())
                    add(step, "all_fail_task_frac", (per_q.max()[multi] < 0.05).mean())
                    add(step, "all_pass_task_frac", (per_q.min()[multi] > 0.95).mean())
        # ---- served source (stale history fallback = generator too slow) ----
        sv = run["served"]
        if len(sv):
            sv = sv.copy()
            sv["step"] = [smap(u) for u in sv["unix"]]
            for step, grp in sv.groupby("step"):
                add(step, "served_from_history_frac", (grp["source"] != "pool").mean())
        # ---- trainer ----
        tr_rows = []
        for r in run["trainer"]:
            d, step = r.get("data") or {}, r["step"]
            for k, v in d.items():
                if isinstance(v, (int, float)) and (k.startswith("val-") or any(
                        re.match(p, k) for _, p in TRAIN_KEYS)):
                    tr_rows.append({"run": name, "step": step, "metric": k, "value": float(v)})
            for short, pat in TRAIN_KEYS:
                for k, v in d.items():
                    if re.match(pat, k) and isinstance(v, (int, float)):
                        add(step, short, v)
                        break
            for k, v in d.items():
                if k.startswith("val-core/") and isinstance(v, (int, float)):
                    add(step, k, v)
        trainer_tables.append(pd.DataFrame(tr_rows))

        # ---- headline summary ----
        val_keys = sorted({k for r in run["trainer"] for k in (r.get("data") or {}) if k.startswith("val-core/")})
        main_val = next((k for k in val_keys if "acc" in k), val_keys[0] if val_keys else None)
        val_curve = [(r["step"], r["data"][main_val]) for r in run["trainer"]
                     if main_val and main_val in (r.get("data") or {})]
        summary[name] = {
            "path": run["path"],
            "best_of_n": int(c["best_of_n"].iloc[0]) if len(c) and "best_of_n" in c else None,
            "trainer_steps": len(run["trainer"]),
            "val_metric": main_val,
            "val_first": _f(val_curve[0][1]) if val_curve else None,
            "val_last": _f(val_curve[-1][1]) if val_curve else None,
            "val_best": _f(max(v for _, v in val_curve)) if val_curve else None,
            "candidates": int(len(c)),
            "accepted_tasks": int(len(acc)),
            "candidates_per_accepted_task": _f(len(c) / len(acc)) if len(acc) else None,
            "mean_novelty_selected": _f(acc["novelty_at_pick"].mean()) if len(acc) and "novelty_at_pick" in acc else None,
            "mean_novelty_all_candidates": _f(c["novelty_at_proposal"].mean()) if len(c) and "novelty_at_proposal" in c else None,
            "mean_task_novelty": _f(acc["task_novelty"].mean()) if len(acc) and "task_novelty" in acc else None,
            "mean_task_nn_cos_to_benchmark": _f(acc["task_ref_max_cos"].mean()) if len(acc) and "task_ref_max_cos" in acc else None,
            "entropy_task_type": _f(_entropy(acc["task_obs_task_type"])) if len(acc) and "task_obs_task_type" in acc else None,
            "entropy_specialty": _f(_entropy(acc["task_obs_specialty"])) if len(acc) and "task_obs_specialty" in acc else None,
            "specialty_adherence": _f((acc["task_obs_specialty"] == acc["specialty"]).mean()) if len(acc) and "task_obs_specialty" in acc else None,
            "solver_score_on_generated": _f(rep["accuracy"].mean()) if len(rep) else None,
            "learnable_task_frac": _f(fb.loc[fb["n_reports"] >= 2, "spread"].mean()) if len(fb) else None,
        }

    tasks = pd.concat([t for t in task_tables if len(t)], ignore_index=True) if any(len(t) for t in task_tables) else pd.DataFrame()
    cands = pd.concat(cand_tables, ignore_index=True) if cand_tables else pd.DataFrame()
    trainer = pd.concat([t for t in trainer_tables if len(t)], ignore_index=True) if any(len(t) for t in trainer_tables) else pd.DataFrame()

    # ---- joint embedding map + clusters (accepted tasks, all runs) ----
    emb_map = {}
    if len(tasks):
        X, keep = [], []
        for i, r in tasks.iterrows():
            run = next(rr for rr in runs if rr["name"] == r["run"])
            e = run["embs"].get("task", {}).get(r["qid"])
            if e is not None:
                X.append(e)
                keep.append(i)
        if len(X) >= 8:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from sklearn.feature_extraction.text import TfidfVectorizer

            X = np.stack(X)
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            xy = PCA(n_components=2, random_state=0).fit_transform(X)
            k = max(2, min(k_clusters, len(X) // 5))
            lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
            tasks.loc[keep, "map_x"], tasks.loc[keep, "map_y"] = xy[:, 0], xy[:, 1]
            tasks.loc[keep, "cluster"] = lab
            texts = tasks.loc[keep, "task_text"].fillna("").tolist()
            names = {}
            try:
                tf = TfidfVectorizer(stop_words="english", max_features=5000, min_df=1)
                M = tf.fit_transform(texts)
                vocab = np.array(tf.get_feature_names_out())
                for cl in range(k):
                    rows = np.where(lab == cl)[0]
                    w = np.asarray(M[rows].mean(axis=0)).ravel()
                    names[cl] = ", ".join(vocab[np.argsort(-w)[:4]])
            except Exception:
                names = {cl: f"cluster {cl}" for cl in range(k)}
            tasks["cluster_name"] = tasks["cluster"].map(lambda x: names.get(int(x)) if pd.notna(x) else None)
            emb_map = {"k": k, "names": {int(a): b for a, b in names.items()}}
            for name in tasks["run"].unique():
                sub = tasks[(tasks["run"] == name) & tasks["cluster"].notna()].sort_values("unix")
                seen = set()
                for step, grp in sub.groupby("step"):
                    seen |= set(grp["cluster"].astype(int))
                    tidy.append({"run": name, "step": int(step), "metric": "cumulative_clusters_covered", "value": len(seen)})
                    tidy.append({"run": name, "step": int(step), "metric": "entropy_cluster",
                                 "value": _entropy(grp["cluster"].astype(int).astype(str)) or 0.0})
                summary[name]["clusters_covered"] = len(set(sub["cluster"].astype(int)))
                summary[name]["entropy_cluster"] = _f(_entropy(sub["cluster"].astype(int).astype(str)))
    return pd.DataFrame(tidy), summary, tasks, cands, trainer, emb_map, axis_label


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
SECTIONS = [
    ("Does Best-of-N make the benchmark score grow faster?", [
        ("val-core", "HealthBench Pro validation score (all 525 real tasks, graded by the GPT judge). "
                     "The headline outcome: higher and earlier is better."),
        ("train_score", "Mean rubric score of the policy's own rollouts on generated training tasks."),
    ]),
    ("How novel are the proposed questions?", [
        ("novelty_selected", "Novelty of each kept request: 1 − cosine similarity to its nearest previously "
                             "accepted request. Best-of-N should sit above control here; that gap is what the "
                             "selection buys."),
        ("novelty_all_candidates", "Novelty of every candidate the proposer wrote, before any selection, "
                                   "against history only (no within-batch term, so it reads higher than the "
                                   "kept-request curve). Shows how novel the proposer is on its own."),
        ("task_novelty", "Novelty of the final task text (what the trainer actually sees) against all earlier tasks."),
        ("intra_batch_max_cos_selected", "Similarity of each kept request to its closest sibling in the same "
                                         "batch. Lower = less within-batch duplication."),
    ]),
    ("Do the questions still look like the benchmark?", [
        ("task_nn_cos_to_benchmark", "Cosine similarity of each generated task to its nearest real HealthBench Pro "
                                     "prompt. Chasing novelty can drift away from the target distribution; a drop "
                                     "here is the warning sign."),
    ]),
    ("Which categories are covered?", [
        ("entropy_task_type", None),
        ("entropy_task_obs_task_type", "Diversity (Shannon entropy) of the task type the questions actually are "
                                       "(diagnosis, dosing, documentation …), classified from embeddings. Higher = "
                                       "more even coverage."),
        ("entropy_task_obs_specialty", "Same for the clinical specialty the question is actually about."),
        ("cumulative_clusters_covered", "How many of the topic clusters (k-means over all runs' tasks) a run has "
                                        "touched so far. A steeper curve = faster exploration."),
        ("cumulative_type_x_specialty_cells", "How many task-type × specialty combinations have appeared so far."),
    ]),
    ("Are the generated tasks good training signal?", [
        ("solver_score_on_generated", "Mean rubric score the policy gets on freshly generated tasks (the "
                                      "proposer's 'accuracy' curve). Rising means tasks are getting easier for "
                                      "the model, falling means harder."),
        ("learnable_task_frac", "Share of tasks whose 8 rollouts scored differently, so GRPO gets a gradient. "
                                "This is the useful fraction of what the proposer makes."),
        ("all_fail_task_frac", "Share of tasks where every rollout scored ~0 (too hard / broken rubric)."),
        ("yield_accepted_per_candidate", "Accepted tasks per proposed candidate. Best-of-N is ≈1/N by design."),
        ("generator_fail_rate", "Share of generator attempts that failed validation."),
        ("served_from_history_frac", "Share of training samples served from the stale history buffer because "
                                     "fresh generation could not keep up. Should stay near 0 in both arms, "
                                     "or the comparison is confounded."),
    ]),
    ("Policy health", [
        ("response_length", "Mean response length (tokens)."),
        ("entropy", "Policy entropy."),
        ("grad_norm", "Gradient norm."),
        ("step_time_s", "Wall-clock seconds per training step."),
    ]),
]

SUMMARY_ROWS = [
    ("val_first", "Val score at start", "{:.3f}"), ("val_last", "Val score at end", "{:.3f}"),
    ("val_best", "Best val score", "{:.3f}"), ("trainer_steps", "Trainer steps logged", "{}"),
    ("accepted_tasks", "Accepted tasks", "{}"), ("candidates", "Candidates proposed", "{}"),
    ("candidates_per_accepted_task", "Candidates per accepted task", "{:.2f}"),
    ("mean_novelty_selected", "Novelty of kept requests", "{:.3f}"),
    ("mean_novelty_all_candidates", "Novelty of all candidates", "{:.3f}"),
    ("mean_task_novelty", "Novelty of final tasks", "{:.3f}"),
    ("mean_task_nn_cos_to_benchmark", "Similarity to benchmark", "{:.3f}"),
    ("entropy_task_type", "Task-type entropy", "{:.3f}"),
    ("entropy_specialty", "Specialty entropy", "{:.3f}"),
    ("clusters_covered", "Topic clusters covered", "{}"),
    ("specialty_adherence", "Observed = requested specialty", "{:.1%}"),
    ("solver_score_on_generated", "Policy score on generated tasks", "{:.3f}"),
    ("learnable_task_frac", "Learnable-task share", "{:.1%}"),
]


def _fmt(v, f):
    if v is None:
        return "–"
    try:
        return f.format(v)
    except Exception:
        return str(v)


def render_html(tidy, summary, tasks, emb_map, axis_label, out_path):
    runs = list(summary)
    palette = ["#4C6EF5", "#F08C00", "#2B8A3E", "#C2255C", "#5F3DC4"]
    color = {r: palette[i % len(palette)] for i, r in enumerate(runs)}
    charts, blocks = [], []
    metrics = set(tidy["metric"]) if len(tidy) else set()
    for title, items in SECTIONS:
        sec = []
        for key, desc in items:
            keys = sorted(m for m in metrics if m.startswith("val-core/")) if key == "val-core" else (
                [key] if key in metrics else [])
            for mk in keys:
                if desc is None:
                    continue
                sub = tidy[tidy["metric"] == mk]
                traces = []
                for r in runs:
                    s = sub[sub["run"] == r].sort_values("step")
                    if len(s):
                        traces.append({"x": s["step"].tolist(), "y": s["value"].tolist(), "name": r,
                                       "mode": "lines+markers", "line": {"color": color[r], "width": 2},
                                       "marker": {"size": 5}})
                if traces:
                    cid = f"c{len(charts)}"
                    charts.append({"id": cid, "traces": traces, "x": axis_label, "y": mk})
                    sec.append(f'<figure><div id="{cid}" class="chart"></div><figcaption><b>{mk}</b> — {desc}</figcaption></figure>')
        if sec:
            blocks.append(f"<section><h2>{title}</h2><div class='grid'>{''.join(sec)}</div></section>")

    # category distributions (accepted tasks)
    if len(tasks):
        sec = []
        for col, label in [("task_obs_task_type", "Observed task type"), ("task_obs_specialty", "Observed specialty"),
                           ("use_case", "Requested use case"), ("language", "Language"),
                           ("cluster_name", "Topic cluster (k-means)")]:
            if col not in tasks:
                continue
            traces = []
            cats = tasks[col].dropna().value_counts().index.tolist()
            for r in runs:
                s = tasks[(tasks["run"] == r)][col].dropna()
                if len(s):
                    vc = s.value_counts(normalize=True)
                    traces.append({"type": "bar", "orientation": "h", "name": r, "y": cats,
                                   "x": [float(vc.get(c_, 0.0)) for c_ in cats], "marker": {"color": color[r]}})
            if traces:
                cid = f"c{len(charts)}"
                charts.append({"id": cid, "traces": traces, "x": "share of accepted tasks", "y": "",
                               "bar": True, "h": max(320, 22 * len(cats) + 120)})
                sec.append(f'<figure><div id="{cid}" class="chart"></div><figcaption><b>{label}</b> — share of each '
                           f'run\'s accepted tasks.</figcaption></figure>')
        # novelty vs learning signal
        if "novelty_at_pick" in tasks and "spread" in tasks and tasks["spread"].notna().any():
            traces = []
            for r in runs:
                s = tasks[(tasks["run"] == r) & tasks["spread"].notna() & tasks["novelty_at_pick"].notna()]
                if len(s) >= 10:
                    q = pd.qcut(s["novelty_at_pick"], q=min(5, s["novelty_at_pick"].nunique()), duplicates="drop")
                    g = s.groupby(q, observed=True).agg(nov=("novelty_at_pick", "mean"), lf=("spread", "mean"),
                                                        sc=("mean_score", "mean"))
                    traces.append({"x": g["nov"].tolist(), "y": g["lf"].tolist(), "name": f"{r} learnable",
                                   "mode": "lines+markers", "line": {"color": color[r]}})
                    traces.append({"x": g["nov"].tolist(), "y": g["sc"].tolist(), "name": f"{r} score",
                                   "mode": "lines+markers", "line": {"color": color[r], "dash": "dot"}})
            if traces:
                cid = f"c{len(charts)}"
                charts.append({"id": cid, "traces": traces, "x": "novelty of kept request (quintile mean)",
                               "y": "share / score"})
                sec.append(f'<figure><div id="{cid}" class="chart"></div><figcaption><b>Novelty vs. learning '
                           f'signal</b> — tasks binned by novelty: solid = learnable share, dotted = policy '
                           f'score. Tells whether more novel tasks actually teach more.</figcaption></figure>')
        if "map_x" in tasks and tasks["map_x"].notna().any():
            traces = []
            for r in runs:
                s = tasks[(tasks["run"] == r) & tasks["map_x"].notna()]
                traces.append({"x": s["map_x"].round(4).tolist(), "y": s["map_y"].round(4).tolist(), "name": r,
                               "mode": "markers", "type": "scattergl",
                               "text": [t[:140].replace("<", "&lt;") for t in s["task_text"].fillna("")],
                               "hoverinfo": "text+name",
                               "marker": {"color": color[r], "size": 5, "opacity": 0.6}})
            cid = f"c{len(charts)}"
            charts.append({"id": cid, "traces": traces, "x": "PC1", "y": "PC2", "h": 460})
            sec.append(f'<figure class="wide"><div id="{cid}" class="chart"></div><figcaption><b>Map of all accepted '
                       f'tasks</b> — task embeddings projected to 2-D (PCA fitted jointly on every run). A run that '
                       f'spreads over more of the map explores more. Hover a dot to read the task.</figcaption></figure>')
        if sec:
            blocks.insert(min(4, len(blocks)), f"<section><h2>What do the questions look like?</h2><div class='grid'>{''.join(sec)}</div></section>")

    head = "".join(f"<th style='color:{color[r]}'>{r}</th>" for r in runs)
    body = "".join(
        f"<tr><td>{label}</td>" + "".join(f"<td>{_fmt(summary[r].get(k), f)}</td>" for r in runs) + "</tr>"
        for k, label, f in SUMMARY_ROWS)
    paths = "".join(f"<li><b>{r}</b>: <code>{summary[r]['path']}</code> (Best-of-N = {summary[r].get('best_of_n')})</li>" for r in runs)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Novelty Selection Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>
<style>
:root {{ --bg:#fbfbfa; --fg:#1d1d1f; --muted:#6b6b70; --card:#fff; --line:#e6e6e3; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#16171a; --fg:#ececee; --muted:#a0a0a8; --card:#1f2024; --line:#2e2f34; }} }}
:root[data-theme="dark"] {{ --bg:#16171a; --fg:#ececee; --muted:#a0a0a8; --card:#1f2024; --line:#2e2f34; }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; margin:0; padding:24px 16px 64px; }}
main {{ max-width:1180px; margin:0 auto; }}
h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ font-size:19px; margin:36px 0 12px; }}
p.lead, li, figcaption {{ color:var(--muted); }}
table {{ border-collapse:collapse; background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; width:100%; max-width:720px; }}
td, th {{ padding:7px 12px; border-bottom:1px solid var(--line); text-align:right; font-variant-numeric:tabular-nums; }}
td:first-child, th:first-child {{ text-align:left; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr)); gap:16px; }}
figure {{ margin:0; background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px 12px 12px; min-width:0; }}
figure.wide {{ grid-column:1/-1; }}
.chart {{ width:100%; height:320px; }}
code {{ font-size:12px; word-break:break-all; }}
</style></head><body><main>
<h1>Best-of-N novelty selection vs. control</h1>
<p class="lead">Generated {datetime.now():%Y-%m-%d %H:%M}. X-axis: {axis_label}. Each Best-of-N iteration
proposes N×K requests and keeps the K most novel that pass validation,
q* = argmax (1 − max<sub>h∈H</sub> cos(e<sub>q</sub>, e<sub>h</sub>)), H = all previously accepted requests.
Control keeps the proposer's K requests as written. Both arms log the same metrics.</p>
<ul>{paths}</ul>
<h2>Headline numbers</h2><table><tr><th>metric</th>{head}</tr>{body}</table>
{''.join(blocks)}
</main><script>
const CHARTS = {json.dumps(charts)};
const dark = matchMedia('(prefers-color-scheme: dark)').matches && document.documentElement.dataset.theme !== 'light';
const fg = dark ? '#ececee' : '#1d1d1f', grid = dark ? '#2e2f34' : '#ececec';
for (const c of CHARTS) {{
  const el = document.getElementById(c.id); if (c.h) el.style.height = c.h + 'px';
  Plotly.newPlot(el, c.traces, {{
    margin: {{l: c.bar ? 170 : 56, r: 12, t: 12, b: 44}}, barmode: 'group',
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {{color: fg, size: 12}},
    xaxis: {{title: c.x, gridcolor: grid, zeroline: false}}, yaxis: {{title: c.y, gridcolor: grid, zeroline: false, automargin: true}},
    legend: {{orientation: 'h', y: 1.12}}
  }}, {{displaylogo: false, responsive: true}});
}}
</script></body></html>"""
    with open(out_path, "w") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=EXP_LOG_DIR (repeatable)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--clusters", type=int, default=16)
    args = ap.parse_args()
    runs = []
    for spec in args.run:
        name, _, path = spec.partition("=")
        runs.append(load_run(name, path))
        r = runs[-1]
        print(f"{name}: {len(r['cands'])} candidates, {len(r['reports'])} reports, "
              f"{len(r['trainer'])} trainer steps, {len(r['embs'].get('task', {}))} task embeddings")
    tidy, summary, tasks, cands, trainer, emb_map, axis_label = build(runs, args.clusters)
    os.makedirs(args.out_dir, exist_ok=True)
    tidy.to_csv(os.path.join(args.out_dir, "timeseries.csv"), index=False)
    drop = [c for c in ("text",) if c in cands]
    if len(cands):
        c2 = cands.drop(columns=drop)
        c2["text"] = cands["text"].str.slice(0, 300) if "text" in cands else None
        c2.to_csv(os.path.join(args.out_dir, "candidates.csv"), index=False)
    if len(tasks):
        t2 = tasks.copy()
        if "text" in t2:
            t2["text"] = t2["text"].str.slice(0, 300)
        t2["task_text"] = t2["task_text"].str.slice(0, 600)
        t2.to_csv(os.path.join(args.out_dir, "tasks.csv"), index=False)
    if len(trainer):
        trainer.to_csv(os.path.join(args.out_dir, "trainer_metrics.csv"), index=False)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"runs": summary, "clusters": emb_map, "axis": axis_label}, f, indent=2)
    render_html(tidy, summary, tasks, emb_map, axis_label, os.path.join(args.out_dir, "report.html"))
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_dir}/report.html")


if __name__ == "__main__":
    main()
