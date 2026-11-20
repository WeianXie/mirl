"""Offline analysis of QueryProposer behavior from generation-server logs.

Joins the three (or four) per-run logs in --log_dir:
  server_proposer_<ts>.jsonl   one record per proposer call (queries, funnel,
                               diversity metrics)          [new instrumentation]
  server_accepted_<ts>.jsonl   every entry pushed to the pool
  server_rejected_<ts>.jsonl   validator rejections
  server_reports_<ts>.jsonl    trainer accuracy feedback per question_id

and prints: the generation funnel, diversity trends over time, duplicate
queries, retrieval-source mix, seed coverage, and trainer-signal quality by
source / by query. Degrades gracefully when a log is missing (e.g. runs that
predate the proposer log).

Works for both server modes. Rubric-mode (HealthBench) runs log mode=gen_task
with use_case/specialty per proposal and report a CONTINUOUS rubric score
instead of 0/1; for those the feedback tables add a mean_score column and a
"spread" column (rollouts' scores differed => nonzero GRPO advantage, the
continuous analogue of "mixed"), and a per-use_case breakdown is printed. On
0/1-reward runs the output is unchanged.

Usage:
  python scripts/self_evolving/analyze_proposer.py --log_dir <dir> [--ts <ts>]
  # --ts defaults to the newest server_*_<ts>.jsonl group in --log_dir
"""

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict


def _load_jsonl(path):
    if not path or not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _latest_ts(log_dir):
    stamps = set()
    for p in glob.glob(os.path.join(log_dir, "server_*_*.jsonl")):
        m = re.search(r"server_[a-z]+_(\d{8}_\d{6})\.jsonl$", p)
        if m:
            stamps.add(m.group(1))
    return max(stamps) if stamps else None


def _norm(q):
    return " ".join(q.lower().split())


def _fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def _section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def _trend(rows, keys, buckets=10):
    """Bucket rows (already time-ordered) into `buckets` slices and average
    each key per slice — shows drift over the run."""
    if not rows:
        return
    per = max(1, len(rows) // buckets)
    print("bucket  n      " + "  ".join(f"{k:>22s}" for k in keys))
    for b in range(0, len(rows), per):
        chunk = rows[b:b + per]
        cells = []
        for k in keys:
            vals = [r[k] for r in chunk if r.get(k) is not None]
            cells.append(f"{(sum(vals) / len(vals)):>22.3f}" if vals else f"{'-':>22s}")
        print(f"{b // per:>6d}  {len(chunk):<5d}  " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--ts", default=None, help="log timestamp suffix; default newest")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    ts = args.ts or _latest_ts(args.log_dir)
    if not ts:
        raise SystemExit(f"no server_*_<ts>.jsonl logs found in {args.log_dir}")
    print(f"log group: {ts}  ({args.log_dir})")

    p = lambda name: os.path.join(args.log_dir, f"server_{name}_{ts}.jsonl")
    proposals = _load_jsonl(p("proposer"))
    accepted = _load_jsonl(p("accepted"))
    rejected = _load_jsonl(p("rejected"))
    reports = _load_jsonl(p("reports"))
    print(f"records: proposer={len(proposals)} accepted={len(accepted)} "
          f"rejected={len(rejected)} reports={len(reports)}")

    # ---------------- funnel ----------------
    if proposals:
        _section("proposer funnel")
        by_mode = defaultdict(list)
        for r in proposals:
            by_mode[r.get("mode", "?")].append(r)
        for mode, rows in sorted(by_mode.items()):
            n_q = sum(r.get("n_queries", 0) for r in rows)
            zero = sum(r.get("zero_hit_queries", 0) for r in rows)
            gen_err = sum(oc.get("gen_errors", 0) for r in rows
                          for oc in r.get("query_outcomes", []))
            acc = sum(r.get("accepted", 0) for r in rows)
            rej = sum(r.get("rejected", 0) for r in rows)
            print(f"[{mode}] proposals={len(rows)} queries={n_q} "
                  f"zero_hit={zero} ({zero / max(1, n_q):.0%}) "
                  f"gen_errors={gen_err} accepted={acc} rejected={rej} "
                  f"accept/query={acc / max(1, n_q):.2f}")

        # Rubric mode (HealthBench): the same funnel split by the axes the
        # task proposer is conditioned on.
        rubric = [r for r in proposals if r.get("use_case")]
        if rubric:
            leaks = sum(1 for r in rubric for oc in r.get("query_outcomes", [])
                        if oc.get("seed_leak"))
            print(f"(rubric mode: {leaks} requests dropped by the style-seed leak guard)")
            for axis in ("use_case", "task_mode", "language", "anchored", "style_seeded"):
                groups = defaultdict(list)
                for r in rubric:
                    groups[str(r.get(axis))].append(r)
                cells = []
                for k, rows in sorted(groups.items()):
                    n_q = sum(r.get("n_queries", 0) for r in rows)
                    acc = sum(r.get("accepted", 0) for r in rows)
                    cells.append(f"{k}: n={len(rows)} acc/q={acc / max(1, n_q):.2f}")
                print(f"  by {axis:<12s} " + " | ".join(cells))

        _section("diversity trend (time buckets)")
        _trend(proposals, ["intra_mean_cos", "nn_vs_recent_cos",
                           "distinct_trigram_ratio", "accepted"])

        _section(f"top {args.top} repeated queries (normalized)")
        qc = Counter(_norm(oc.get("query", ""))
                     for r in proposals for oc in r.get("query_outcomes", []))
        for q, n in qc.most_common(args.top):
            if n > 1:
                print(f"{n:>4d}x  {q[:120]}")

        _section("retrieval source mix (top-5 hits per query)")
        sc = Counter(s for r in proposals for oc in r.get("query_outcomes", [])
                     for s in oc.get("top_sources", []))
        total_s = sum(sc.values()) or 1
        for s, n in sc.most_common(args.top):
            print(f"{n / total_s:>6.1%}  {s}")

        _section("seed coverage")
        tid = Counter(r.get("target_id", "") for r in proposals)
        n_seeds = len(tid)
        top_share = sum(n for _, n in tid.most_common(max(1, n_seeds // 10)))
        print(f"distinct seeds targeted: {n_seeds}; "
              f"top-10% of seeds get {top_share / max(1, len(proposals)):.0%} of proposals")

        _section("history conditioning")
        hh = [r for r in proposals if r.get("history_hits") is not None]
        with_ctx = [r for r in hh if r.get("history_context_chars", 0) > 0]
        scored = [r["history_mean_acc"] for r in hh if r.get("history_mean_acc") is not None]
        print(f"proposals with history context: {len(with_ctx)}/{len(hh)}; "
              f"mean neighbor accuracy when scored: "
              f"{_fmt(sum(scored) / len(scored) if scored else None)}")

    # ---------------- trainer feedback ----------------
    if reports:
        # Reports are binary 0/1 per rollout in diagnosis mode, so aggregate per
        # question first: qid -> (n reports, c correct). A qid with 0 < c < n had
        # disagreeing rollouts => nonzero GRPO advantage ("mixed" = useful
        # question). Rubric mode reports a continuous score, where c (>= 0.5)
        # is lossy; s/lo/hi carry the score-aware view.
        per_qid = defaultdict(lambda: {"n": 0, "c": 0, "s": 0.0, "lo": None, "hi": None,
                                       "source": "unknown", "query": None, "qid": None})
        continuous = False
        for r in reports:
            d = per_qid[r.get("question_id", "?")]
            a = float(r.get("accuracy", 0.0))
            continuous = continuous or a not in (0.0, 1.0)
            d["n"] += 1
            d["c"] += 1 if a >= 0.5 else 0
            d["s"] += a
            d["lo"] = a if d["lo"] is None else min(d["lo"], a)
            d["hi"] = a if d["hi"] is None else max(d["hi"], a)
            d["source"] = r.get("source", d["source"]) or d["source"]
            d["query"] = r.get("retrieval_query") or d["query"]

        def _useful(d):
            # Rollouts disagreed => nonzero GRPO advantage. Identical to
            # 0 < c < n when every report is exactly 0 or 1.
            if continuous:
                return d["hi"] - d["lo"] > 1e-6
            return 0 < d["c"] < d["n"]

        label = "spread" if continuous else "mixed"

        _section("trainer feedback by source (per-question aggregates)")
        by_src = defaultdict(list)
        for d in per_qid.values():
            by_src[d["source"]].append(d)
        print(f"{'source':<12s} {'qids':>6s} {'reports':>8s} {'mean_acc':>9s} "
              f"{'all_wrong':>10s} {'all_right':>10s} {'mixed':>7s}"
              + (f" {'mean_score':>11s} {'spread':>7s}" if continuous else ""))
        for src, ds in sorted(by_src.items()):
            multi = [d for d in ds if d["n"] >= 2]
            rates = [d["c"] / d["n"] for d in ds]
            mixed = (sum(1 for d in multi if 0 < d["c"] < d["n"]) / len(multi)
                     if multi else None)
            aw = (sum(1 for d in multi if d["c"] == 0) / len(multi)) if multi else None
            ar = (sum(1 for d in multi if d["c"] == d["n"]) / len(multi)) if multi else None
            extra = ""
            if continuous:
                ms = sum(d["s"] / d["n"] for d in ds) / len(ds)
                sp = (sum(1 for d in multi if _useful(d)) / len(multi)) if multi else None
                extra = (f" {ms:>11.3f} "
                         f"{('-' if sp is None else f'{sp:.0%}'):>7s}")
            print(f"{src:<12s} {len(ds):>6d} {sum(d['n'] for d in ds):>8d} "
                  f"{sum(rates) / len(rates):>9.3f} "
                  f"{('-' if aw is None else f'{aw:.0%}'):>10s} "
                  f"{('-' if ar is None else f'{ar:.0%}'):>10s} "
                  f"{('-' if mixed is None else f'{mixed:.0%}'):>7s}" + extra)
        print("(mixed = rollouts disagreed on a question = nonzero GRPO "
              "advantage; all_wrong/all_right questions contribute ~no signal)")
        if continuous:
            print("(continuous rewards detected: mean_acc/mixed binarize at 0.5; "
                  "mean_score is the raw mean and spread = rollouts' scores "
                  "differed at all, the exact nonzero-advantage condition)")

        rq = [d for d in per_qid.values() if d["query"] and d["n"] >= 2]
        if rq:
            _section(f"top {args.top} retrieval queries by {label}-question yield")
            per_q = defaultdict(list)
            for d in rq:
                per_q[_norm(d["query"])].append(d)
            scoreable = [
                (sum(1 for d in ds if _useful(d)) / len(ds), len(ds), q)
                for q, ds in per_q.items() if len(ds) >= 2
            ]
            for eff, n, q in sorted(scoreable, reverse=True)[:args.top]:
                print(f"{label}={eff:.0%} qids={n:>3d}  {q[:110]}")

        # correlate proposal diversity with downstream signal quality
        if proposals:
            qid2prop = {qid: r for r in proposals for qid in r.get("accepted_qids", [])}
            usable = []
            for qid, d in per_qid.items():
                pr = qid2prop.get(qid)
                if pr and pr.get("intra_mean_cos") is not None and d["n"] >= 2:
                    usable.append((pr["intra_mean_cos"],
                                   1.0 if _useful(d) else 0.0))
            if usable:
                _section("diversity vs signal quality (proposal intra_mean_cos quartiles)")
                usable.sort(key=lambda x: x[0])
                qtr = max(1, len(usable) // 4)
                for i in range(0, len(usable), qtr):
                    chunk = usable[i:i + qtr]
                    cos = sum(c for c, _ in chunk) / len(chunk)
                    eff = sum(e for _, e in chunk) / len(chunk)
                    print(f"Q{i // qtr + 1}: mean_cos={cos:.3f} "
                          f"{label}_frac={eff:.0%} (n={len(chunk)})")

            # Rubric mode (HealthBench): signal by use_case, the axis the
            # benchmark reports on (care_consult / writing_documentation /
            # medical_research).
            uc_rows = defaultdict(list)
            for qid, d in per_qid.items():
                pr = qid2prop.get(qid)
                if pr and pr.get("use_case"):
                    uc_rows[pr["use_case"]].append(d)
            if uc_rows:
                _section("trainer feedback by HealthBench use_case")
                print(f"{'use_case':<24s} {'qids':>6s} {'reports':>8s} "
                      f"{'mean_score':>11s} {label:>7s}")
                for uc, ds in sorted(uc_rows.items()):
                    multi = [d for d in ds if d["n"] >= 2]
                    sp = (sum(1 for d in multi if _useful(d)) / len(multi)) if multi else None
                    print(f"{uc:<24s} {len(ds):>6d} {sum(d['n'] for d in ds):>8d} "
                          f"{sum(d['s'] / d['n'] for d in ds) / len(ds):>11.3f} "
                          f"{('-' if sp is None else f'{sp:.0%}'):>7s}")

    if not proposals and accepted:
        _section("legacy fallback (no proposer log)")
        qc = Counter(_norm(q) for r in accepted for q in (r.get("queries_used") or []))
        print(f"accepted entries with queries_used: "
              f"{sum(1 for r in accepted if r.get('queries_used'))}")
        for q, n in qc.most_common(args.top):
            if n > 1:
                print(f"{n:>4d}x  {q[:120]}")


if __name__ == "__main__":
    main()
