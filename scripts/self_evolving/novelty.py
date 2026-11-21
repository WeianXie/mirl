"""Embedding-novelty selection + instrumentation for the rubric-mode proposer.

Best-of-N selection keeps, from N*K proposed candidates, the K most novel ones
that pass the validator:

    q* = argmax_{q in candidates} ( 1 - max_{h in H} cos(e_q, e_h) )

applied greedily: after each pick passes the validator its embedding joins H, so
two near-duplicates from the same batch are not both kept. H is the run-wide
history of ACCEPTED requests (persisted to server_embeddings_*.jsonl and reloaded
on restart, so a resumed job keeps measuring novelty against the same history).

Everything here is also computed for the control arm (N=1, no selection), so the
two arms log identical metrics. All math is numpy on L2-normalised float32 rows.
"""

from __future__ import annotations

import base64
import glob
import json
import logging
import math
import os
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding (de)serialisation: fp16 + base64 keeps a 2048-d row at ~5.5 KB.
# ---------------------------------------------------------------------------
def encode_emb(vec) -> str:
    return base64.b64encode(np.asarray(vec, dtype=np.float16).tobytes()).decode("ascii")


def decode_emb(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float16).astype(np.float32)


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)


class EmbeddingHistory:
    """Growable matrix of normalised embeddings with ids (H in the formula)."""

    def __init__(self, name: str):
        self.name = name
        self._m: np.ndarray | None = None
        self._n = 0
        self.ids: list[str] = []

    def __len__(self) -> int:
        return self._n

    @property
    def matrix(self) -> np.ndarray:
        if self._m is None:
            return np.zeros((0, 0), dtype=np.float32)
        return self._m[: self._n]

    def add(self, emb, item_id: str) -> None:
        row = _normalize(emb)[0]
        if self._m is None:
            self._m = np.zeros((1024, row.shape[0]), dtype=np.float32)
        if row.shape[0] != self._m.shape[1]:
            logger.warning(f"{self.name}: dim mismatch {row.shape[0]} != {self._m.shape[1]}; skipped")
            return
        if self._n == self._m.shape[0]:
            self._m = np.concatenate([self._m, np.zeros_like(self._m)], axis=0)
        self._m[self._n] = row
        self._n += 1
        self.ids.append(item_id)

    def stats(self, embs: np.ndarray, k: int = 10) -> list[dict]:
        """Per-row similarity to H: max cos (+ nearest id) and mean of top-k cos.

        Empty H -> max_cos None and novelty 1.0 (nothing to be similar to yet)."""
        q = _normalize(embs)
        if self._n == 0:
            return [{"max_cos": None, "nn_id": None, "knn_cos": None, "novelty": 1.0,
                     "hist_size": 0} for _ in range(q.shape[0])]
        sims = q @ self.matrix.T  # (M, n)
        out = []
        kk = min(k, self._n)
        for i in range(q.shape[0]):
            row = sims[i]
            j = int(row.argmax())
            topk = np.partition(row, -kk)[-kk:] if kk < self._n else row
            mx = float(row[j])
            out.append({"max_cos": mx, "nn_id": self.ids[j], "knn_cos": float(topk.mean()),
                        "novelty": 1.0 - mx, "hist_size": self._n})
        return out


def greedy_pick(cand: np.ndarray, hist: EmbeddingHistory, picked: list[np.ndarray],
                remaining: list[int]) -> tuple[int, float]:
    """Index (into cand rows) of the most novel remaining candidate w.r.t.
    H plus the already-picked rows, and its novelty. Ties break by order."""
    sub = cand[remaining]
    if len(hist):
        mx = (sub @ hist.matrix.T).max(axis=1)
    else:
        mx = np.full(len(remaining), -1.0, dtype=np.float32)
    if picked:
        mx = np.maximum(mx, (sub @ np.stack(picked).T).max(axis=1))
    best = int(np.argmax(-mx))  # argmax(1 - mx)
    nov = 1.0 - float(mx[best]) if mx[best] > -1.0 else 1.0
    return remaining[best], nov


def novelty_of(row: np.ndarray, hist: EmbeddingHistory, picked: list[np.ndarray]) -> float:
    """1 - max cos of one normalised row against H plus the already-picked rows
    (what greedy_pick would score it). Used to log the control arm identically."""
    mx = -1.0
    if len(hist):
        mx = float((hist.matrix @ row).max())
    if picked:
        mx = max(mx, float((np.stack(picked) @ row).max()))
    return 1.0 - mx if mx > -1.0 else 1.0


def intra_batch_stats(embs: np.ndarray) -> list[dict]:
    """Per-row max/mean cosine to the OTHER rows of the same candidate batch."""
    q = _normalize(embs)
    n = q.shape[0]
    if n < 2:
        return [{"intra_max_cos": None, "intra_mean_cos": None} for _ in range(n)]
    sims = q @ q.T
    np.fill_diagonal(sims, np.nan)
    return [{"intra_max_cos": float(np.nanmax(sims[i])), "intra_mean_cos": float(np.nanmean(sims[i]))}
            for i in range(n)]


def pairwise_cos(a, b) -> float:
    return float((_normalize(a) @ _normalize(b).T)[0, 0])


# ---------------------------------------------------------------------------
# Observed-category classifier (zero-shot, embedding prototypes)
#
# The proposer is CONDITIONED on use_case x specialty; these classify what it
# actually wrote, so the report can show category coverage and conditioning
# adherence. Several phrasings per label are averaged into one centroid.
# ---------------------------------------------------------------------------
_SPECIALTY_NAMES = {
    "cardiology": "cardiology: heart failure, arrhythmia, chest pain, hypertension, coronary disease",
    "neurology": "neurology: stroke, seizures, headache, dementia, neuropathy, multiple sclerosis",
    "oncology": "oncology: cancer staging, chemotherapy, tumors, immunotherapy, palliative care",
    "infectious_disease": "infectious disease: antibiotics, sepsis, HIV, tuberculosis, fungal and viral infections",
    "endocrinology": "endocrinology: diabetes, insulin, thyroid, adrenal, pituitary disorders",
    "nephrology": "nephrology: kidney disease, eGFR, dialysis, electrolytes, acute kidney injury",
    "pulmonology": "pulmonology: asthma, COPD, pneumonia, pulmonary embolism, ventilation",
    "gastroenterology": "gastroenterology: liver disease, GI bleeding, IBD, pancreatitis, endoscopy",
    "hematology": "hematology: anemia, anticoagulation, thrombosis, bleeding disorders, leukemia",
    "psychiatry": "psychiatry: depression, anxiety, psychosis, antidepressants, suicide risk",
    "emergency_medicine": "emergency medicine: trauma, resuscitation, overdose, acute triage in the ED",
    "pediatrics": "pediatrics: children, infants, neonates, pediatric dosing, vaccinations",
    "obgyn": "obstetrics and gynecology: pregnancy, labor, contraception, menstruation, postpartum",
    "dermatology": "dermatology: rash, skin lesions, eczema, psoriasis, melanoma",
    "rheumatology": "rheumatology: arthritis, lupus, gout, vasculitis, autoimmune disease",
    "general_internal_medicine": "general internal medicine: adult primary and inpatient care, multimorbidity",
    "family_medicine": "family medicine: outpatient primary care, prevention, screening, routine visits",
    "surgery": "surgery: operative management, perioperative care, postoperative complications",
    "ent": "otolaryngology (ENT): ear, nose, throat, sinusitis, hearing loss, tonsils",
    "radiology": "radiology: imaging, CT, MRI, X-ray, ultrasound interpretation and ordering",
}

_USE_CASE_PROTOS = {
    "care_consult": [
        "What is the best treatment or management for this patient?",
        "Help me work through the differential diagnosis and next steps for this patient.",
        "Should I start, stop or change this medication for my patient?",
    ],
    "writing_documentation": [
        "Write a clinical note, discharge summary or referral letter for this patient.",
        "Draft a message to my patient explaining their results in plain language.",
        "What ICD-10 or billing code should I use for this encounter documentation?",
    ],
    "medical_research": [
        "Summarize the evidence and clinical trials on this treatment question.",
        "What does the literature and current guidelines say about this intervention?",
        "Find and synthesize research studies comparing these therapies.",
    ],
}

_TASK_TYPE_PROTOS = {
    "diagnosis_differential": ["What is the likely diagnosis and differential for these symptoms?"],
    "treatment_management": ["What is the recommended treatment and management plan?"],
    "dosing_pharmacology": ["What dose of this drug should I give, and how do I adjust it?",
                            "Are there drug interactions or contraindications with this medication?"],
    "workup_tests": ["Which lab tests, imaging or workup should I order, and how do I interpret the result?"],
    "triage_emergency": ["Is this an emergency that needs urgent referral or admission right now?"],
    "prevention_screening": ["What screening, vaccination or preventive care is recommended?"],
    "prognosis_risk": ["What is the prognosis, risk score or likelihood of recurrence?"],
    "documentation_notes": ["Write or summarize a clinical note, discharge summary or letter."],
    "coding_billing": ["Which ICD-10, CPT or billing code applies to this diagnosis?"],
    "patient_communication": ["Explain this condition to the patient in simple, empathetic language."],
    "evidence_synthesis": ["Summarize the evidence from trials and guidelines on this question."],
    "procedure_technique": ["How do I perform this procedure and what are the steps and complications?"],
}


def _entropy(counter: Counter) -> float | None:
    n = sum(counter.values())
    if not n:
        return None
    return -sum((c / n) * math.log(c / n) for c in counter.values() if c)


class CategoryClassifier:
    """Nearest-centroid zero-shot labels for three taxonomies."""

    TAXONOMIES = {
        "specialty": {k: [v] for k, v in _SPECIALTY_NAMES.items()},
        "use_case": _USE_CASE_PROTOS,
        "task_type": _TASK_TYPE_PROTOS,
    }

    def __init__(self):
        self.centroids: dict[str, tuple[list[str], np.ndarray]] = {}

    @property
    def ready(self) -> bool:
        return bool(self.centroids)

    def all_texts(self) -> list[str]:
        return [t for tax in self.TAXONOMIES.values() for protos in tax.values() for t in protos]

    def fit(self, embs: list) -> None:
        """`embs` aligned with all_texts()."""
        it = iter(embs)
        for name, tax in self.TAXONOMIES.items():
            labels, cents = [], []
            for label, protos in tax.items():
                rows = _normalize(np.stack([np.asarray(next(it), dtype=np.float32) for _ in protos]))
                c = rows.mean(axis=0)
                cents.append(c / (np.linalg.norm(c) + 1e-8))
                labels.append(label)
            self.centroids[name] = (labels, np.stack(cents))

    def classify(self, embs: np.ndarray) -> list[dict]:
        """Per row: {<tax>: label, <tax>_score: cos, <tax>_margin: top1-top2}."""
        q = _normalize(embs)
        out = [dict() for _ in range(q.shape[0])]
        for name, (labels, cents) in self.centroids.items():
            sims = q @ cents.T
            order = np.argsort(-sims, axis=1)
            for i in range(q.shape[0]):
                a, b = order[i, 0], order[i, 1]
                out[i][name] = labels[a]
                out[i][f"{name}_score"] = float(sims[i, a])
                out[i][f"{name}_margin"] = float(sims[i, a] - sims[i, b])
        return out


# ---------------------------------------------------------------------------
# Reference set (the HealthBench Pro val prompts): METRIC ONLY.
# Similarity to the target distribution shows whether novelty-seeking drifts
# away from what the benchmark asks. Never used for selection or training.
# ---------------------------------------------------------------------------
def load_reference_texts(parquet_path: str, limit: int = 1000) -> list[str]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    texts = []
    for _, row in df.iterrows():
        prompt = row.get("prompt")
        try:
            msgs = list(prompt)
            user = [m.get("content", "") for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
            txt = user[-1] if user else ""
        except Exception:
            txt = ""
        if not txt:
            ei = row.get("extra_info")
            if isinstance(ei, dict):
                txt = ei.get("question") or ""
        if isinstance(txt, str) and txt.strip():
            texts.append(txt)
        if len(texts) >= limit:
            break
    return texts


def reload_histories(log_dir: str, req_hist: EmbeddingHistory, task_hist: EmbeddingHistory) -> int:
    """Rebuild H from earlier server_embeddings_*.jsonl in log_dir (resumed job)."""
    n = 0
    for path in sorted(glob.glob(os.path.join(log_dir, "server_embeddings_*.jsonl"))):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not r.get("accepted") or not r.get("emb"):
                    continue
                vec = decode_emb(r["emb"])
                if r.get("kind") == "request":
                    req_hist.add(vec, r.get("qid") or r.get("cand_id"))
                    n += 1
                elif r.get("kind") == "task":
                    task_hist.add(vec, r.get("qid"))
    return n


def category_entropy(rows: list[dict], key: str) -> float | None:
    return _entropy(Counter(r.get(key) for r in rows if r.get(key)))
