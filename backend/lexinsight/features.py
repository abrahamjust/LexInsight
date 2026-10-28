"""Extract structured legal features for every judgment.

Run:  python -m lexinsight.features

Output: data/processed/legal_features.jsonl, one record per judgment with
Acts, provisions (with head/body salience), legal-issue concepts, treated
citations and outcome labels.
"""

import json
from collections import Counter

from . import config
from .legal import citations as cit
from .legal import outcome as out
from .legal.extract import (
    default_act_map, extract_acts, parse_list_of_acts, resolve_unattributed, summarize_zones,
)
from .legal.lexicon import act_label, find_concepts


def load_corpus() -> list[dict]:
    with config.CORPUS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def legal_summary(case: dict) -> str:
    s = case["sections"]
    return "\n".join(x for x in (s.get("issue"), s.get("headnotes"), s.get("keywords")) if x)


def resolve_with_labels(provisions: dict, display: dict, listed: dict) -> dict:
    """Doc-level resolution of unattributed provisions, keeping display labels."""
    resolved = resolve_unattributed(provisions, listed)
    out = {}
    for pid, v in resolved.items():
        label = display.get(pid)
        if label is None:
            ref = pid.split(":", 1)[1]
            label = display.get(f"?:{ref}", pid)
            act_id = pid.split(":")[0]
            if act_id != "?":
                label = f"{act_label(act_id)} {label}"
        out[pid] = {**v, "label": label}
    return out


def case_features(case: dict) -> dict:
    s = case["sections"]
    head_text = "\n".join(x for x in (s.get("issue"), s.get("headnotes"), s.get("keywords")) if x)
    body_text = s.get("judgment", "")

    listed = parse_list_of_acts(s.get("acts", ""))
    head_acts, head_names = extract_acts(head_text)
    body_acts, body_names = extract_acts(body_text)

    acts = {}
    for act_id in set(listed) | set(head_acts) | set(body_acts):
        name, year = listed.get(act_id) or head_names.get(act_id) or body_names.get(act_id)
        acts[act_id] = {
            "name": name,
            "year": year,
            "listed": act_id in listed,
            "head": head_acts.get(act_id, 0),
            "body": body_acts.get(act_id, 0),
        }
    # Drop rare body-only Acts that are probably passing references,
    # unless listed by the editor.
    acts = {
        a: v for a, v in acts.items()
        if v["listed"] or v["head"] or v["body"] >= 2
    }

    defaults = default_act_map(listed)
    provisions, display = summarize_zones(
        {"issue": s.get("issue", ""), "headnotes": s.get("headnotes", ""),
         "acts": s.get("acts", ""), "keywords": s.get("keywords", ""), "judgment": body_text},
        defaults,
    )
    provisions = resolve_with_labels(provisions, display, listed)

    head_concepts = find_concepts(head_text)
    body_concepts = find_concepts(body_text[:60000])
    concepts = {
        c: {"head": head_concepts.get(c, 0), "body": body_concepts.get(c, 0)}
        for c in set(head_concepts) | set(body_concepts)
        if head_concepts.get(c, 0) or body_concepts.get(c, 0) >= 3
    }

    cited = cit.parse_case_law_cited(s.get("case_law_cited", ""))

    tail = body_text[-3000:]
    held = s.get("headnotes", "")
    outcome = {
        "procedural": out.procedural_outcome(s.get("result", ""), case.get("disposal_nature")),
        "result_text": s.get("result", ""),
        "substantive": out.substantive_outcomes(tail, held),
    }

    return {
        "case_id": case["case_id"],
        "acts": acts,
        "provisions": provisions,
        "concepts": concepts,
        "citations": cited,
        "outcome": outcome,
    }


def main():
    cases = load_corpus()
    features = [case_features(c) for c in cases]

    with config.LEGAL_FEATURES_PATH.open("w", encoding="utf-8") as f:
        for record in features:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    act_counts = Counter(a for r in features for a in r["acts"])
    prov_counts = Counter(p for r in features for p, v in r["provisions"].items() if v["head"])
    concept_counts = Counter(c for r in features for c, v in r["concepts"].items() if v["head"])
    pol = Counter(c["polarity"] for r in features for c in r["citations"])
    proc = Counter(r["outcome"]["procedural"] for r in features)
    sub = Counter(k for r in features for k in r["outcome"]["substantive"])
    unattributed = sum(1 for r in features for p in r["provisions"] if p.startswith("?:"))
    total_prov = sum(len(r["provisions"]) for r in features)

    print(f"Judgments: {len(features)}")
    print(f"Distinct Acts: {len(act_counts)}; top: {act_counts.most_common(12)}")
    print(f"Provisions: {total_prov} (unattributed {unattributed}); top headnote provisions: {prov_counts.most_common(15)}")
    print(f"Concepts in headnotes: {concept_counts.most_common(25)}")
    print(f"Citations: {sum(pol.values())} by polarity {dict(pol)}")
    print(f"Procedural outcomes: {dict(proc)}")
    print(f"Substantive outcomes: {dict(sub)}")
    print(f"Saved: {config.LEGAL_FEATURES_PATH}")


if __name__ == "__main__":
    main()
