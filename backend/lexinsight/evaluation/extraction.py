"""Legal-extraction quality (precision / recall / F1).

Run:  python -m lexinsight.evaluation.extraction

1. Acts: extraction from Issue + Headnotes + Keywords is compared with the
   editor-curated "List of Acts" of the same judgment (an independent,
   human-prepared field), over every judgment that has one.
2. Provisions: extraction from the start of Issue + Headnotes is compared
   with manually annotated gold sets: a development sample of 20 judgments
   (seed 7; used while fixing the extractor, so optimistic) and a held-out
   sample of 14 judgments (seed 11; annotated after the extractor was
   frozen). Scored strictly (Act + number) and leniently (number only).
3. Procedural outcome: agreement with the metadata disposal nature, an
   independent source.
4. Substantive outcome and citation treatments: manual spot checks of
   sampled predictions (see SPOT_CHECKS).

Gold annotations were drafted by Claude for the project and should be
checked by the project author.
"""

import json
import random

from .. import config
from ..features import load_corpus
from ..legal.extract import extract_acts, extract_provisions, parse_list_of_acts, resolve_unattributed
from ..legal.lexicon import CONST, canonical_act
from . import figures as F

SAMPLE_SEED = 7
SAMPLE_SIZE = 20
WINDOW = 1800

# case index -> list of (act name as written, kind, number); kind: s | art | r
GOLD_PROVISIONS = {
    38: [("Income Tax Act, 1961", "s", "147"), ("Income Tax Act, 1961", "s", "148")],
    49: [("Environment (Protection) Act, 1986", "s", "3")],
    59: [("Unlawful Activities (Prevention) Act, 1967", "s", n) for n in ("43D", "17", "18", "19")]
        + [("Code of Criminal Procedure, 1973", "s", "439")]
        + [("Penal Code, 1860", "s", n) for n in ("124A", "153A", "153B", "120B")]
        + [("Arms Act, 1959", "s", "25"), ("Arms Act, 1959", "s", "54")],
    71: [],
    74: [("Companies Act", "s", "182"), ("Constitution of India", "art", "14"), ("Constitution of India", "art", "19"),
         ("Representation of the People Act, 1951", "s", "29C"), ("Income Tax Act, 1961", "s", "13A")],
    88: [("Code of Criminal Procedure, 1973", "s", n) for n in ("41", "41A", "438", "440", "88", "170", "204", "209", "436A")],
    92: [("Food Safety and Standards Act, 2006", "s", "89"), ("Food Safety and Standards Act, 2006", "s", "59"),
         ("Penal Code, 1860", "s", "272"), ("Penal Code, 1860", "s", "273")],
    96: [("Income Tax Act, 1961", "s", "194H"), ("Contract Act, 1872", "s", "182")],
    154: [],
    219: [("Constitution of India", "art", "226")],
    246: [],
    331: [("Central Excise Act, 1944", "s", "4A"),
          ("Standards of Weight & Measures (Packaged Commodity) Rules, 1977", "r", "2"),
          ("Standards of Weight & Measures (Packaged Commodity) Rules, 1977", "r", "34")],
    374: [],
    404: [("Maharashtra Slum Areas (Improvement, Clearance and Redevelopment) Act, 1971", "s", "13"),
          ("Constitution of India", "art", "226")],
    428: [],
    444: [],
    519: [],
    548: [("Income Tax Act, 1961", "s", n) for n in ("149", "148", "151")]
         + [("Taxation and Other Laws (Relaxation and Amendment of Certain Provisions) Act, 2020", "s", "3")],
    596: [],
    666: [],
}

# Held-out sample (seed 11, 1,200 characters), annotated AFTER the
# extractor was frozen; index 96 overlaps the development sample and is
# excluded. This is the unbiased estimate.
HELDOUT_SEED = 11
HELDOUT_SIZE = 15
HELDOUT_WINDOW = 1200
IPC_, CRPC_, POCSO_ = "Penal Code, 1860", "Code of Criminal Procedure, 1973", "Protection of Children from Sexual Offences Act, 2012"
HELDOUT_PROVISIONS = {
    189: [(IPC_, "s", "302"), (IPC_, "s", "34")],
    190: [],
    194: [],
    457: [("Finance Act, 1994", "s", "65")],
    462: [("Delhi School Education Rules, 1973", "r", "46"), ("Delhi School Education Rules, 1973", "r", "47")],
    463: [(IPC_, "s", n) for n in ("409", "420", "467", "468", "471", "120B")]
         + [("Maharashtra Protection of Interest of Depositors (in Financial Establishments) Act, 1999", "s", "3"),
            (CRPC_, "s", "439")],
    476: [],
    487: [("Constitution of India", "art", "226")],
    520: [(POCSO_, "s", "15"), (POCSO_, "s", "30"), ("Information Technology Act, 2000", "s", "67B"), (CRPC_, "s", "482")],
    524: [(IPC_, "s", n) for n in ("450", "376", "376D", "376A", "302", "34")] + [(POCSO_, "s", "5"), (POCSO_, "s", "6")],
    573: [],
    601: [(IPC_, "s", "302"), (IPC_, "s", "34"), (CRPC_, "s", "161")],
    628: [("Constitution of India", "art", "14")],
    644: [("Indian Divorce Act, 1869", "s", "10")],
}
# Order XLI Rule 33 CPC in judgment 476 is scored separately as an order/rule id.
HELDOUT_ORDER_RULES = {476: ["code_of_civil_procedure:o41r33"]}

# Manual verification of sampled outputs (drafted by Claude; each entry
# records the judgment, the extracted value and whether it was correct).
SPOT_CHECKS_PATH = config.EVAL_QRELS_DIR / "extraction_spot_checks.json"


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {"precision": None if p is None else round(p, 4), "recall": None if r is None else round(r, 4),
            "f1": None if f is None else round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def gold_ids(entries):
    out = set()
    for act, kind, num in entries:
        resolved = canonical_act(act)
        act_id = CONST if kind == "art" else (resolved[0] if resolved else "?")
        out.add(f"{act_id}:{kind}{num}")
    return out


def evaluate_acts(cases):
    tp = fp = fn = 0
    n = 0
    for c in cases:
        gold = set(parse_list_of_acts(c["sections"].get("acts", "")))
        if not gold:
            continue
        n += 1
        s = c["sections"]
        pred = set(extract_acts("\n".join([s.get("issue", ""), s.get("headnotes", ""), s.get("keywords", "")]))[0])
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    return {"label": "Acts (headnotes vs editor's List of Acts)", "n_docs": n, **prf(tp, fp, fn)}


def _score_provisions(cases, sample, gold_map, window, extra=None):
    strict, lenient, errors = [0, 0, 0], [0, 0, 0], []
    for i in sample:
        s = cases[i]["sections"]
        text = (s["issue"] + " " + s["headnotes"]).replace(chr(10), " ")[:window]
        listed = parse_list_of_acts(s.get("acts", ""))
        pred = set(resolve_unattributed(dict(extract_provisions(text)[0]), listed))
        gold = gold_ids(gold_map[i]) | set((extra or {}).get(i, []))
        for bucket, p, g in ((strict, pred, gold),
                             (lenient, {x.split(":")[1] for x in pred}, {x.split(":")[1] for x in gold})):
            bucket[0] += len(p & g)
            bucket[1] += len(p - g)
            bucket[2] += len(g - p)
        if pred != gold:
            errors.append({"case_id": cases[i]["case_id"], "missed": sorted(gold - pred), "spurious": sorted(pred - gold)})
    return strict, lenient, errors


def evaluate_provisions(cases):
    rng = random.Random(SAMPLE_SEED)
    sample = sorted(rng.sample(range(len(cases)), SAMPLE_SIZE))
    assert sample == sorted(GOLD_PROVISIONS), "gold annotations no longer match the sampled judgments"
    strict, lenient, errors = _score_provisions(cases, sample, GOLD_PROVISIONS, WINDOW)
    rng = random.Random(HELDOUT_SEED)
    held = [i for i in sorted(rng.sample(range(len(cases)), HELDOUT_SIZE)) if i not in GOLD_PROVISIONS]
    assert held == sorted(HELDOUT_PROVISIONS), "held-out annotations no longer match the sample"
    h_strict, h_lenient, h_errors = _score_provisions(cases, held, HELDOUT_PROVISIONS, HELDOUT_WINDOW, HELDOUT_ORDER_RULES)
    return (
        {"label": "Provisions, dev sample, strict (Act + number)", "n_docs": len(sample), **prf(*strict)},
        {"label": "Provisions, dev sample, lenient (number only)", "n_docs": len(sample), **prf(*lenient)},
        {"label": "Provisions, held-out, strict (Act + number)", "n_docs": len(held), **prf(*h_strict)},
        {"label": "Provisions, held-out, lenient (number only)", "n_docs": len(held), **prf(*h_lenient)},
        {"development": errors, "held_out": h_errors},
    )


def evaluate_procedural(cases, features):
    mapping = {"appeal(s) allowed": "allowed", "case allowed": "allowed", "dismissed": "dismissed",
               "case partly allowed": "partly_allowed", "disposed off": "disposed"}
    agree = total = 0
    confusion = {}
    for c, f in zip(cases, features):
        meta = (c.get("disposal_nature") or "").lower().split("|")[0].strip()
        gold = mapping.get(meta)
        pred = f["outcome"]["procedural"]
        if gold is None or pred is None:
            continue
        total += 1
        agree += gold == pred
        confusion.setdefault(gold, {}).setdefault(pred, 0)
        confusion[gold][pred] += 1
    return {"label": "Procedural outcome vs metadata disposal", "n_docs": total,
            "precision": round(agree / total, 4), "recall": round(agree / total, 4), "f1": round(agree / total, 4),
            "agreement": round(agree / total, 4), "confusion": confusion}


def evaluate_spot_checks():
    if not SPOT_CHECKS_PATH.exists():
        return {}
    data = json.loads(SPOT_CHECKS_PATH.read_text(encoding="utf-8"))
    out = {}
    for key, items in data["checks"].items():
        correct = sum(1 for x in items if x["correct"])
        out[key] = {"label": data["labels"][key], "n_docs": len(items),
                    "precision": round(correct / len(items), 4) if items else None,
                    "recall": None, "f1": None}
    return out


def main():
    cases = load_corpus()
    with config.LEGAL_FEATURES_PATH.open(encoding="utf-8") as fh:
        features = [json.loads(line) for line in fh]
    acts = evaluate_acts(cases)
    strict, lenient, h_strict, h_lenient, errors = evaluate_provisions(cases)
    proc = evaluate_procedural(cases, features)
    results = {"acts": acts, "provisions_dev_strict": strict, "provisions_dev_lenient": lenient,
               "provisions_heldout_strict": h_strict, "provisions_heldout_lenient": h_lenient,
               "procedural_outcome": proc}
    results.update(evaluate_spot_checks())
    out = {
        "note": "Acts: gold = editor's List of Acts; provisions: 20 annotated judgments; "
                "procedural outcome: agreement with the metadata disposal field (P = R = agreement); spot checks: "
                "precision of sampled extracted values judged manually.",
        "results": results,
        "provision_errors": errors,
    }
    config.EVAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (config.EVAL_METRICS_DIR / "extraction_quality.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in results.items():
        print(f"{v['label']:55s} P {v['precision']}  R {v['recall']}  F1 {v['f1']}  (n={v['n_docs']})")

    short = {
        "acts": "Acts\n(vs List\nof Acts)", "provisions_dev_strict": "Provisions\ndev, strict",
        "provisions_dev_lenient": "Provisions\ndev, lenient", "provisions_heldout_strict": "Provisions\nheld-out,\nstrict",
        "provisions_heldout_lenient": "Provisions\nheld-out,\nlenient", "procedural_outcome": "Procedural\noutcome",
        "outcome_dev_v1": "Outcome\nv1 (dev)", "outcome_heldout_v2": "Outcome\nv2 (held-out)",
        "citation_treatment": "Citation\ntreatment",
    }
    rows = list(results.items())
    labels = [short.get(k, v["label"]) for k, v in rows]
    F.grouped_bars(
        config.EVAL_FIGURES_DIR / "fig7_extraction_quality.png", labels,
        {"Precision": [r[1]["precision"] for r in rows], "Recall": [r[1]["recall"] for r in rows],
         "F1": [r[1]["f1"] for r in rows]},
        "Legal information extraction quality", "Score",
        subtitle="Spot checks report precision only; procedural outcome reports agreement with metadata",
        figsize=(12, 5), rotate=False,
    )


if __name__ == "__main__":
    main()
