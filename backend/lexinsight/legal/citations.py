"""Parse the editor-curated 'Case Law Cited' block into treated citations.

SCR lists cited authorities in groups, each closed by a treatment:
"A v. B [2007] 3 SCR 507 : (2007) 3 SCC 755; C v. D (1976) 1 SCC 181
– referred to. E v. F [2023] 14 SCR 666 – held inapplicable."

Each treatment is mapped to a polarity. A citation is therefore not
assumed to be an endorsement: "distinguished", "held inapplicable" and
"overruled" are negative, "referred to" is neutral.
"""

import re

POSITIVE = {
    "relied on", "relied upon", "followed", "approved", "affirmed", "held applicable",
    "held good law", "held correct law", "correct law", "impliedly affirmed",
    "explained and relied on", "clarified and relied on", "applied", "minority view held correct",
}
NEGATIVE = {
    "distinguished", "dsitinguished", "held inapplicable", "held not applicable", "overruled",
    "disapproved", "held not correct law", "held, not good law", "not a binding precedent",
    "held per incuriam", "per incuriam", "partially overruled", "held overruled", "bad in law",
    "held, erroneous", "not followed", "doubted", "recalled",
}
NEUTRAL = {
    "referred to", "referred", "refered to", "referred on", "explained", "clarified", "considered",
    "need not be overruled", "modified and clarified to an extent", "noticed", "cited",
}

TREATMENT_RE = re.compile(
    r"\s[–—-]\s*((?:held,?\s)?(?:not\s)?[a-z][a-z ,]{2,40}?)\s*(?:\.|;|$)(?=\s|$)"
)

CITATION_PATTERNS = [
    ("scr", re.compile(r"\[(\d{4})\]\s*(\d{1,2})?\s*(?:Suppl?\.?\s*)?S\.?\s?C\.?\s?R\.?\s*(\d{1,4})")),
    ("scc", re.compile(r"\((\d{4})\)\s*(\d{1,2}|Supp\.?\s?\d?)\s*S\.?\s?C\.?\s?C\.?\s*(\d{1,4})")),
    ("air", re.compile(r"AIR\s*(\d{4})\s*SC\s*(\d{1,4})")),
    ("insc", re.compile(r"(\d{4})\s*INSC\s*(\d{1,5})")),
    ("online", re.compile(r"(\d{4})\s*SCC\s*OnLine\s*\w+\s*(\d{1,6})")),
]


def polarity(treatment: str) -> str:
    t = re.sub(r"\s+", " ", treatment.strip().lower())
    if t in POSITIVE or t.startswith(("relied", "followed", "approved", "affirmed")):
        return "positive"
    if t in NEGATIVE or t.startswith(("distinguish", "overrul", "held inapplicable", "disapprov")):
        return "negative"
    if "overruled" in t or "inapplicable" in t:
        return "negative"
    return "neutral"


def normalize_party_name(name: str) -> str:
    name = name.lower().replace("’", "'")
    name = re.sub(r"\(dead\)|through lrs?\.?|thr\.? lrs?|& ors\.?|and ors\.?|& anr\.?|and anr\.?|"
                  r"and others|& others|and another|m/s\.?|\bors\b|\banr\b", " ", name)
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def case_key(case_name: str) -> str:
    """Stable matching key: normalized 'first party v second party'."""
    parts = re.split(r"\s+(?:v\.?|vs\.?|versus)\s+", case_name, maxsplit=1, flags=re.I)
    if len(parts) == 2:
        return normalize_party_name(parts[0]) + " v " + normalize_party_name(parts[1])
    return normalize_party_name(case_name)


def _split_entry(entry: str):
    entry = entry.strip(" ;.")
    first_cite = None
    cites = []
    for kind, pattern in CITATION_PATTERNS:
        for m in pattern.finditer(entry):
            cites.append((kind, re.sub(r"\s+", " ", m.group(0)).strip()))
            first_cite = m.start() if first_cite is None else min(first_cite, m.start())
    name = entry[:first_cite] if first_cite is not None else entry
    name = re.sub(r"[,:;\s]+$", "", name).strip()
    return name, cites


def scr_key(citation: str):
    m = re.match(r"\[(\d{4})\]\s*(\d{1,2})?\s*(?:Suppl?\.?\s*)?S\.?\s?C\.?\s?R\.?\s*(\d{1,4})", citation)
    if not m:
        return None
    return f"[{m.group(1)}] {m.group(2) or ''} SCR {m.group(3)}".replace("  ", " ")


def parse_case_law_cited(block: str) -> list[dict]:
    block = re.sub(r"\s+", " ", block or "").strip()
    if not block:
        return []
    out = []
    position = 0
    for m in TREATMENT_RE.finditer(block):
        group_text = block[position:m.start()]
        position = m.end()
        treatment = m.group(1).strip(" ,")
        if len(treatment.split()) > 6:
            continue
        for raw in re.split(r";\s*", group_text):
            name, cites = _split_entry(raw)
            if len(name) < 5 or not re.search(r"\sv\.?\s|\svs\.?\s|versus|\bIn re\b|\bre:", name, re.I):
                continue
            out.append({
                "name": name,
                "key": case_key(name),
                "citations": [c for _, c in cites],
                "scr": next((scr_key(c) for k, c in cites if k == "scr"), None),
                "treatment": treatment,
                "polarity": polarity(treatment),
            })
    return out


VERSUS_RE = re.compile(
    r"([A-Z][\w.&@'’()-]*(?:\s+(?:[A-Z][\w.&@'’()-]*|of|and|&|the|@))*)\s+(?:v\.?|vs\.?|versus)\s+"
    r"((?:State|Union|[A-Z])[\w.&@'’()-]*(?:\s+(?:[A-Z][\w.&@'’()-]*|of|and|&|the))*)"
)


def find_named_precedents(query: str) -> list[str]:
    """Case names written in a query ("Arnesh Kumar v. State of Bihar")."""
    return [case_key(f"{a} v. {b}") for a, b in VERSUS_RE.findall(query)]


GENERIC_PARTY = {"state", "union of india", "union", "the state", "state of", "commissioner", "government"}


def key_matches(query_key: str, candidate_key: str) -> float:
    """Fuzzy match between two case keys in [0, 1]."""
    q_parts = query_key.split(" v ")
    c_parts = candidate_key.split(" v ")
    q_first = set(q_parts[0].split())
    c_first = set(c_parts[0].split())
    if not q_first or not c_first:
        return 0.0
    first = len(q_first & c_first) / len(q_first)
    if first < 0.99 and not (first >= 0.6 and len(q_first) >= 2):
        return 0.0
    if len(q_parts) == 2 and len(c_parts) == 2:
        q_second, c_second = set(q_parts[1].split()), set(c_parts[1].split())
        second = len(q_second & c_second) / max(1, len(q_second))
        return 0.6 * first + 0.4 * second
    return 0.8 * first
