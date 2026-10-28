"""Regex + dictionary extraction of Acts and statutory provisions.

A provision is identified as "<act_id>:s<number>" (sections, rules) or
"constitution_of_india:art<number>". Sub-clauses are dropped for matching
("30(2)(b)" -> "30") but kept for display. Provisions whose Act cannot be
determined are stored as "?:s<number>".
"""

import re
from collections import Counter, defaultdict

from .lexicon import (
    ABBREVIATION_PATTERN, CONST, act_label, canonical_act, resolve_abbreviation,
)

# "Penal Code, 1860", "Code of Criminal Procedure, 1973", "Insolvency and
# Bankruptcy Code, 2016", "Constitution of India", "Bharatiya Nyaya Sanhita".
ACT_NAME_RE = re.compile(
    r"((?:(?:[A-Z][A-Za-z'’&.-]*|\((?:[A-Z][A-Za-z ,&'’-]*)\)|of|and|the|for|in|on|to|with|from|by),?\s){0,14}?"
    r"(?:Act|Code|Sanhita|Adhiniyam|Rules|Regulations|Order|Ordinance)"
    r"(?:,?\s?\(?(?:1[789]\d\d|20\d\d)\)?)?)"
    r"|(Constitution of India|Constitution)"
)

NUM = r"\d{1,4}(?:-\s?[A-Z]{1,2}\b|[A-Z]{1,2}\b)?(?:\s?\(\s?[\w]{1,4}\s?\))*"
LIST_SEP = r"\s*(?:,|/|&|\band\b|\bor\b|\bread with\b|\br/w\b|\bto\b)\s*"

PROVISION_RE = re.compile(
    r"(?P<marker>\bu/ss?\.|\bu/s\b|\bss\.|\bs\.|\b[Ss]ections?\b|\bSecs?\.|\b[Aa]rticles?\b|\bArts?\.|\bArt\.|"
    r"\brr?\.|\bRules?\b|\bRegulations?\b|\breg\.)"
    r"\s?(?P<nums>" + NUM + r"(?:" + LIST_SEP + NUM + r")*)"
)

ORDER_RULE_RE = re.compile(
    r"\b(?:Order|O\.)\s?(?P<order>[IVXL]{1,6}|\d{1,2})\s?(?:,\s?)?(?:Rule|r\.|R\.)\s?(?P<rule>\d{1,3}[A-Z]?)"
)

AFTER_ACT_RE = re.compile(
    r"^\s*(?:,\s*)?(?:of\s+(?:the\s+)?|under\s+(?:the\s+)?|of\s+)?"
)


def _normalize_number(raw: str) -> str:
    raw = re.sub(r"\s+", "", raw)
    base = re.match(r"\d{1,4}(?:-?[A-Z]{1,3})?", raw)
    if not base:
        return ""
    number = base.group(0).replace("-", "")
    # "120(B)" style suffix in parentheses -> "120B"
    paren = re.match(r"\d{1,4}\(([A-Z]{1,2})\)", raw)
    if paren and number.isdigit():
        number += paren.group(1)
    return number


def _split_numbers(nums: str) -> list[tuple[str, str]]:
    parts = re.split(LIST_SEP, nums)
    out = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        number = _normalize_number(part)
        if number:
            out.append((number, part))
    return out


def _act_after(text: str, pos: int):
    """Act named immediately after a provision list."""
    window = text[pos:pos + 140]
    prefix = AFTER_ACT_RE.match(window)
    rest = window[prefix.end():] if prefix else window
    abbr = ABBREVIATION_PATTERN.match(rest)
    if abbr:
        return resolve_abbreviation(abbr.group(0))
    named = ACT_NAME_RE.match(rest)
    if named:
        candidate = named.group(0)
        if candidate.strip() in ("the Code", "Code", "the Act", "Act"):
            return "GENERIC_" + candidate.strip().split()[-1].upper()
        return canonical_act(candidate)
    if re.match(r"(?:the\s+)?(?:said\s+)?(Code|Act)\b", rest):
        return "GENERIC_" + re.match(r"(?:the\s+)?(?:said\s+)?(Code|Act)", rest).group(1).upper()
    return None


def _act_before(text: str, pos: int):
    """Headnote style: 'Penal Code, 1860 – ss. 302/34 ...'."""
    window = text[max(0, pos - 160):pos]
    match = re.search(
        r"((?:(?:[A-Z][A-Za-z'’-]*|\([A-Z][^()]{0,80}\)|&),?\s(?:of\s|and\s|the\s|for\s|from\s|in\s|on\s)?){0,14}"
        r"(?:Act|Code|Procedure|Sanhita|Adhiniyam|Rules|Regulations|India)(?:,?\s?(?:1[789]\d\d|20\d\d))?)"
        r"\s*[–—-]\s*(?:[^–—]{0,40}[–—]\s*)?$",
        window,
    )
    if match:
        return canonical_act(match.group(1))
    return None


def _alias_regex():
    from .lexicon import ACTS
    long_aliases, short_aliases = set(), set()
    for canonical, (_, _, aliases) in ACTS.items():
        for alias in aliases + [canonical]:
            (long_aliases if " " in alias else short_aliases).add(alias)
    long_part = "|".join(re.escape(a) for a in sorted(long_aliases, key=len, reverse=True))
    return re.compile(r"(?<![A-Za-z])(?:Indian\s|The\s)?(" + long_part + r")(?:,?\s?(?:1[789]\d\d|20\d\d))?(?![A-Za-z])", re.I)


KNOWN_ACT_RE = _alias_regex()
YEAR_ACT_RE = re.compile(
    r"((?:[A-Z][A-Za-z'’&-]*|\([A-Z][A-Za-z ,&'’-]*\))(?:\s(?:(?:of|and|the|for|in|on|to|from|with)\s)?"
    r"(?:[A-Z][A-Za-z'’&-]*|\([A-Z][A-Za-z ,&'’-]*\))){0,12}"
    r"\s(?:Act|Code|Sanhita|Adhiniyam|Rules|Regulations|Order|Ordinance),?\s(?:1[789]\d\d|20\d\d))"
)


def extract_acts(text: str):
    """Count Act mentions by act_id; also return {act_id: (name, year)}.

    Known Acts are found through the alias dictionary (including
    abbreviations such as IPC, CrPC, NDPS). Other Acts are accepted only in
    the explicit "<Name> Act, <year>" form.
    """
    counts = Counter()
    names = {}
    spans = []
    for match in KNOWN_ACT_RE.finditer(text):
        resolved = canonical_act(match.group(0))
        if resolved:
            counts[resolved[0]] += 1
            names.setdefault(resolved[0], (resolved[1], resolved[2]))
            spans.append(match.span())
    for match in YEAR_ACT_RE.finditer(text):
        if any(a <= match.start() < b or a < match.end() <= b for a, b in spans):
            continue
        words = match.group(1).split()
        resolved = None
        # Choose the longest suffix that starts with a capitalised word and
        # does not swallow another Act name ("Constitution of India and ...").
        for i in range(len(words)):
            if not words[i][:1].isupper():
                continue
            candidate = " ".join(words[i:])
            head = " ".join(words[i:-2])
            if re.search(r"(Act|Code|India|Constitution)", head):
                continue
            resolved = canonical_act(candidate)
            if resolved:
                break
        if resolved:
            counts[resolved[0]] += 1
            names.setdefault(resolved[0], (resolved[1], resolved[2]))
    for match in ABBREVIATION_PATTERN.finditer(text):
        if any(a <= match.start() < b for a, b in spans):
            continue
        resolved = resolve_abbreviation(match.group(0))
        if resolved:
            counts[resolved[0]] += 1
            names.setdefault(resolved[0], (resolved[1], resolved[2]))
    if "constitution" in counts:
        counts[CONST] += counts.pop("constitution")
        names.setdefault(CONST, ("Constitution of India", None))
        names.pop("constitution", None)
    return counts, names


def extract_provisions(text: str, default_acts: dict | None = None):
    """Return Counter of provision ids and a display map.

    default_acts maps "CODE"/"ACT" -> act_id, used to resolve "the Code"
    or "the Act" when a document's List of Acts names exactly one.
    """
    default_acts = default_acts or {}
    counts = Counter()
    display = {}

    for match in PROVISION_RE.finditer(text):
        marker = match.group("marker").lower()
        numbers = _split_numbers(match.group("nums"))
        if not numbers:
            continue
        is_article = marker.startswith("art")
        is_rule = marker.startswith(("r", "reg"))
        if is_article:
            act = (CONST, "Constitution of India", None)
            after = _act_after(text, match.end())
            if after and not isinstance(after, str) and after[0] != CONST:
                act = after  # e.g. "Article 12 of the Convention"
        else:
            act = _act_after(text, match.end())
            if isinstance(act, str):
                generic = act.split("_")[1]
                act_id = default_acts.get(generic)
                act = (act_id, act_id, None) if act_id else None
            if act is None:
                act = _act_before(text, match.start())
        act_id = act[0] if act else "?"
        kind = "art" if is_article else ("r" if is_rule else "s")
        for number, raw in numbers:
            if len(number) > 4 or (number.isdigit() and int(number) == 0):
                continue
            pid = f"{act_id}:{kind}{number}"
            counts[pid] += 1
            if pid not in display:
                label = act_label(act_id) if act_id != "?" else ""
                prefix = {"art": "Art.", "r": "r.", "s": "s."}[kind]
                display[pid] = f"{label} {prefix} {raw.strip()}".strip()

    for match in ORDER_RULE_RE.finditer(text):
        after = _act_after(text, match.end())
        act_id = after[0] if after and not isinstance(after, str) else "code_of_civil_procedure"
        pid = f"{act_id}:o{match.group('order')}r{match.group('rule')}"
        counts[pid] += 1
        display.setdefault(pid, f"{act_label(act_id)} O.{match.group('order')} r.{match.group('rule')}")

    return counts, display


def parse_list_of_acts(acts_section: str):
    """Editor-curated 'List of Acts' -> {act_id: (name, year)}."""
    out = {}
    for part in re.split(r";|\.\s(?=[A-Z])", acts_section):
        part = re.sub(r"\s*[–—-]\s*(?:s|ss|Art|Arts|r|rr|O|Sch)\b.*$", "", part).strip(" .")
        part = re.sub(r"\s*[–—]\s.*$", "", part).strip(" .")
        if not part:
            continue
        resolved = canonical_act(part)
        if resolved:
            act_id, name, year = resolved
            if act_id == "constitution":
                act_id, name = CONST, "Constitution of India"
            out[act_id] = (name, year)
    return out


def default_act_map(listed_acts: dict) -> dict:
    codes = [a for a, (name, _) in listed_acts.items() if "Code" in name or "Sanhita" in name]
    acts = [a for a, (name, _) in listed_acts.items() if name.endswith("Act") or " Act" in name]
    out = {}
    if len(codes) == 1:
        out["CODE"] = codes[0]
    if len(acts) == 1:
        out["ACT"] = acts[0]
    return out


def provision_matches(query_pid: str, doc_pids) -> list[str]:
    """Doc provisions that satisfy a query provision.

    An unattributed query provision ("?:s302") matches the same number in
    any Act; an attributed one matches the same Act or an unattributed doc
    mention of the same number.
    """
    act, ref = query_pid.split(":", 1)
    out = []
    for pid in doc_pids:
        d_act, d_ref = pid.split(":", 1)
        if d_ref != ref:
            continue
        if act == "?" or d_act == act or d_act == "?":
            out.append(pid)
    return out


def resolve_unattributed(provisions: dict, listed_acts=None) -> dict:
    """Attach "?:N" mentions to an Act when the text makes it unambiguous.

    1. exactly one Act has an attributed provision with the same number;
    2. otherwise, the editor's List of Acts names exactly one Act.
    Values are merged by addition (ints or {"head","body"} dicts).
    """
    listed_acts = listed_acts or {}
    owners = {}
    for pid in provisions:
        act, ref = pid.split(":", 1)
        if act != "?":
            owners.setdefault(ref, set()).add(act)
    out = {}
    for pid, value in provisions.items():
        act, ref = pid.split(":", 1)
        target = pid
        if act == "?":
            same = owners.get(ref, set())
            if len(same) == 1:
                target = f"{next(iter(same))}:{ref}"
            elif not same and len(listed_acts) == 1 and ref.startswith("s"):
                target = f"{next(iter(listed_acts))}:{ref}"
        if target in out and isinstance(value, dict):
            out[target] = {k: out[target].get(k, 0) + value.get(k, 0) for k in ("head", "body")}
        elif target in out:
            out[target] += value
        else:
            out[target] = dict(value) if isinstance(value, dict) else value
    return out


def summarize_zones(zone_texts: dict, default_acts=None):
    """Extract provisions per zone and merge with salience information."""
    per_zone = {}
    display = {}
    for zone, text in zone_texts.items():
        counts, disp = extract_provisions(text, default_acts)
        per_zone[zone] = counts
        for k, v in disp.items():
            display.setdefault(k, v)
    merged = defaultdict(lambda: {"head": 0, "body": 0})
    for zone, counts in per_zone.items():
        bucket = "head" if zone in ("issue", "headnotes", "acts", "keywords") else "body"
        for pid, n in counts.items():
            merged[pid][bucket] += n
    return dict(merged), display
