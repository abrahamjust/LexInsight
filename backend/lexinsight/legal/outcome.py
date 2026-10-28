"""Outcome labels for judgments and outcome requirements in queries.

Two levels are kept apart:
- procedural: how the proceeding before the Supreme Court ended
  (allowed / partly allowed / dismissed / disposed / referred), read from
  the editor's "Result of the case" line, with the metadata disposal
  nature as fallback;
- substantive: what happened to the underlying matter (acquittal,
  conviction upheld, bail granted, proceedings quashed, ...), read from the
  operative final paragraphs and the "Held" headnotes with cue phrases.

The procedural result alone is never used to infer the substantive one: an
appeal being "dismissed" does not say whether a conviction or an acquittal
was upheld.
"""

import re

PROCEDURAL_RULES = [
    ("partly_allowed", r"partly allowed|partly|allowed in part"),
    ("allowed", r"\ballowed\b|\bgranted\b|\bset aside\b|\bquashed\b"),
    ("dismissed", r"\bdismissed\b|\brejected\b"),
    ("referred", r"referred|larger bench|placed before|reference answered|answered"),
    ("disposed", r"disposed|directions issued|closed|concluded|listed"),
]


def procedural_outcome(result: str, disposal_nature: str | None) -> str | None:
    for source in (result or "", disposal_nature or ""):
        text = source.lower()
        for label, pattern in PROCEDURAL_RULES:
            if re.search(pattern, text):
                return label
    return None


SUBSTANTIVE_CUES = {
    "acquittal": [
        r"\b(?:are|is|stand|stands|be|hereby)\s+acquitted\b",
        r"\bacquitted of (?:all|the) charges?\b",
        r"\bconvictions? (?:and sentences? )?(?:is |are |stands? |is hereby |are hereby )?(?:set aside|quashed)\b",
        r"\bwe set aside the (?:judgment of )?conviction\b",
        r"\bentitled to (?:an )?acquittal\b",
        r"\bacquittal (?:is|stands) (?:restored|upheld|affirmed|confirmed)\b",
    ],
    "conviction_upheld": [
        r"\bconvictions? (?:of the appellants? )?(?:is|are|stands?) (?:hereby )?(?:upheld|affirmed|maintained|confirmed|sustained)\b",
        r"\bwe (?:uphold|affirm|confirm|maintain) the conviction\b",
        r"\bno (?:ground|reason|infirmity) to interfere with the (?:judgment of |order of )?conviction\b",
    ],
    "sentence_modified": [
        r"\bsentences? [^.;]{0,60}(?:is|are|stands?) (?:hereby )?(?:reduced|modified|commuted)\b",
        r"\bcommute(?:d)? the (?:death )?sentence\b", r"\b(?:we )?reduce the sentence\b",
        r"\bsentence[^.;]{0,40}period already undergone\b",
    ],
    "bail_granted": [
        r"\b(?:is|are|be|shall be) (?:hereby )?(?:released|enlarged) on bail\b",
        r"\bdirected to be released on (?:regular |interim )?bail\b",
        r"\bbail is (?:hereby )?granted to\b",
        r"\bgrant(?:ed)? (?:of )?(?:regular |interim |anticipatory )?bail to the (?:appellant|accused|petitioner|respondent)\b",
        r"\bentitled to be released on bail\b",
        r"\banticipatory bail (?:is|stands) (?:hereby )?granted\b",
    ],
    "bail_refused": [
        r"\bbail (?:granted [^.;]{0,60})?(?:is|stands|was rightly) (?:hereby )?(?:cancelled|rejected|refused|denied)\b",
        r"\bwe cancel the bail\b", r"\bnot entitled to (?:the )?(?:grant of )?(?:regular |anticipatory )?bail\b",
        r"\bbail (?:application|plea|prayer) (?:is|stands) (?:hereby )?rejected\b",
    ],
    "proceedings_quashed": [
        r"\b(?:fir|f\.i\.r\.|proceedings?|complaint|criminal case|chargesheet|charge sheet|summoning order|prosecutions?)"
        r"[^.;]{0,80}?(?:is|are|stands?|stand|hereby|thus|be) (?:hereby |accordingly )?quashed\b",
        r"\bwe quash the (?:fir|proceedings|complaint|criminal proceedings|summoning order)\b",
    ],
    "remanded": [
        r"\b(?:is|are|stands?|hereby|accordingly|be) (?:hereby )?(?:remanded|remitted)\b",
        r"\b(?:matter|case|appeal) (?:is )?remanded\b", r"\bwe (?:remand|remit) the (?:matter|case)\b",
    ],
    "compensation_enhanced": [
        r"\bcompensation (?:is|stands) (?:hereby )?(?:enhanced|increased)\b", r"\bwe enhance the (?:amount of )?compensation\b",
        r"\bentitled to (?:an )?enhanced compensation\b",
    ],
    "award_set_aside": [r"\b(?:arbitral )?award (?:is|stands) (?:hereby )?set aside\b"],
    "statute_upheld": [r"\bconstitutional validity [^.;]{0,60}(?:is|stands) upheld\b", r"\b(?:is|are) not ultra vires\b",
                       r"\b(?:is|are) intra vires\b", r"\bwe uphold the (?:constitutional )?validity\b"],
    "statute_struck_down": [r"\b(?:is|are|stands|hereby) struck down\b", r"\b(?:is|are|declared) unconstitutional\b"],
}
_COMPILED = {label: [re.compile(p, re.I) for p in pats] for label, pats in SUBSTANTIVE_CUES.items()}

# A cue inside a hedged, negated, hypothetical or conditional clause is not
# an outcome ("we do not find that the matter needs to be remanded",
# "open to the prosecution to seek cancellation of bail").
HEDGE_RE = re.compile(
    r"\b(?:not|no|neither|nor|whether|if|unless|would|could|might|had been|in the event|open to|breach|"
    r"entail|seek|move the|liberty to|in case|question|contend\w*|submi\w*|argu\w*|prayed|prayer for|sought)\b",
    re.I,
)
# Actions attributed to a lower forum describe procedural history, not the
# Supreme Court's own outcome.
LOWER_FORUM_RE = re.compile(
    r"\b(?:trial court|sessions (?:court|judge)|district (?:and sessions )?judge|magistrate|tribunal|"
    r"single judge|division bench|appellate court|first appellate|high court|commission|ncdrc|nclat|nclt)\b"
    r"[^.;–]{0,80}\b(?:set aside|acquitted|remanded|remitted|quashed|upheld|granted|rejected|cancelled|reduced)",
    re.I,
)


def _segments(text: str) -> list[str]:
    # headnotes separate propositions with en dashes rather than full stops
    parts = re.split(r"(?<=[.;:])\s+|\s+[–—]\s+", text)
    return [p for p in parts if p.strip()]


def _hit(patterns, text: str) -> bool:
    for seg in _segments(text):
        if HEDGE_RE.search(seg) or LOWER_FORUM_RE.search(seg):
            continue
        if any(p.search(seg) for p in patterns):
            return True
    return False


CONFLICTS = {
    "acquittal": {"conviction_upheld"},
    "conviction_upheld": {"acquittal"},
    "bail_granted": {"bail_refused"},
    "bail_refused": {"bail_granted"},
    "statute_upheld": {"statute_struck_down"},
    "statute_struck_down": {"statute_upheld"},
    "allowed": {"dismissed"},
    "dismissed": {"allowed", "partly_allowed"},
    "partly_allowed": {"dismissed"},
}


def substantive_outcomes(judgment_tail: str, held_text: str) -> dict:
    """Return {label: confidence}. Final paragraphs weigh more than headnotes."""
    found = {}
    for label, patterns in _COMPILED.items():
        tail_hit = _hit(patterns, judgment_tail)
        head_hit = _hit(patterns, held_text)
        if tail_hit and head_hit:
            found[label] = 1.0
        elif tail_hit:
            found[label] = 0.8
        elif head_hit:
            found[label] = 0.6
    if "acquittal" in found and "conviction_upheld" in found:
        # Mixed results (e.g. some accused acquitted, others not): keep both
        # but mark as uncertain.
        found["acquittal"] = min(found["acquittal"], 0.5)
        found["conviction_upheld"] = min(found["conviction_upheld"], 0.5)
    return found


QUERY_OUTCOME_CUES = {
    "acquittal": r"acquit|acquittal (?:was )?(?:restored|upheld)|set aside (?:the )?conviction|overturn\w* (?:the )?conviction|"
                 r"conviction (?:was )?(?:set aside|overturned|reversed)|benefit of doubt (?:was )?given",
    "conviction_upheld": r"(?:upheld|upholding|confirm\w*|affirm\w*|sustain\w*|maintain\w*) (?:the )?conviction|"
                         r"conviction (?:was |is )?(?:upheld|confirmed|affirmed|sustained|maintained)",
    "bail_granted": r"(?:granted|grant of|allowed|released on) (?:anticipatory )?bail|bail (?:was )?granted",
    "bail_refused": r"(?:denied|refused|rejected|cancel\w*) (?:anticipatory )?bail|bail (?:was )?(?:denied|refused|rejected|cancelled)",
    "proceedings_quashed": r"quash\w*",
    "remanded": r"remand\w*|remit\w* back|fresh consideration",
    "compensation_enhanced": r"enhance\w* (?:the )?compensation|compensation (?:was )?enhanced|increase\w* (?:the )?compensation",
    "award_set_aside": r"set aside (?:the |an )?(?:arbitral )?award|award (?:was )?set aside",
    "statute_upheld": r"upheld (?:the )?(?:constitutional )?validity|validity (?:was )?upheld",
    "statute_struck_down": r"struck down|declared unconstitutional",
    "allowed": r"appeals? (?:was |were )?allowed|(?:succeeded|successful) (?:appeal|appellant)",
    "dismissed": r"appeals? (?:was |were )?dismissed|appeals? (?:was |were )?rejected",
}
_QUERY_COMPILED = {k: re.compile(v, re.I) for k, v in QUERY_OUTCOME_CUES.items()}


def query_outcome_requirements(query: str) -> list[str]:
    found = [label for label, pattern in _QUERY_COMPILED.items() if pattern.search(query)]
    # "quash" inside "refused to quash" is a request for the opposite.
    if "proceedings_quashed" in found and re.search(r"(refus|declin|not)\w*\s+to\s+quash", query, re.I):
        found.remove("proceedings_quashed")
    return found


def outcome_compatibility(requirements: list[str], doc_outcome: dict) -> tuple[float, str]:
    """O(q,d): 1 compatible, 0 conflicting, 0.5 unknown. Returns (score, reason)."""
    if not requirements:
        return 0.5, ""
    labels = dict(doc_outcome.get("substantive", {}))
    procedural = doc_outcome.get("procedural")
    if procedural:
        labels.setdefault(procedural, 0.9)
    best, reason = None, ""
    for req in requirements:
        if req in labels:
            score = 0.5 + 0.5 * labels[req]
            if best is None or score > best:
                best, reason = score, f"outcome '{req.replace('_', ' ')}' matches request"
        elif any(c in labels for c in CONFLICTS.get(req, ())):
            conflicting = next(c for c in CONFLICTS[req] if c in labels)
            score = 0.5 - 0.5 * labels[conflicting]
            if best is None:
                best, reason = score, f"outcome '{conflicting.replace('_', ' ')}' conflicts with request"
    if best is None:
        return 0.5, "outcome could not be determined"
    return best, reason
