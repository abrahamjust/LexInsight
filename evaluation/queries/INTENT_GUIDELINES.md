# Query-intent labelling guidelines

Each query gets exactly one of five intents. Decide by asking what kind of
authority the user wants returned, not which words the query contains.

| Intent | Operational definition | Typical cues | Counter-example |
|---|---|---|---|
| `FACTUAL_SIMILARITY` | The user describes a concrete fact pattern (events, parties, circumstances) and wants judgments with similar facts. | Narrative; "a man who…", "where the husband…", "my client", "similar facts", dates, objects, relationships. | "Can a conviction rest on circumstantial evidence?" is a question of law → `LEGAL_ISSUE`. |
| `LEGAL_ISSUE` | The user asks a question of law or wants judgments on a legal principle or doctrine, without making a specific provision the centre of the request. | "Can…", "whether…", "what are the requirements for…", "principles of…", doctrine names. | "Cases interpreting Section 438 CrPC" → `STATUTE_BASED`. |
| `STATUTE_BASED` | The request is anchored on a named Act or provision: judgments applying, interpreting or deciding offences under it. | "Section 302 IPC", "Article 21", "under the NDPS Act", "interpretation of s.34 of the Arbitration Act". | A legal question that merely mentions an Act in passing → `LEGAL_ISSUE`. |
| `SUPPORTING_PRECEDENT` | The user wants authorities that support, follow, apply or rely on a stated proposition or a named precedent. | "cases relying on…", "authorities supporting…", "followed…", "precedents holding that…", "to support my argument". | "cases that distinguished X" → `CONTRASTING_PRECEDENT`. |
| `CONTRASTING_PRECEDENT` | The user wants authorities that distinguish, depart from, refuse to apply, doubt or overrule a proposition or a named precedent, or that take the contrary view. | "distinguished", "held inapplicable", "contrary view", "overruled", "refused to follow", "exceptions to…", "to counter the argument". | "Find judgments on exceptions in Section 300 IPC" is about a provision → `STATUTE_BASED`. |

Tie-breaking rules

1. Precedent intents win over the others when the user explicitly asks for
   supporting or contrary authority.
2. Between `STATUTE_BASED` and `LEGAL_ISSUE`: if removing the provision
   leaves the request meaningless, it is `STATUTE_BASED`.
3. A fact narrative followed by "what is the law" is still
   `FACTUAL_SIMILARITY` when the facts dominate the query.

Provenance: the queries in `intent_queries.json` and `retrieval_queries.json`
were authored for this project (drafted by Claude, to be reviewed by the
project author). Paraphrases of the same need share a `group` id, and the
train/test split is made by group so paraphrases never straddle the split.
