"""Maintained legal dictionaries: Act aliases and legal-issue concepts.

Act identifiers are slugs of the canonical Act name (without the year).
The year is kept separately because some names were re-enacted
(e.g. Code of Criminal Procedure 1898/1973, Companies Act 1956/2013).

Newer criminal statutes (BNS, BNSS, BSA 2023) are kept as separate Acts
from IPC/CrPC/Evidence Act: provisions are renumbered and the applicable
law depends on the date of the offence, so they are never treated as
interchangeable.
"""

import re

# canonical name -> (default year, short display label, aliases)
ACTS = {
    "Penal Code": (1860, "IPC", ["IPC", "I.P.C.", "Indian Penal Code", "Penal Code"]),
    "Code of Criminal Procedure": (1973, "CrPC", [
        "CrPC", "Cr.P.C.", "Cr.P.C", "Cr. P.C.", "Cr.PC", "CrPC.", "Criminal Procedure Code",
        "Code of Criminal Procedure",
    ]),
    "Code of Civil Procedure": (1908, "CPC", ["CPC", "C.P.C.", "Civil Procedure Code", "Code of Civil Procedure"]),
    "Constitution of India": (None, "Constitution", ["Constitution of India", "Constitution", "Indian Constitution"]),
    "Evidence Act": (1872, "Evidence Act", ["Evidence Act", "Indian Evidence Act"]),
    "Bharatiya Nyaya Sanhita": (2023, "BNS", ["BNS", "Bharatiya Nyaya Sanhita", "Bhartiya Nyaya Sanhita"]),
    "Bharatiya Nagarik Suraksha Sanhita": (2023, "BNSS", [
        "BNSS", "Bharatiya Nagarik Suraksha Sanhita", "Bhartiya Nagarik Suraksha Sanhita",
    ]),
    "Bharatiya Sakshya Adhiniyam": (2023, "BSA", ["BSA", "Bharatiya Sakshya Adhiniyam", "Bhartiya Sakshya Adhiniyam"]),
    "Narcotic Drugs and Psychotropic Substances Act": (1985, "NDPS Act", [
        "NDPS", "NDPS Act", "N.D.P.S. Act", "Narcotic Drugs and Psychotropic Substances Act",
    ]),
    "Prevention of Corruption Act": (1988, "PC Act", ["PC Act", "P.C. Act", "Prevention of Corruption Act"]),
    "Protection of Children from Sexual Offences Act": (2012, "POCSO", [
        "POCSO", "POCSO Act", "Protection of Children from Sexual Offences Act",
    ]),
    "Unlawful Activities (Prevention) Act": (1967, "UAPA", [
        "UAPA", "UA(P) Act", "UAP Act", "Unlawful Activities (Prevention) Act", "Unlawful Activities Prevention Act",
    ]),
    "Prevention of Money Laundering Act": (2002, "PMLA", [
        "PMLA", "PML Act", "Prevention of Money Laundering Act", "Prevention of Money-Laundering Act",
    ]),
    "Insolvency and Bankruptcy Code": (2016, "IBC", ["IBC", "IB Code", "Insolvency and Bankruptcy Code", "Insolvency Code"]),
    "Negotiable Instruments Act": (1881, "NI Act", ["NI Act", "N.I. Act", "Negotiable Instruments Act"]),
    "Motor Vehicles Act": (1988, "MV Act", ["MV Act", "M.V. Act", "Motor Vehicles Act"]),
    "Arbitration and Conciliation Act": (1996, "Arbitration Act", [
        "Arbitration and Conciliation Act", "A&C Act", "Arbitration Act 1996",
    ]),
    "Arbitration Act": (1940, "Arbitration Act 1940", ["Arbitration Act, 1940"]),
    "Limitation Act": (1963, "Limitation Act", ["Limitation Act"]),
    "Specific Relief Act": (1963, "Specific Relief Act", ["Specific Relief Act", "SR Act"]),
    "Transfer of Property Act": (1882, "TPA", ["Transfer of Property Act", "TP Act", "T.P. Act"]),
    "Contract Act": (1872, "Contract Act", ["Contract Act", "Indian Contract Act"]),
    "Land Acquisition Act": (1894, "LA Act", ["Land Acquisition Act", "LA Act", "L.A. Act"]),
    "Right to Fair Compensation and Transparency in Land Acquisition, Rehabilitation and Resettlement Act": (
        2013, "LARR Act", ["RFCTLARR", "LARR Act", "2013 Act on land acquisition", "Right to Fair Compensation Act"],
    ),
    "Consumer Protection Act": (None, "CP Act", ["Consumer Protection Act", "CP Act"]),
    "Income Tax Act": (1961, "IT Act", ["Income Tax Act", "Income-tax Act", "Income-Tax Act", "I.T. Act"]),
    "Information Technology Act": (2000, "Information Technology Act", ["Information Technology Act"]),
    "Companies Act": (None, "Companies Act", ["Companies Act"]),
    "General Clauses Act": (1897, "General Clauses Act", ["General Clauses Act"]),
    "Hindu Marriage Act": (1955, "HMA", ["Hindu Marriage Act", "HMA"]),
    "Hindu Succession Act": (1956, "HSA", ["Hindu Succession Act"]),
    "Arms Act": (1959, "Arms Act", ["Arms Act"]),
    "Customs Act": (1962, "Customs Act", ["Customs Act"]),
    "Dowry Prohibition Act": (1961, "Dowry Prohibition Act", ["Dowry Prohibition Act"]),
    "Contempt of Courts Act": (1971, "Contempt of Courts Act", ["Contempt of Courts Act"]),
    "Industrial Disputes Act": (1947, "ID Act", ["Industrial Disputes Act", "ID Act", "I.D. Act"]),
    "Juvenile Justice (Care and Protection of Children) Act": (2015, "JJ Act", [
        "Juvenile Justice Act", "JJ Act", "Juvenile Justice (Care and Protection of Children) Act",
    ]),
    "Representation of the People Act": (1951, "RP Act", ["Representation of the People Act", "RP Act", "R.P. Act"]),
    "Registration Act": (1908, "Registration Act", ["Registration Act"]),
    "Stamp Act": (1899, "Stamp Act", ["Stamp Act", "Indian Stamp Act"]),
    "Electricity Act": (2003, "Electricity Act", ["Electricity Act"]),
    "Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act": (1989, "SC/ST Act", [
        "SC/ST Act", "SC & ST Act", "Atrocities Act", "SC/ST (Prevention of Atrocities) Act",
        "Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act",
    ]),
    "Protection of Women from Domestic Violence Act": (2005, "DV Act", [
        "DV Act", "Domestic Violence Act", "Protection of Women from Domestic Violence Act",
    ]),
    "Securitisation and Reconstruction of Financial Assets and Enforcement of Security Interest Act": (
        2002, "SARFAESI", ["SARFAESI", "SARFAESI Act", "Securitisation Act"],
    ),
    "Central Goods and Services Tax Act": (2017, "CGST Act", ["CGST Act", "CGST", "GST Act", "Central Goods and Services Tax Act"]),
    "Prevention of Food Adulteration Act": (1954, "PFA Act", ["Prevention of Food Adulteration Act", "PFA Act"]),
    "Food Safety and Standards Act": (2006, "FSS Act", ["Food Safety and Standards Act", "FSS Act"]),
    "Citizenship Act": (1955, "Citizenship Act", ["Citizenship Act"]),
    "Foreigners Act": (1946, "Foreigners Act", ["Foreigners Act"]),
    "Rights of Persons with Disabilities Act": (2016, "RPwD Act", ["Rights of Persons with Disabilities Act", "RPwD Act"]),
    "Right to Information Act": (2005, "RTI Act", ["Right to Information Act", "RTI Act"]),
    "Environment (Protection) Act": (1986, "EP Act", ["Environment (Protection) Act", "Environment Protection Act"]),
    "Forest (Conservation) Act": (1980, "FC Act", ["Forest (Conservation) Act", "Forest Conservation Act"]),
    "Industrial Employment (Standing Orders) Act": (1946, "Standing Orders Act", ["Standing Orders Act"]),
    "Payment of Gratuity Act": (1972, "Gratuity Act", ["Payment of Gratuity Act", "Gratuity Act"]),
    "Employees' Compensation Act": (1923, "EC Act", ["Employees' Compensation Act", "Workmen's Compensation Act"]),
    "Mines and Minerals (Development and Regulation) Act": (1957, "MMDR Act", ["MMDR Act", "Mines and Minerals (Development and Regulation) Act"]),
    "Central Excise Act": (1944, "Central Excise Act", ["Central Excise Act"]),
    "National Investigation Agency Act": (2008, "NIA Act", ["NIA Act", "National Investigation Agency Act"]),
    "Indian Succession Act": (1925, "Succession Act", ["Succession Act", "Indian Succession Act"]),
}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _normalize_act_name(name: str) -> str:
    name = name.replace("’", "'").replace("Bhartiya", "Bharatiya")
    name = re.sub(r"\s+", " ", name).strip(" .,;:–-")
    name = re.sub(r"^(The|THE|the)\s+", "", name)
    if name.startswith("Indian ") and name not in ("Indian Succession Act",):
        name = name[len("Indian "):]
    name = re.sub(r"^Criminal Procedure Code", "Code of Criminal Procedure", name)
    name = re.sub(r"^Civil Procedure Code", "Code of Civil Procedure", name)
    name = re.sub(r"^Income[- ][Tt]ax Act", "Income Tax Act", name)
    return name


_ALIAS_TO_CANONICAL = {}
for canonical, (_, _, aliases) in ACTS.items():
    for alias in aliases + [canonical]:
        _ALIAS_TO_CANONICAL[_normalize_act_name(alias).lower()] = canonical


def canonical_act(name: str):
    """Return (act_id, canonical name, year) for an Act mention, or None."""
    if not name:
        return None
    raw = _normalize_act_name(name)
    year_match = re.search(r",?\s*\(?(1[789]\d\d|20\d\d)\)?\s*$", raw)
    year = int(year_match.group(1)) if year_match else None
    base = raw[: year_match.start()].strip(" ,") if year_match else raw
    if not base or len(base) < 3:
        return None
    canonical = _ALIAS_TO_CANONICAL.get(base.lower())
    if canonical is None and year == 1940 and base.lower() == "arbitration act":
        canonical = "Arbitration Act"
    if canonical is None and base.lower() in ("constitution", "constitution of india"):
        canonical = "Constitution of India"
    if canonical:
        default_year = ACTS[canonical][0]
        if canonical == "Code of Criminal Procedure" and year == 1898:
            return ("code_of_criminal_procedure_1898", "Code of Criminal Procedure, 1898", 1898)
        if canonical == "Arbitration and Conciliation Act" and year == 1940:
            canonical = "Arbitration Act"
        return (slug(canonical), canonical, year or default_year)
    if not re.search(r"\b(Act|Code|Sanhita|Adhiniyam|Rules|Regulations|Order|Constitution|Ordinance|Scheme)\b", base):
        return None
    return (slug(base), base, year)


def act_label(act_id: str, fallback: str | None = None) -> str:
    for canonical, (_, label, _) in ACTS.items():
        if slug(canonical) == act_id:
            return label
    return fallback or act_id.replace("_", " ").title()


# Abbreviations that can appear right after a section number in running
# text ("Section 302 IPC", "u/s. 438 Cr.P.C.").
ABBREVIATION_PATTERN = re.compile(
    r"(?<![A-Za-z])(I\.?P\.?C\.?|Cr\.?\s?P\.?\s?C\.?|C\.?P\.?C\.?|NDPS|N\.D\.P\.S\.|PMLA|POCSO|UAPA|"
    r"IBC|BNSS|BNS|BSA|NI Act|N\.I\. Act|MV Act|M\.V\. Act|PC Act|P\.C\. Act|SARFAESI|CGST|"
    r"HMA|TPA|RP Act|JJ Act|DV Act|SC/ST Act|ID Act)(?![A-Za-z])"
)


def resolve_abbreviation(text: str):
    compact = re.sub(r"[\s.]", "", text).upper()
    table = {
        "IPC": "Penal Code", "CRPC": "Code of Criminal Procedure", "CPC": "Code of Civil Procedure",
        "NDPS": "Narcotic Drugs and Psychotropic Substances Act", "PMLA": "Prevention of Money Laundering Act",
        "POCSO": "Protection of Children from Sexual Offences Act", "UAPA": "Unlawful Activities (Prevention) Act",
        "IBC": "Insolvency and Bankruptcy Code", "BNSS": "Bharatiya Nagarik Suraksha Sanhita",
        "BNS": "Bharatiya Nyaya Sanhita", "BSA": "Bharatiya Sakshya Adhiniyam",
        "NIACT": "Negotiable Instruments Act", "MVACT": "Motor Vehicles Act",
        "PCACT": "Prevention of Corruption Act",
        "SARFAESI": "Securitisation and Reconstruction of Financial Assets and Enforcement of Security Interest Act",
        "CGST": "Central Goods and Services Tax Act", "HMA": "Hindu Marriage Act", "TPA": "Transfer of Property Act",
        "RPACT": "Representation of the People Act", "JJACT": "Juvenile Justice (Care and Protection of Children) Act",
        "DVACT": "Protection of Women from Domestic Violence Act",
        "SC/STACT": "Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act",
        "IDACT": "Industrial Disputes Act",
    }
    canonical = table.get(compact)
    if canonical is None:
        return None
    return (slug(canonical), canonical, ACTS[canonical][0])


# --------------------------------------------------
# Legal-issue concepts
# --------------------------------------------------
# concept -> (phrases that signal it, implied provisions "act_id:sNUM")
# Implied provisions are weak evidence: they are used only when the query
# does not cite a provision explicitly.

IPC, CRPC, CPC, CONST = "penal_code", "code_of_criminal_procedure", "code_of_civil_procedure", "constitution_of_india"
EVID = "evidence_act"

CONCEPTS = {
    "murder": (["murder", "culpable homicide", "homicide", "killing", "killed"], [f"{IPC}:s302", f"{IPC}:s300", f"{IPC}:s304"]),
    "circumstantial evidence": (["circumstantial evidence", "chain of circumstances", "chain of evidence",
                                 "last seen", "no eyewitness", "no eye witness", "no direct evidence"], []),
    "reasonable doubt": (["reasonable doubt", "benefit of doubt", "burden of proof", "standard of proof"], []),
    "eyewitness testimony": (["eyewitness", "eye witness", "ocular evidence", "interested witness",
                              "related witness", "hostile witness", "sole witness"], []),
    "dying declaration": (["dying declaration"], [f"{EVID}:s32"]),
    "recovery and disclosure": (["disclosure statement", "recovery of weapon", "recovery at the instance",
                                 "discovery of fact", "section 27"], [f"{EVID}:s27"]),
    "confession": (["confession", "confessional statement", "extra-judicial confession", "extra judicial confession"], [f"{EVID}:s25"]),
    "forensic and medical evidence": (["forensic", "post mortem", "postmortem", "medical evidence", "ballistic",
                                       "dna", "fsl report", "serologist"], []),
    "test identification parade": (["test identification parade", "identification parade"], []),
    "reversal of acquittal": (["reversal of acquittal", "appeal against acquittal", "reversing acquittal",
                               "reversed the acquittal", "order of acquittal", "overturn acquittal",
                               "overturned acquittal"], [f"{CRPC}:s378"]),
    "common intention": (["common intention", "common object", "unlawful assembly"], [f"{IPC}:s34", f"{IPC}:s149"]),
    "criminal conspiracy": (["criminal conspiracy", "conspiracy"], [f"{IPC}:s120B"]),
    "sentencing": (["sentence", "sentencing", "death penalty", "death sentence", "life imprisonment",
                    "remission", "premature release", "mitigating", "commutation", "rarest of rare"], []),
    "juvenility": (["juvenile", "juvenility", "child in conflict with law"], []),
    "rape and sexual offences": (["rape", "sexual assault", "sexual offence", "sexual intercourse", "outraging modesty",
                                  "penetrative"], [f"{IPC}:s376"]),
    "child sexual abuse": (["pocso", "child victim", "minor victim"], ["protection_of_children_from_sexual_offences_act:s6"]),
    "dowry death": (["dowry death", "dowry", "bride burning", "harassment for dowry"], [f"{IPC}:s304B", f"{IPC}:s498A"]),
    "cruelty": (["cruelty", "matrimonial cruelty", "harassment by husband"], [f"{IPC}:s498A"]),
    "abetment of suicide": (["abetment of suicide", "abetment to suicide", "instigation to commit suicide",
                             "suicide note"], [f"{IPC}:s306"]),
    "kidnapping and abduction": (["kidnapping", "abduction", "abducted", "ransom"], [f"{IPC}:s364", f"{IPC}:s364A"]),
    "cheating and fraud": (["cheating", "fraud", "forgery", "forged", "criminal breach of trust",
                            "misappropriation"], [f"{IPC}:s420", f"{IPC}:s406"]),
    "corruption": (["corruption", "bribe", "bribery", "illegal gratification", "disproportionate assets",
                    "public servant"], ["prevention_of_corruption_act:s7", "prevention_of_corruption_act:s13"]),
    "money laundering": (["money laundering", "proceeds of crime", "enforcement directorate"],
                         ["prevention_of_money_laundering_act:s3", "prevention_of_money_laundering_act:s45"]),
    "narcotics": (["narcotic", "ganja", "heroin", "contraband", "psychotropic", "conscious possession",
                   "drug trafficking"], ["narcotic_drugs_and_psychotropic_substances_act:s50"]),
    "terrorism and uapa": (["terrorist", "terrorism", "unlawful activities", "uapa"],
                           ["unlawful_activities_prevention_act:s43D"]),
    "anticipatory bail": (["anticipatory bail", "pre-arrest bail", "protection from arrest", "apprehending arrest"],
                          [f"{CRPC}:s438", "bharatiya_nagarik_suraksha_sanhita:s482"]),
    "bail": (["bail", "regular bail", "grant of bail", "cancellation of bail", "enlarged on bail", "release on bail"],
             [f"{CRPC}:s437", f"{CRPC}:s439"]),
    "default bail": (["default bail", "statutory bail", "chargesheet not filed", "incomplete chargesheet"], [f"{CRPC}:s167"]),
    "arrest and custody": (["arrest", "grounds of arrest", "illegal detention", "police custody", "remand"], []),
    "preventive detention": (["preventive detention", "detention order", "detenu"], [f"{CONST}:art22"]),
    "quashing of proceedings": (["quashing", "quash the fir", "quash the proceedings", "quashing of fir",
                                 "inherent powers", "abuse of process"], [f"{CRPC}:s482"]),
    "summoning and cognizance": (["summoning", "cognizance", "additional accused", "summoning order"],
                                 [f"{CRPC}:s319", f"{CRPC}:s190"]),
    "charge and discharge": (["framing of charge", "framing of charges", "discharge application", "discharged the accused",
                              "discharge of the accused", "plea of discharge"], [f"{CRPC}:s227"]),
    "fair investigation": (["further investigation", "transfer of investigation", "cbi investigation", "reinvestigation",
                            "fair investigation", "defective investigation", "investigation by the cbi", "sit"], [f"{CRPC}:s173"]),
    "cheque dishonour": (["dishonour of cheque", "cheque bounce", "dishonoured cheque", "cheque dishonour"],
                         ["negotiable_instruments_act:s138"]),
    "right to life and liberty": (["right to life", "personal liberty", "article 21", "dignity", "right to privacy"],
                                  [f"{CONST}:art21"]),
    "equality and arbitrariness": (["equality", "arbitrary", "arbitrariness", "discrimination", "article 14",
                                    "equal protection", "classification"], [f"{CONST}:art14"]),
    "freedom of speech": (["freedom of speech", "free speech", "freedom of expression", "article 19"], [f"{CONST}:art19"]),
    "reservation": (["reservation", "reserved category", "backward class", "scheduled caste", "obc", "sub-classification",
                     "creamy layer"], [f"{CONST}:art15", f"{CONST}:art16"]),
    "constitutional validity": (["constitutional validity", "ultra vires", "unconstitutional", "vires",
                                 "legislative competence", "fundamental rights"], []),
    "writ jurisdiction": (["writ petition", "writ jurisdiction", "article 226", "article 32", "mandamus", "certiorari",
                           "judicial review"], [f"{CONST}:art226", f"{CONST}:art32"]),
    "complete justice article 142": (["article 142", "complete justice"], [f"{CONST}:art142"]),
    "natural justice": (["natural justice", "audi alteram partem", "opportunity of hearing", "right to be heard",
                         "show cause notice", "bias", "reasoned order"], []),
    "service law": (["service law", "disciplinary proceedings", "departmental enquiry", "departmental inquiry",
                     "dismissal from service", "termination of service", "misconduct", "punishment of removal"], []),
    "recruitment and appointment": (["recruitment", "selection process", "appointment to the post", "eligibility criteria",
                                     "merit list", "regularisation", "regularization", "select list"], []),
    "promotion and seniority": (["promotion", "seniority", "pay scale", "pension", "retiral benefits", "gratuity"], []),
    "compassionate appointment": (["compassionate appointment", "compassionate ground"], []),
    "land acquisition compensation": (["land acquisition", "acquisition of land", "compensation for land",
                                       "market value", "solatium", "acquired land", "lapse of acquisition"],
                                      ["land_acquisition_act:s23"]),
    "specific performance": (["specific performance", "agreement to sell", "readiness and willingness",
                              "ready and willing"], ["specific_relief_act:s16"]),
    "property and title": (["title to the property", "title suit", "declaration of title", "ownership", "sale deed",
                            "adverse possession", "suit for possession", "partition", "injunction", "encroachment"], []),
    "succession and wills": (["last will", "testament", "testamentary", "succession", "inheritance", "coparcenary", "probate", "legal heirs"], []),
    "tenancy and eviction": (["eviction", "tenant", "landlord", "rent", "lease"], []),
    "limitation and delay": (["condonation of delay", "delay and laches", "time-barred", "time barred",
                              "barred by limitation", "period of limitation"],
                             ["limitation_act:s5"]),
    "res judicata and civil procedure": (["res judicata", "rejection of plaint", "order vii rule 11", "second appeal",
                                          "substantial question of law", "execution of decree", "ex parte decree"], []),
    "arbitration": (["arbitration", "arbitral award", "arbitrator", "arbitral tribunal", "set aside the award",
                     "arbitration clause"], ["arbitration_and_conciliation_act:s34", "arbitration_and_conciliation_act:s11"]),
    "contract and damages": (["breach of contract", "damages", "liquidated damages", "earnest money", "forfeiture",
                              "contractual"], []),
    "insolvency": (["insolvency", "resolution plan", "corporate insolvency", "operational creditor",
                    "financial creditor", "committee of creditors", "liquidation", "moratorium"], []),
    "banking and recovery": (["bank", "loan", "secured creditor", "auction", "sarfaesi", "recovery of debt",
                              "one time settlement"], []),
    "taxation": (["tax", "taxation", "assessment", "reassessment", "excise", "customs duty", "gst", "income tax",
                  "exemption notification", "levy"], []),
    "consumer protection": (["consumer", "deficiency in service", "unfair trade practice", "consumer commission"], []),
    "motor accident compensation": (["motor accident", "accident claim", "motor accident claims tribunal",
                                     "loss of dependency", "just compensation", "multiplier"], ["motor_vehicles_act:s166"]),
    "matrimonial and maintenance": (["divorce", "maintenance", "alimony", "custody of child", "irretrievable breakdown",
                                     "matrimonial"], [f"{CRPC}:s125"]),
    "contempt": (["contempt", "wilful disobedience", "contempt of court"], []),
    "elections": (["election", "election petition", "electoral", "nomination", "voting"], []),
    "environment": (["environment", "pollution", "forest", "wildlife", "green tribunal", "eco-sensitive"], []),
    "education": (["neet", "medical college", "university", "educational institution", "admission to the course",
                   "medical admission", "school", "students", "teachers"], []),
    "company and securities": (["sebi", "shareholder", "oppression and mismanagement", "company law", "securities",
                                "debenture"], []),
    "labour and industrial disputes": (["workman", "industrial dispute", "retrenchment", "labour court",
                                        "reinstatement", "back wages"], []),
    "citizenship and foreigners": (["citizenship", "foreigner", "illegal migrant", "foreigners tribunal"], []),
    "medical negligence": (["medical negligence", "negligence of doctor", "hospital negligence"], []),
    "disability rights": (["disability", "persons with disabilities", "benchmark disability"], []),
    "judicial precedent": (["precedent", "per incuriam", "larger bench", "binding precedent", "overruled",
                            "stare decisis", "reference to larger bench"], []),
    "interpretation of statutes": (["statutory interpretation", "purposive interpretation", "literal interpretation",
                                    "rule of interpretation", "interpretation of statutes", "retrospective operation",
                                    "retrospective effect", "plain meaning"], []),
}

_CONCEPT_PATTERNS = {
    concept: re.compile(
        r"(?<![a-z])(" + "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True)) + r")(?![a-z])"
    )
    for concept, (phrases, _) in CONCEPTS.items()
}


def find_concepts(text: str) -> dict[str, int]:
    """Count concept phrase hits in lowercase text."""
    lowered = text.lower()
    hits = {}
    for concept, pattern in _CONCEPT_PATTERNS.items():
        n = len(pattern.findall(lowered))
        if n:
            hits[concept] = n
    return hits


def implied_provisions(concepts) -> list[str]:
    out = []
    for concept in concepts:
        out.extend(CONCEPTS.get(concept, ([], []))[1])
    return out
