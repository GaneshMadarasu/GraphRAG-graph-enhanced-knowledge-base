"""
Medical entity normalizer.

Resolves synonyms so graph nodes are not duplicated:
  - Drug trade names → generic names  (Tylenol → acetaminophen)
  - Disease synonyms  (Type 2 Diabetes → Diabetes Mellitus Type 2)
  - Gene aliases      (TP53 → Tumor Protein P53)
  - Symptom variants  (high blood pressure → hypertension)

Strategy:
  1. Exact lookup in bundled dictionaries.
  2. Case-insensitive lookup.
  3. RapidFuzz fuzzy match (threshold ≥ 85) as last resort.
  4. If no match → return original name lowercased & title-cased.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Lookup tables ─────────────────────────────────────────────────────────────
# Format: {alias_lower: canonical_name}

DRUG_SYNONYMS: Dict[str, str] = {
    # Trade → generic
    "tylenol": "acetaminophen",
    "paracetamol": "acetaminophen",
    "panadol": "acetaminophen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "nuprin": "ibuprofen",
    "aspirin": "acetylsalicylic acid",
    "bayer aspirin": "acetylsalicylic acid",
    "coumadin": "warfarin",
    "jantoven": "warfarin",
    "glucophage": "metformin",
    "glucophage xr": "metformin",
    "fortamet": "metformin",
    "ozempic": "semaglutide",
    "wegovy": "semaglutide",
    "rybelsus": "semaglutide",
    "victoza": "liraglutide",
    "jardiance": "empagliflozin",
    "farxiga": "dapagliflozin",
    "invokana": "canagliflozin",
    "januvia": "sitagliptin",
    "tradjenta": "linagliptin",
    "actos": "pioglitazone",
    "avandia": "rosiglitazone",
    "lantus": "insulin glargine",
    "toujeo": "insulin glargine",
    "novolog": "insulin aspart",
    "humalog": "insulin lispro",
    "lasix": "furosemide",
    "aldactone": "spironolactone",
    "lopressor": "metoprolol",
    "coreg": "carvedilol",
    "zestril": "lisinopril",
    "prinivil": "lisinopril",
    "norvasc": "amlodipine",
    "lipitor": "atorvastatin",
    "crestor": "rosuvastatin",
    "zocor": "simvastatin",
    "plavix": "clopidogrel",
    "xarelto": "rivaroxaban",
    "eliquis": "apixaban",
    "pradaxa": "dabigatran",
    "nexium": "esomeprazole",
    "prilosec": "omeprazole",
    "protonix": "pantoprazole",
    "zithromax": "azithromycin",
    "augmentin": "amoxicillin/clavulanate",
    "cipro": "ciprofloxacin",
    "diflucan": "fluconazole",
    "tamiflu": "oseltamivir",
    "remdesivir": "veklury",
    "doxorubicin": "adriamycin",
    "adriamycin": "doxorubicin",
    "taxol": "paclitaxel",
    "platinol": "cisplatin",
    "nolvadex": "tamoxifen",
    "arimidex": "anastrozole",
    "herceptin": "trastuzumab",
    "keytruda": "pembrolizumab",
    "opdivo": "nivolumab",
    "humira": "adalimumab",
    "enbrel": "etanercept",
    "remicade": "infliximab",
    "rituxan": "rituximab",
    "prednisone": "prednisone",
    "prednisolone": "prednisolone",
    "dexamethasone": "dexamethasone",
    "methylprednisolone": "methylprednisolone",
    "medrol": "methylprednisolone",
    # Common aliases
    "asa": "acetylsalicylic acid",
    "hctz": "hydrochlorothiazide",
    "ace inhibitor": "ACE inhibitor",
    "arb": "angiotensin receptor blocker",
    "nsaid": "nonsteroidal anti-inflammatory drug",
    "ssri": "selective serotonin reuptake inhibitor",
    "snri": "serotonin-norepinephrine reuptake inhibitor",
    "ppi": "proton pump inhibitor",
    "statin": "HMG-CoA reductase inhibitor",
}

DISEASE_SYNONYMS: Dict[str, str] = {
    "type 2 diabetes": "Diabetes Mellitus Type 2",
    "type 2 diabetes mellitus": "Diabetes Mellitus Type 2",
    "t2dm": "Diabetes Mellitus Type 2",
    "t2d": "Diabetes Mellitus Type 2",
    "diabetes type 2": "Diabetes Mellitus Type 2",
    "dm2": "Diabetes Mellitus Type 2",
    "adult-onset diabetes": "Diabetes Mellitus Type 2",
    "type 1 diabetes": "Diabetes Mellitus Type 1",
    "type 1 diabetes mellitus": "Diabetes Mellitus Type 1",
    "t1dm": "Diabetes Mellitus Type 1",
    "juvenile diabetes": "Diabetes Mellitus Type 1",
    "heart failure": "Heart Failure",
    "congestive heart failure": "Heart Failure",
    "chf": "Heart Failure",
    "mi": "Myocardial Infarction",
    "heart attack": "Myocardial Infarction",
    "myocardial infarction": "Myocardial Infarction",
    "afib": "Atrial Fibrillation",
    "atrial fibrillation": "Atrial Fibrillation",
    "af": "Atrial Fibrillation",
    "hypertension": "Hypertension",
    "high blood pressure": "Hypertension",
    "htn": "Hypertension",
    "elevated blood pressure": "Hypertension",
    "ckd": "Chronic Kidney Disease",
    "chronic kidney disease": "Chronic Kidney Disease",
    "renal failure": "Chronic Kidney Disease",
    "kidney failure": "Chronic Kidney Disease",
    "breast cancer": "Breast Cancer",
    "tnbc": "Triple-Negative Breast Cancer",
    "triple negative breast cancer": "Triple-Negative Breast Cancer",
    "alzheimer's": "Alzheimer's Disease",
    "alzheimers": "Alzheimer's Disease",
    "alzheimer disease": "Alzheimer's Disease",
    "ad": "Alzheimer's Disease",
    "copd": "Chronic Obstructive Pulmonary Disease",
    "chronic obstructive pulmonary disease": "Chronic Obstructive Pulmonary Disease",
    "obesity": "Obesity",
    "overweight": "Obesity",
    "cvd": "Cardiovascular Disease",
    "cardiovascular disease": "Cardiovascular Disease",
    "cad": "Coronary Artery Disease",
    "coronary artery disease": "Coronary Artery Disease",
    "pancreatitis": "Pancreatitis",
    "lactic acidosis": "Lactic Acidosis",
    "hyperglycemia": "Hyperglycemia",
    "hypoglycemia": "Hypoglycemia",
    "dyslipidemia": "Dyslipidemia",
    "hyperlipidemia": "Dyslipidemia",
    "metabolic syndrome": "Metabolic Syndrome",
    "depression": "Major Depressive Disorder",
    "major depressive disorder": "Major Depressive Disorder",
    "mdd": "Major Depressive Disorder",
}

GENE_SYNONYMS: Dict[str, str] = {
    "brca-1": "BRCA1",
    "brca1 gene": "BRCA1",
    "breast cancer gene 1": "BRCA1",
    "brca-2": "BRCA2",
    "brca2 gene": "BRCA2",
    "breast cancer gene 2": "BRCA2",
    "tp53": "TP53",
    "p53": "TP53",
    "tumor suppressor p53": "TP53",
    "ampk": "AMPK",
    "amp-activated protein kinase": "AMPK",
    "prkaa1": "AMPK",
    "egfr gene": "EGFR",
    "her1": "EGFR",
    "her2": "ERBB2",
    "erbb2": "ERBB2",
    "her-2/neu": "ERBB2",
    "kras": "KRAS",
    "k-ras": "KRAS",
    "braf": "BRAF",
    "b-raf": "BRAF",
    "apoe": "APOE",
    "apolipoprotein e": "APOE",
    "ins gene": "INS",
    "insulin gene": "INS",
}

SYMPTOM_SYNONYMS: Dict[str, str] = {
    "high blood pressure": "Hypertension",
    "elevated blood pressure": "Hypertension",
    "chest pain": "Chest Pain",
    "chest tightness": "Chest Pain",
    "dyspnea": "Shortness of Breath",
    "shortness of breath": "Shortness of Breath",
    "sob": "Shortness of Breath",
    "edema": "Edema",
    "swelling": "Edema",
    "peripheral edema": "Edema",
    "fatigue": "Fatigue",
    "tiredness": "Fatigue",
    "nausea": "Nausea",
    "vomiting": "Nausea and Vomiting",
    "n/v": "Nausea and Vomiting",
    "bleeding": "Hemorrhage",
    "hemorrhage": "Hemorrhage",
    "bruising": "Hemorrhage",
    "weight loss": "Weight Loss",
    "weight gain": "Weight Gain",
    "polyuria": "Frequent Urination",
    "polydipsia": "Excessive Thirst",
    "hyperglycemia": "High Blood Sugar",
    "hypoglycemia": "Low Blood Sugar",
    "dizziness": "Dizziness",
    "syncope": "Fainting",
    "palpitations": "Palpitations",
    "inr": "INR",
}

# Combined lookup
_ALL_SYNONYMS: Dict[str, Tuple[str, str]] = {}  # lower_alias → (canonical, entity_type)

for alias, canonical in DRUG_SYNONYMS.items():
    _ALL_SYNONYMS[alias.lower()] = (canonical, "Drug")
for alias, canonical in DISEASE_SYNONYMS.items():
    _ALL_SYNONYMS[alias.lower()] = (canonical, "Disease")
for alias, canonical in GENE_SYNONYMS.items():
    _ALL_SYNONYMS[alias.lower()] = (canonical, "Gene")
for alias, canonical in SYMPTOM_SYNONYMS.items():
    _ALL_SYNONYMS[alias.lower()] = (canonical, "Symptom")


# ── Fuzzy matching (optional dep) ─────────────────────────────────────────────

try:
    from rapidfuzz import process, fuzz
    _RAPIDFUZZ_AVAILABLE = True
    _SYNONYM_KEYS = list(_ALL_SYNONYMS.keys())
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False
    logger.warning("rapidfuzz not installed — fuzzy matching disabled")


@lru_cache(maxsize=4096)
def _fuzzy_lookup(name_lower: str) -> Optional[Tuple[str, str]]:
    if not _RAPIDFUZZ_AVAILABLE or not _SYNONYM_KEYS:
        return None
    match, score, _ = process.extractOne(
        name_lower,
        _SYNONYM_KEYS,
        scorer=fuzz.token_sort_ratio,
    )
    if score >= 85:
        return _ALL_SYNONYMS[match]
    return None


# ── Public normalizer ─────────────────────────────────────────────────────────

def normalize_entity_name(name: str, entity_type: Optional[str] = None) -> str:
    """Return the canonical name for a given entity name.

    Lookup order:
      1. Exact (case-insensitive)
      2. Fuzzy match (rapidfuzz ≥85)
      3. Title-case original
    """
    name_stripped = name.strip()
    name_lower = name_stripped.lower()

    # Exact
    if name_lower in _ALL_SYNONYMS:
        canonical, _ = _ALL_SYNONYMS[name_lower]
        return canonical

    # Fuzzy
    fuzzy_result = _fuzzy_lookup(name_lower)
    if fuzzy_result:
        canonical, _ = fuzzy_result
        logger.debug("Fuzzy normalised '%s' → '%s'", name, canonical)
        return canonical

    # Default: return cleaned title case
    return _clean_name(name_stripped)


def _clean_name(name: str) -> str:
    """Remove extra whitespace and normalise punctuation."""
    name = re.sub(r"\s+", " ", name).strip()
    # Keep known all-caps acronyms (e.g., BRCA1, AMPK) as-is
    if re.match(r"^[A-Z0-9]+$", name) and len(name) <= 10:
        return name
    return name.title()


def normalize_extraction_result(result) -> None:  # type: ignore[override]
    """In-place normalise all entity names in an ExtractionResult."""
    for entity in result.entities:
        entity.name = normalize_entity_name(entity.name, entity.type)
    for rel in result.relationships:
        rel.source = normalize_entity_name(rel.source)
        rel.target = normalize_entity_name(rel.target)
