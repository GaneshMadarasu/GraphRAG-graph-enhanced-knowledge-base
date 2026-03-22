"""
Generate realistic sample medical documents for all 4 source types.

Creates cross-document connections on shared entities:
  Metformin, semaglutide, warfarin, T2DM, BRCA1, AMPK, etc.

Run: python -m scripts.seed_sample_data
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DOCUMENTS = {
    "data/research_papers/research_paper_1.txt": """\
Metformin and Cardiovascular Outcomes in Type 2 Diabetes Patients Over 65

Authors: Dr. Sarah Chen, Dr. James Okafor, Dr. Maria Rodriguez
Journal: Journal of Clinical Endocrinology & Metabolism
DOI: 10.1210/jc.2024.001234
Date: 2024-03-15

Abstract
Metformin (glucophage) remains the first-line pharmacotherapy for Type 2 Diabetes
Mellitus (T2DM). This prospective cohort study examined cardiovascular outcomes in
1,847 patients aged 65 and older receiving metformin monotherapy over 5 years.

Introduction
Type 2 Diabetes Mellitus (T2DM, also known as Diabetes Mellitus Type 2) affects
over 537 million adults worldwide. Metformin, chemically known as dimethylbiguanide,
activates AMP-activated protein kinase (AMPK), which plays a central role in glucose
homeostasis and energy regulation. AMPK activation by metformin reduces hepatic
glucose production and improves insulin sensitivity.

Methods
Patients were enrolled from three tertiary care centers between January 2019 and
December 2023. Inclusion criteria: age ≥ 65, confirmed T2DM diagnosis (ICD-10: E11),
HbA1c between 7.0% and 10.0%. Exclusion criteria: eGFR < 30 mL/min (contraindication
for metformin), pregnancy, or active liver disease.

Drug Interactions
A critical finding of this study involves metformin interactions with contrast media.
Metformin must be withheld before iodinated contrast administration due to risk of
contrast-induced nephropathy and subsequent lactic acidosis.

Metformin and alcohol consumption: concurrent use increases lactic acidosis risk.
Patients on metformin who are also prescribed cimetidine show elevated metformin
plasma levels due to renal tubular secretion competition.

In our diabetic cohort, 23% of patients were co-prescribed aspirin (acetylsalicylic
acid) for cardiovascular protection. No direct pharmacokinetic interaction was
identified, though both agents affect platelet function pathways.

Patients receiving furosemide (a loop diuretic) concurrently with metformin showed
a 12% increased risk of acute kidney injury, particularly in dehydrated states.

Side Effects and Adverse Events
The most common side effects of metformin in our cohort:
- Gastrointestinal disturbances (nausea, diarrhea): 34% of patients
- Vitamin B12 deficiency: 18% after 3+ years of use
- Lactic acidosis: 0.03 cases per 1000 patient-years (rare but serious)

Lactic acidosis risk was significantly elevated in patients with:
- Chronic kidney disease (CKD, ICD-10: N18)
- Heart failure (ICD-10: I50)
- Liver disease

AMPK Gene and Metformin Mechanism
The AMPK gene (officially PRKAA1/PRKAA2) encodes the AMP-activated protein kinase
catalytic subunit. Metformin's primary mechanism involves AMPK activation in hepatocytes,
leading to inhibition of mTOR complex 1 and reduced gluconeogenesis. Polymorphisms in
AMPK genes have been associated with variable metformin response in T2DM patients.

Results
In patients aged 65+, metformin monotherapy was associated with:
- 19% reduction in major adverse cardiovascular events (MACE)
- 11% reduction in all-cause mortality
- Significant HbA1c reduction (mean: 1.2%)
- No increased hypoglycemia vs placebo

Cardiovascular disease (CVD) remained the leading cause of mortality in the T2DM
cohort. Comorbid hypertension (ICD-10: I10) was present in 67% of patients.
Obesity (ICD-10: E66) was identified in 54% of the study population.

Conclusion
Metformin remains safe and effective for elderly T2DM patients. Contraindications
must be strictly observed, particularly in patients with CKD, heart failure, or
planned contrast procedures. The AMPK pathway represents a promising target for
next-generation T2DM therapeutics.

References
[1] UK Prospective Diabetes Study Group. NEJM. 1998.
[2] Diabetes Prevention Program. NEJM. 2002.
[3] EMPA-REG OUTCOME. NEJM. 2015.
""",

    "data/research_papers/research_paper_2.txt": """\
BRCA1 and BRCA2 Mutations in Triple-Negative Breast Cancer: Treatment Implications

Authors: Dr. Lisa Park, Dr. Ahmed Hassan, Dr. Jennifer Moore
Journal: Nature Medicine
DOI: 10.1038/nm.2024.0892
Date: 2024-01-20

Abstract
Triple-negative breast cancer (TNBC) accounts for 15-20% of all breast cancer diagnoses
and is associated with poor prognosis. This study examined BRCA1 and BRCA2 gene mutations
in 534 TNBC patients and their response to platinum-based chemotherapy.

Introduction
Breast cancer (ICD-10: C50) remains the most prevalent cancer in women worldwide.
Triple-negative breast cancer (TNBC) lacks estrogen receptor (ER), progesterone receptor
(PR), and HER2 (ERBB2) expression, limiting targeted therapy options.

BRCA1 (Breast Cancer Gene 1) and BRCA2 (Breast Cancer Gene 2) are tumor suppressor genes
located on chromosomes 17 and 13 respectively. BRCA1 and BRCA2 encode proteins critical
for homologous recombination DNA repair. Germline mutations in BRCA1 or BRCA2 confer
lifetime breast cancer risk of 65-80%.

Drug Treatments and Interactions
Tamoxifen (Nolvadex) is a selective estrogen receptor modulator (SERM) used for
hormone receptor-positive breast cancer. For TNBC, tamoxifen is generally not indicated
due to absent ER expression. However, in borderline cases, tamoxifen may be considered
alongside chemotherapy.

CRITICAL DRUG INTERACTION: Tamoxifen and doxorubicin (adriamycin) combination:
Concurrent administration of tamoxifen and anthracyclines such as doxorubicin has been
associated with increased cardiotoxicity risk. Cardiac monitoring is mandatory in
patients receiving this combination. Evidence from our trial confirms a 2.3-fold
increased risk of cardiomyopathy.

Doxorubicin (adriamycin) side effects include:
- Cardiotoxicity (cumulative dose-dependent)
- Myelosuppression
- Alopecia
- Nausea and vomiting

Chemotherapy Regimens for BRCA-mutated TNBC
Platinum-based agents (cisplatin, carboplatin) demonstrate superior response rates in
BRCA1/BRCA2-mutated TNBC compared to non-mutated tumors. Response rate: 68% vs 33%.

PARP inhibitors (olaparib, niraparib) are FDA-approved for BRCA-mutated breast cancer.
Olaparib is contraindicated in patients with severe renal impairment.

Clinical Trial NCT04523456
Phase III, randomized controlled trial
Status: Active
Start date: 2022-06-01
End date: 2025-12-31
Treatment: Olaparib + pembrolizumab vs standard chemotherapy in BRCA1/2-mutated TNBC
Institution: Memorial Cancer Research Center

Comorbidities and Breast Cancer
Patients with Type 2 Diabetes Mellitus (T2DM) who develop breast cancer face additional
challenges. Metformin use in T2DM patients with breast cancer has shown potential
anti-proliferative effects via the AMPK pathway, creating an interesting mechanistic
link between AMPK, BRCA1, and breast cancer pathogenesis.

Obesity (ICD-10: E66) is a significant risk factor for breast cancer recurrence and is
associated with worse outcomes in TNBC patients.

Gene-Disease Associations Summary
- BRCA1 → ASSOCIATED_WITH → Breast Cancer (ICD-10: C50)
- BRCA2 → ASSOCIATED_WITH → Breast Cancer (ICD-10: C50)
- BRCA1 → ASSOCIATED_WITH → Ovarian Cancer (ICD-10: C56)
- ERBB2 → ASSOCIATED_WITH → Breast Cancer (ICD-10: C50)
- AMPK → ASSOCIATED_WITH → Type 2 Diabetes Mellitus

Conclusion
BRCA1/2 mutation testing should be standard in all TNBC patients to guide therapy
selection. The tamoxifen-doxorubicin interaction requires careful cardiac monitoring.
PARP inhibitors represent a significant advance in BRCA-mutated TNBC treatment.
""",

    "data/internal_reports/internal_report_1.txt": """\
INTERNAL REPORT — CONFIDENTIAL
Metro General Hospital — Cardiology Department
Q3 2024 Readmission Analysis Report

Department: Cardiology
Date: 2024-10-01
Report ID: MGH-CARD-Q3-2024
Patient Cohort: Anonymized — Adult inpatients admitted for heart failure (ICD-10: I50)
Period: July 1, 2024 – September 30, 2024

Executive Summary
This report analyzes 30-day readmission rates for heart failure patients discharged
from the Cardiology Department during Q3 2024. A total of 342 patients were included
in this analysis. The 30-day readmission rate was 22.8%, exceeding the national
benchmark of 19.9%.

Patient Demographics (Anonymized)
- Age group: 65-85 years (median age: 74)
- Gender: 54% male, 46% female
- Comorbidities: Type 2 Diabetes Mellitus (T2DM): 61%, Hypertension: 78%,
  Chronic Kidney Disease: 43%, Atrial Fibrillation: 38%

Drug Therapy Analysis
Heart failure patients in this cohort were predominantly on the following medications:

1. Furosemide (Lasix): 89% of patients
   - Furosemide is the primary loop diuretic for heart failure management
   - Side effects documented in cohort: electrolyte imbalances (hypokalemia) in 31%

2. Spironolactone (Aldactone): 64% of patients
   CRITICAL DRUG INTERACTION IDENTIFIED:
   Furosemide + Spironolactone combination: While commonly co-prescribed for heart
   failure (complementary mechanisms — loop diuretic vs aldosterone antagonist),
   this combination requires close electrolyte monitoring. In our cohort, 8 patients
   (2.3%) developed hyperkalemia requiring intervention, particularly those with
   concurrent CKD. Risk is highest in elderly patients with reduced renal clearance.

3. Metformin (Glucophage) in T2DM Patients:
   Of the 61% of patients with T2DM, 78% were on metformin prior to admission.
   IMPORTANT: Metformin was withheld in 34% of heart failure admissions due to
   concerns about lactic acidosis risk in acute decompensated heart failure.
   This finding aligns with published literature (see research papers on metformin
   contraindications in heart failure).

4. Carvedilol: 71% of patients
5. Lisinopril: 58% of patients

Readmission Risk Factors Identified
High-risk factors for 30-day readmission in our cohort:
1. BNP > 500 pg/mL at discharge: OR 3.2 (p < 0.001)
2. T2DM + Heart Failure comorbidity: OR 2.1 (p < 0.01)
3. Polypharmacy (≥8 medications): OR 1.8 (p < 0.05)
4. Non-adherence to furosemide: OR 2.7 (p < 0.001)

Drug Interactions Flagged by Pharmacy Review
The following drug interactions were flagged by our pharmacy team:
- Furosemide + NSAIDs (ibuprofen/aspirin): Reduced diuretic efficacy, risk of
  acute kidney injury — 12 cases identified
- Spironolactone + ACE inhibitors (lisinopril): Hyperkalemia risk — 19 cases
- Metformin + contrast media: 7 patients required emergency metformin withholding
  pre-procedure (consistent with published contraindication guidelines)

Type 2 Diabetes and Heart Failure Comorbidity
T2DM and Heart Failure are strongly comorbid conditions. In our cohort, T2DM
was present in 61% of heart failure patients, significantly higher than the
general population prevalence of 10.5%. The interplay between insulin resistance,
hyperglycemia, and cardiac function creates a vicious cycle of worsening outcomes.
Cardiovascular disease remains the leading cause of death in T2DM patients.

Recommendations
1. Implement structured medication reconciliation for high-risk discharge patients
2. Establish electrolyte monitoring protocol for furosemide + spironolactone users
3. Create clear metformin hold/restart protocol for heart failure admissions
4. Enhance T2DM management protocols within cardiology inpatient service
5. Develop targeted readmission prevention program for T2DM + heart failure patients

Department: Cardiology | Prepared by: Quality Improvement Team
Distribution: Department Heads, CMO, Pharmacy Committee
""",

    "data/news_articles/news_article_1.txt": """\
FDA Warns About New Ozempic Drug Interactions — What Patients Need to Know

Source: MedHealth News
Publication: MedHealth Daily
Date: 2024-02-28
Journalist: Rebecca Sullivan, Health Correspondent

WASHINGTON — The Food and Drug Administration (FDA) issued a safety communication
this week warning healthcare providers about potential drug interactions with
semaglutide (Ozempic, Wegovy, Rybelsus), the blockbuster GLP-1 receptor agonist
used for Type 2 Diabetes management and weight loss.

The FDA alert specifically highlights:

INSULIN INTERACTION WARNING
Concurrent use of semaglutide (Ozempic) with insulin or insulin secretagogues
such as sulfonylureas significantly increases the risk of hypoglycemia.
Patients combining Ozempic with insulin may require insulin dose reduction.
The FDA recommends close blood glucose monitoring when initiating semaglutide
alongside existing insulin therapy.

Type 2 Diabetes (T2DM) is a chronic metabolic disease affecting approximately
38 million Americans. Semaglutide has emerged as a preferred treatment for T2DM
due to its cardiovascular benefits and significant weight loss effects (average
15% body weight reduction in the SUSTAIN trials).

PANCREATITIS RISK
The FDA has received reports of acute pancreatitis in patients taking semaglutide.
Patients with a history of pancreatitis should not use Ozempic or any GLP-1 receptor
agonist. Pancreatitis (ICD-10: K85) symptoms include severe abdominal pain radiating
to the back, nausea, and vomiting.

THYROID CANCER CONCERN
Animal studies have shown an association between semaglutide and thyroid C-cell
tumors. Semaglutide is contraindicated in patients with a personal or family history
of medullary thyroid carcinoma (MTC) or Multiple Endocrine Neoplasia type 2 (MEN2).

METFORMIN COMBINATION THERAPY
Ozempic is commonly prescribed alongside metformin for T2DM. The FDA notes that
this combination is generally safe and additive in glycemic control. However,
monitoring for gastrointestinal side effects (nausea, vomiting, diarrhea) is
important as both drugs share GI adverse effects.

WEIGHT MANAGEMENT AND CARDIOVASCULAR OUTCOMES
The SUSTAIN-6 trial demonstrated that semaglutide reduced major adverse cardiovascular
events (MACE) by 26% in high-risk T2DM patients. This makes semaglutide particularly
valuable for T2DM patients with established cardiovascular disease.

Regarding cardiovascular safety: semaglutide does not interact negatively with
commonly used cardiovascular drugs (statins, beta-blockers, ACE inhibitors).

PATIENT GUIDANCE
"Patients should not stop semaglutide without consulting their physician," said
Dr. Michael Torres, endocrinologist at Metro General Hospital. "The benefits for
T2DM management are well-established, but the drug interactions must be carefully
managed, particularly in elderly diabetic patients over 65."

The FDA reminder comes amid a surge in semaglutide prescriptions, which have
increased 400% since 2021. Ozempic (semaglutide) has largely displaced older
GLP-1 agents like liraglutide (Victoza) in clinical practice due to superior
efficacy data.

For patients currently on metformin monotherapy being transitioned to combination
therapy with Ozempic, endocrinologists recommend a gradual titration approach
starting at 0.25mg weekly.

Note: This article is for informational purposes only and does not constitute
medical advice. Patients should consult their healthcare provider.
""",

    "data/emails_slack/email_slack_1.txt": """\
FROM: Dr. Sarah Mitchell (Attending Cardiologist, Metro General Hospital)
TO: Dr. James Patterson (Clinical Pharmacist)
CC: Dr. Linda Zhao (Hematology)
DATE: 2024-11-15
SUBJECT: Urgent — Patient on Warfarin + Aspirin Combination — Bleeding Risk

Dr. Patterson,

I need your urgent input on a patient situation that came up this morning.

We have a 71-year-old patient admitted for atrial fibrillation (ICD-10: I48)
management. The patient is currently on:
- Warfarin 5mg daily (for stroke prevention in atrial fibrillation)
- Aspirin (acetylsalicylic acid) 81mg daily (prescribed by their cardiologist
  for concurrent coronary artery disease)

DRUG INTERACTION CONCERN:
As you know, warfarin + aspirin combination significantly increases bleeding risk.
Aspirin inhibits platelet aggregation while warfarin inhibits clotting factor
synthesis — combining these two creates a dual anticoagulation effect.

The patient's INR this morning was 3.8 (therapeutic range for AF: 2.0-3.0).
This elevated INR + concurrent aspirin use puts the patient at HIGH risk for
major hemorrhage, particularly GI bleeding or intracranial hemorrhage.

We need to decide: continue both, stop aspirin, or switch to a DOAC (direct
oral anticoagulant like apixaban or rivaroxaban)?

The literature suggests that the warfarin + aspirin combination is only justified
when clear benefit outweighs risk (e.g., mechanical heart valves + AF). For this
patient with AF + stable CAD, the WOEST trial data suggests aspirin can be safely
discontinued.

ADDITIONAL CONTEXT:
- Patient also has Type 2 Diabetes (on metformin — acceptable with current renal function)
- Hypertension (on lisinopril)
- The metformin + warfarin combination: metformin can potentiate the anticoagulant
  effect of warfarin in some patients by affecting gut flora and vitamin K metabolism.
  Worth monitoring INR more closely.

Please advise on:
1. Recommended INR monitoring frequency
2. Whether to continue aspirin or switch to antiplatelet-free anticoagulation
3. Any adjustments needed to the warfarin dose

Dr. Mitchell

---

REPLY FROM: Dr. James Patterson (Clinical Pharmacist)
DATE: 2024-11-15

Dr. Mitchell,

Thank you for flagging this. This is a critical drug interaction that needs
immediate attention.

WARFARIN + ASPIRIN: HIGH BLEEDING RISK
The combination of warfarin and aspirin (acetylsalicylic acid) is associated with
a 3-fold increase in major bleeding events compared to warfarin alone. The INR of
3.8 is concerning — this patient is supratherapeutic.

My recommendations:
1. HOLD aspirin immediately pending reassessment
2. Do NOT hold warfarin without hematology input (Dr. Zhao's team)
3. INR monitoring: daily until INR returns to 2.0-3.0 therapeutic range
4. Consider warfarin dose reduction to 4mg daily

METFORMIN INTERACTION NOTE:
You raised a valid point about metformin + warfarin. The interaction is classified
as moderate severity. Metformin does not directly inhibit warfarin metabolism
(CYP2C9), but there are case reports of enhanced anticoagulant effect. Given
this patient's T2DM and current INR situation, we should monitor INR after any
metformin dose change.

THREAD_ID: MGH-PHARM-2024-1145
CASE_REFERENCE: AF-MANAGEMENT-ELDERLY-2024

Also noting for the record: furosemide (if this patient is on it for any reason)
can also potentiate warfarin effect through protein binding displacement.

Kind regards,
Dr. James Patterson
Clinical Pharmacist, Metro General Hospital

---

SLACK MESSAGE - #clinical-alerts channel
Dr. Linda Zhao [2024-11-15 14:23]:
Confirming receipt. Reviewing warfarin + aspirin case. Standard protocol:
atrial fibrillation patients with stable CAD should transition to warfarin
monotherapy per WOEST and AUGUSTUS trial guidelines. Will consult with Dr. Mitchell.
Bleeding risk outweighs benefit of dual therapy in this case.

Also flagging: any patient on warfarin should be counseled to avoid NSAIDs
(ibuprofen, naproxen) as these significantly elevate bleeding risk. Aspirin
at 81mg is a gray area but the current supratherapeutic INR makes it untenable.
""",
}


def seed() -> None:
    base = Path(__file__).parent.parent
    created = []
    for rel_path, content in DOCUMENTS.items():
        full_path = base / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content.strip(), encoding="utf-8")
        created.append(str(full_path.relative_to(base)))
        print(f"✅ Created: {rel_path}")

    print(f"\n📄 {len(created)} sample documents created in healthcare_graphrag/data/")
    print("\nCross-document connections seeded:")
    print("  • Metformin  → research_paper_1 + internal_report_1 + news_article_1 + email_slack_1")
    print("  • T2DM       → research_paper_1 + research_paper_2 + internal_report_1 + news_article_1 + email_slack_1")
    print("  • BRCA1/2    → research_paper_2")
    print("  • AMPK gene  → research_paper_1 + research_paper_2")
    print("  • Warfarin   → email_slack_1")
    print("  • Aspirin    → research_paper_1 + internal_report_1 + email_slack_1")
    print("  • Semaglutide → news_article_1")
    print("  • Furosemide  → internal_report_1 + email_slack_1")
    print("  • Heart Failure → research_paper_1 + internal_report_1")
    print("\nRun ingestion: python -m scripts.ingest --data-dir data/")


if __name__ == "__main__":
    seed()
