"""
Pre-built Cypher query templates for medical graph traversal.

Each template is async and returns a list of dicts ready for the hybrid retriever.
All queries include a ``graph_hop_score`` field (1 / hop_count) for scoring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.utils.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class MedicalGraphRetriever:
    def __init__(self, client: Neo4jClient) -> None:
        self._neo4j = client

    # ── a) Drug interactions ──────────────────────────────────────────────────

    async def get_drug_interactions(
        self, drug_name: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (d1:Drug {name: $drug_name})-[r:INTERACTS_WITH]-(d2:Drug)
        RETURN d1.name AS drug1, d2.name AS drug2, d2.drug_class AS drug_class,
               r.evidence AS evidence, r.confidence AS confidence
        LIMIT $limit
        """
        try:
            rows = await self._neo4j.run_query(
                cypher, {"drug_name": drug_name, "limit": limit}
            )
            return [
                {**row, "graph_hop_score": 1.0, "query_type": "drug_interaction"}
                for row in rows
            ]
        except Exception as exc:
            logger.error("drug_interactions query failed: %s", exc)
            return []

    # ── b) Drug–disease contraindications ────────────────────────────────────

    async def get_drug_contraindications(
        self, drug_name: str
    ) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (d:Drug {name: $drug_name})-[r:CONTRAINDICATED_FOR]->(dis:Disease)
        RETURN d.name AS drug, dis.name AS disease,
               dis.icd10_code AS icd10_code, r.evidence AS evidence,
               r.confidence AS confidence
        """
        try:
            rows = await self._neo4j.run_query(cypher, {"drug_name": drug_name})
            return [
                {**row, "graph_hop_score": 1.0, "query_type": "contraindication"}
                for row in rows
            ]
        except Exception as exc:
            logger.error("drug_contraindications query failed: %s", exc)
            return []

    # ── c) Gene–disease associations ─────────────────────────────────────────

    async def get_gene_disease_associations(
        self, disease_name: str
    ) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (g:Gene)-[r:ASSOCIATED_WITH]->(dis:Disease)
        WHERE dis.name = $disease_name
        RETURN g.name AS gene, g.symbol AS symbol, g.function AS gene_function,
               r.evidence AS evidence, r.confidence AS confidence
        """
        try:
            rows = await self._neo4j.run_query(
                cypher, {"disease_name": disease_name}
            )
            return [
                {**row, "graph_hop_score": 1.0, "query_type": "gene_disease"}
                for row in rows
            ]
        except Exception as exc:
            logger.error("gene_disease_associations query failed: %s", exc)
            return []

    # ── d) Comorbidity chain (2-hop) ──────────────────────────────────────────

    async def get_comorbidities(
        self, disease_name: str, max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        cypher = f"""
        MATCH (d1:Disease {{name: $disease}})-[:COMORBID_WITH*1..{max_hops}]-(d2:Disease)
        RETURN DISTINCT d2.name AS comorbid_disease, d2.category AS category,
               d2.icd10_code AS icd10_code,
               {1.0 / max_hops} AS graph_hop_score
        """
        try:
            rows = await self._neo4j.run_query(cypher, {"disease": disease_name})
            return [{**row, "query_type": "comorbidity"} for row in rows]
        except Exception as exc:
            logger.error("comorbidity query failed: %s", exc)
            return []

    # ── e) Cross-document evidence ────────────────────────────────────────────

    async def get_cross_document_evidence(
        self, entity_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (c:Chunk)-[:MENTIONS]->(e)
        WHERE e.name = $entity_name
        MATCH (c)-[:PART_OF]->(doc:Document)
        RETURN c.text AS chunk_text, doc.title AS title,
               doc.type AS doc_type, doc.date AS date,
               doc.id AS doc_id, c.id AS chunk_id,
               1.0 AS graph_hop_score
        ORDER BY doc.date DESC
        LIMIT $limit
        """
        try:
            rows = await self._neo4j.run_query(
                cypher, {"entity_name": entity_name, "limit": limit}
            )
            return [{**row, "query_type": "cross_document"} for row in rows]
        except Exception as exc:
            logger.error("cross_document_evidence query failed: %s", exc)
            return []

    # ── f) Treatment pathway (multi-hop) ─────────────────────────────────────

    async def get_treatment_pathway(
        self, disease_name: str, limit: int = 15
    ) -> List[Dict[str, Any]]:
        cypher = """
        MATCH path = (dis:Disease)-[:HAS_SYMPTOM|TREATS|USED_IN*1..3]-(t:Treatment)
        WHERE dis.name = $disease_name
        WITH t, length(path) AS hops, path
        RETURN t.name AS treatment, t.type AS treatment_type,
               t.evidence_level AS evidence_level,
               hops, (1.0 / hops) AS graph_hop_score
        ORDER BY hops ASC
        LIMIT $limit
        """
        try:
            rows = await self._neo4j.run_query(
                cypher, {"disease_name": disease_name, "limit": limit}
            )
            return [{**row, "query_type": "treatment_pathway"} for row in rows]
        except Exception as exc:
            logger.error("treatment_pathway query failed: %s", exc)
            return []

    # ── Full disease profile ──────────────────────────────────────────────────

    async def get_disease_full_profile(
        self, disease_name: str
    ) -> Dict[str, Any]:
        """Aggregate all info about a disease from the graph."""
        symptoms_q = """
        MATCH (dis:Disease {name: $disease})-[:HAS_SYMPTOM]->(s:Symptom)
        RETURN s.name AS name, s.severity AS severity, s.body_system AS body_system
        """
        drugs_q = """
        MATCH (d:Drug)-[:TREATS]->(dis:Disease {name: $disease})
        RETURN d.name AS name, d.drug_class AS drug_class, d.fda_status AS fda_status
        """
        genes_q = """
        MATCH (g:Gene)-[:ASSOCIATED_WITH]->(dis:Disease {name: $disease})
        RETURN g.name AS name, g.symbol AS symbol, g.function AS gene_function
        """
        trials_q = """
        MATCH (t:Treatment)-[:USED_IN]->(ct:ClinicalTrial)
        WHERE EXISTS { MATCH (t)-[:TREATS|USED_IN]-(dis:Disease {name: $disease}) }
        RETURN ct.trial_id AS trial_id, ct.phase AS phase,
               ct.status AS status, ct.start_date AS start_date
        LIMIT 10
        """
        comorbidities_q = """
        MATCH (dis:Disease {name: $disease})-[:COMORBID_WITH]-(d2:Disease)
        RETURN DISTINCT d2.name AS name, d2.category AS category
        """
        params = {"disease": disease_name}
        try:
            symptoms = await self._neo4j.run_query(symptoms_q, params)
            drugs = await self._neo4j.run_query(drugs_q, params)
            genes = await self._neo4j.run_query(genes_q, params)
            trials = await self._neo4j.run_query(trials_q, params)
            comorbidities = await self._neo4j.run_query(comorbidities_q, params)
            return {
                "disease": disease_name,
                "symptoms": symptoms,
                "drugs": drugs,
                "genes": genes,
                "clinical_trials": trials,
                "comorbidities": comorbidities,
            }
        except Exception as exc:
            logger.error("disease_full_profile query failed: %s", exc)
            return {"disease": disease_name, "error": str(exc)}

    # ── Generic entity lookup ─────────────────────────────────────────────────

    async def find_entity(
        self, name: str, entity_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if entity_type:
            cypher = f"MATCH (n:{entity_type}) WHERE n.name =~ '(?i).*' + $name + '.*' RETURN n LIMIT 5"
        else:
            cypher = "MATCH (n) WHERE n.name =~ '(?i).*' + $name + '.*' RETURN n, labels(n) AS labels LIMIT 10"
        try:
            return await self._neo4j.run_query(cypher, {"name": name})
        except Exception as exc:
            logger.error("find_entity failed: %s", exc)
            return []

    # ── Drug full profile ─────────────────────────────────────────────────────

    async def get_drug_full_profile(self, drug_name: str) -> Dict[str, Any]:
        treats_q = "MATCH (d:Drug {name: $drug})-[:TREATS]->(dis:Disease) RETURN dis.name AS disease, dis.icd10_code AS icd10"
        interacts_q = "MATCH (d:Drug {name: $drug})-[:INTERACTS_WITH]-(d2:Drug) RETURN d2.name AS drug, d2.drug_class AS drug_class"
        side_effects_q = "MATCH (d:Drug {name: $drug})-[:CAUSES_SIDE_EFFECT]->(s:Symptom) RETURN s.name AS symptom, s.severity AS severity"
        contraindications_q = "MATCH (d:Drug {name: $drug})-[:CONTRAINDICATED_FOR]->(dis:Disease) RETURN dis.name AS disease"
        params = {"drug": drug_name}
        try:
            treats = await self._neo4j.run_query(treats_q, params)
            interacts = await self._neo4j.run_query(interacts_q, params)
            side_effects = await self._neo4j.run_query(side_effects_q, params)
            contraindications = await self._neo4j.run_query(contraindications_q, params)
            return {
                "drug": drug_name,
                "treats": treats,
                "interactions": interacts,
                "side_effects": side_effects,
                "contraindications": contraindications,
            }
        except Exception as exc:
            logger.error("drug_full_profile failed: %s", exc)
            return {"drug": drug_name, "error": str(exc)}
