"""Code graph with 2-stage neural search (Bi-Encoder → Cross-Encoder)."""

import logging
from typing import Any, Dict, List

import networkx as nx
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger("nexusmcp.graph")


class CodeGraph:
    """Directed graph of code symbols with semantic search capabilities.

    Parameters
    ----------
    bi_encoder_model : str
        Sentence-transformer model for embedding code nodes.
    cross_encoder_model : str
        Cross-encoder model for precision re-ranking.
    """

    def __init__(
        self,
        bi_encoder_model: str = "all-MiniLM-L6-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.graph = nx.DiGraph()

        log.info("Loading Bi-Encoder: %s", bi_encoder_model)
        self.encoder = SentenceTransformer(bi_encoder_model)

        log.info("Loading Cross-Encoder: %s", cross_encoder_model)
        self.reranker = CrossEncoder(cross_encoder_model)
        log.info("Models loaded.")

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_from_extractions(self, extractions: List[tuple]) -> None:
        """Populate the graph from parser output.

        Parameters
        ----------
        extractions : list of (nodes, edges)
            One tuple per parsed file, as returned by ``PythonParser``.
        """
        # Collect all nodes and batch-encode their source text.
        all_nodes: List[dict] = []
        for nodes, _ in extractions:
            all_nodes.extend(nodes)

        if all_nodes:
            log.info("Generating embeddings for %d nodes…", len(all_nodes))
            sources = [node["source"] for node in all_nodes]
            embeddings = self.encoder.encode(sources)
            for idx, node in enumerate(all_nodes):
                node["embedding"] = embeddings[idx]

        for node in all_nodes:
            self.graph.add_node(node["id"], **node)

        # Build a name → id lookup for edge resolution.
        name_to_id: Dict[str, str] = {}
        for node_id, data in self.graph.nodes(data=True):
            if "name" in data:
                name_to_id[data["name"]] = node_id

        # Add edges, resolving CALLS targets by name.
        for _, edges in extractions:
            for edge in edges:
                source = edge["source"]
                target = edge["target"]
                edge_type = edge["type"]

                if edge_type == "CALLS":
                    lookup = target.split(".")[-1] if "." in target else target
                    if lookup in name_to_id:
                        self.graph.add_edge(source, name_to_id[lookup], type="CALLS")
                    else:
                        if not self.graph.has_node(target):
                            self.graph.add_node(target, type="external", name=target)
                        self.graph.add_edge(source, target, type="CALLS")
                else:
                    self.graph.add_edge(source, target, type=edge_type)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_codebase(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Two-stage semantic search: vector fetch → cross-encoder rerank."""
        log.info("Searching for: '%s'", query)

        # Stage 1 — Bi-Encoder vector similarity
        query_embedding = self.encoder.encode([query])[0]
        candidate_ids: List[str] = []
        candidate_embeddings = []

        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") != "external" and "embedding" in data:
                candidate_ids.append(node_id)
                candidate_embeddings.append(data["embedding"])

        if not candidate_ids:
            return []

        similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
        fetch_k = min(50, len(candidate_ids))
        stage_1_indices = np.argsort(similarities)[::-1][:fetch_k]

        stage_1_candidates = []
        cross_encoder_pairs = []
        for idx in stage_1_indices:
            data = self.graph.nodes[candidate_ids[idx]]
            stage_1_candidates.append(data)
            cross_encoder_pairs.append((query, data["source"]))

        # Stage 2 — Cross-Encoder precision reranking
        rerank_scores = self.reranker.predict(cross_encoder_pairs)
        stage_2_indices = np.argsort(rerank_scores)[::-1][:top_k]

        results = []
        for idx in stage_2_indices:
            data = stage_1_candidates[idx]
            result = {
                k: v for k, v in data.items() if k not in ("source", "embedding")
            }
            src = data["source"]
            result["snippet"] = (src[:200] + "…") if len(src) > 200 else src
            result["rerank_score"] = round(float(rerank_scores[idx]), 4)
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------

    def get_call_graph(self, function_name: str) -> Dict[str, Any]:
        """Return callers and callees of *function_name*."""
        callers: List[str] = []
        callees: List[str] = []

        target_ids = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("name") == function_name
        ]

        for tid in target_ids:
            for caller_id in self.graph.predecessors(tid):
                edge = self.graph.get_edge_data(caller_id, tid)
                if edge and edge.get("type") == "CALLS":
                    callers.append(self.graph.nodes[caller_id].get("name", caller_id))

            for callee_id in self.graph.successors(tid):
                edge = self.graph.get_edge_data(tid, callee_id)
                if edge and edge.get("type") == "CALLS":
                    callees.append(self.graph.nodes[callee_id].get("name", callee_id))

        return {"function": function_name, "called_by": callers, "calls": callees}

    def get_file_structure(self, filepath: str) -> List[Dict[str, Any]]:
        """Return all symbols defined in *filepath* (without source/embeddings)."""
        return [
            {k: v for k, v in data.items() if k not in ("source", "embedding")}
            for _, data in self.graph.nodes(data=True)
            if data.get("filepath") == filepath
        ]
