from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase

from ..config import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME


class Neo4jGraphService:
    def __init__(self) -> None:
        self.uri = NEO4J_URI.strip()
        self.username = NEO4J_USERNAME.strip()
        self.password = NEO4J_PASSWORD.strip()
        self.database = NEO4J_DATABASE.strip()
        self._driver = None

    def is_configured(self) -> bool:
        return bool(self.uri and self.username and self.password)

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def status(self) -> dict[str, Any]:
        status = {
            "configured": self.is_configured(),
            "connected": False,
            "database": self.database or "neo4j",
            "uri": self.uri,
        }
        if not status["configured"]:
            return status
        try:
            driver = self._get_driver()
            with driver.session(database=self.database or None) as session:
                session.run("RETURN 1 AS ok").single()
            status["connected"] = True
        except Exception as exc:  # pragma: no cover - depends on environment
            status["error"] = str(exc)
        return status

    def sync_version_graph(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._ensure_configured()
        version_meta = {
            "knowledge_base_id": payload["knowledge_base_id"],
            "knowledge_base_name": payload["knowledge_base_name"],
            "version_id": payload["version_id"],
            "version_label": payload["version_label"],
            "summary": payload.get("summary", ""),
            "domain": payload.get("domain", ""),
            "source": payload.get("source", ""),
            "owner": payload.get("owner", ""),
            "layers": payload.get("layers", []),
        }
        nodes = payload.get("nodes", [])
        edges = payload.get("edges", [])
        with self._get_driver().session(database=self.database or None) as session:
            session.execute_write(self._replace_graph_tx, version_meta, nodes, edges)
        return {
            "knowledge_base_id": version_meta["knowledge_base_id"],
            "version_id": version_meta["version_id"],
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    def get_version_visualization(self, knowledge_base_id: str, version_id: str) -> dict[str, Any]:
        self._ensure_configured()
        with self._get_driver().session(database=self.database or None) as session:
            version = session.execute_read(self._get_version_meta_tx, knowledge_base_id, version_id)
            if not version:
                raise ValueError("Neo4j 中未找到该知识图谱版本。")
            nodes = session.execute_read(self._get_nodes_tx, knowledge_base_id, version_id)
            relationships = session.execute_read(self._get_edges_tx, knowledge_base_id, version_id)
        metrics = {
            "entities": len(nodes),
            "relations": len(relationships),
            "sources": 1 if nodes or relationships else 0,
        }
        return {
            "version": version,
            "nodes": nodes,
            "relationships": relationships,
            "metrics": metrics,
        }

    def _ensure_configured(self) -> None:
        if not self.is_configured():
            raise RuntimeError("Neo4j 连接未配置，请设置 NEO4J_URI、NEO4J_USERNAME、NEO4J_PASSWORD。")

    def _get_driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        return self._driver

    @staticmethod
    def _replace_graph_tx(tx, version_meta: dict[str, Any], nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
        tx.run(
            """
            MATCH (n:KGNode {knowledge_base_id: $knowledge_base_id, version_id: $version_id})
            DETACH DELETE n
            """,
            knowledge_base_id=version_meta["knowledge_base_id"],
            version_id=version_meta["version_id"],
        )
        tx.run(
            """
            MERGE (v:KGVersion {knowledge_base_id: $knowledge_base_id, version_id: $version_id})
            SET v.knowledge_base_name = $knowledge_base_name,
                v.version_label = $version_label,
                v.summary = $summary,
                v.domain = $domain,
                v.source = $source,
                v.owner = $owner,
                v.layers = $layers
            """,
            **version_meta,
        )
        if nodes:
            tx.run(
                """
                UNWIND $nodes AS node
                MERGE (n:KGNode {
                  knowledge_base_id: $knowledge_base_id,
                  version_id: $version_id,
                  node_id: node.id
                })
                SET n.label = node.label,
                    n.node_type = node.type
                WITH n
                MATCH (v:KGVersion {knowledge_base_id: $knowledge_base_id, version_id: $version_id})
                MERGE (v)-[:HAS_NODE]->(n)
                """,
                knowledge_base_id=version_meta["knowledge_base_id"],
                version_id=version_meta["version_id"],
                nodes=nodes,
            )
        if edges:
            tx.run(
                """
                UNWIND $edges AS edge
                MATCH (source:KGNode {
                  knowledge_base_id: $knowledge_base_id,
                  version_id: $version_id,
                  node_id: edge.source
                })
                MATCH (target:KGNode {
                  knowledge_base_id: $knowledge_base_id,
                  version_id: $version_id,
                  node_id: edge.target
                })
                MERGE (source)-[r:KG_REL {
                  knowledge_base_id: $knowledge_base_id,
                  version_id: $version_id,
                  edge_id: edge.id
                }]->(target)
                SET r.relation = edge.relation
                """,
                knowledge_base_id=version_meta["knowledge_base_id"],
                version_id=version_meta["version_id"],
                edges=edges,
            )

    @staticmethod
    def _get_version_meta_tx(tx, knowledge_base_id: str, version_id: str) -> dict[str, Any] | None:
        record = tx.run(
            """
            MATCH (v:KGVersion {knowledge_base_id: $knowledge_base_id, version_id: $version_id})
            RETURN v {
              .knowledge_base_id,
              .knowledge_base_name,
              .version_id,
              .version_label,
              .summary,
              .domain,
              .source,
              .owner,
              .layers
            } AS version
            """,
            knowledge_base_id=knowledge_base_id,
            version_id=version_id,
        ).single()
        return record["version"] if record else None

    @staticmethod
    def _get_nodes_tx(tx, knowledge_base_id: str, version_id: str) -> list[dict[str, Any]]:
        records = tx.run(
            """
            MATCH (n:KGNode {knowledge_base_id: $knowledge_base_id, version_id: $version_id})
            RETURN n.node_id AS id, n.label AS label, n.node_type AS type
            ORDER BY n.label
            """,
            knowledge_base_id=knowledge_base_id,
            version_id=version_id,
        )
        return [dict(record) for record in records]

    @staticmethod
    def _get_edges_tx(tx, knowledge_base_id: str, version_id: str) -> list[dict[str, Any]]:
        records = tx.run(
            """
            MATCH (source:KGNode {knowledge_base_id: $knowledge_base_id, version_id: $version_id})
              -[r:KG_REL {knowledge_base_id: $knowledge_base_id, version_id: $version_id}]->
              (target:KGNode {knowledge_base_id: $knowledge_base_id, version_id: $version_id})
            RETURN r.edge_id AS id,
                   source.node_id AS source,
                   target.node_id AS target,
                   r.relation AS relation
            ORDER BY r.edge_id
            """,
            knowledge_base_id=knowledge_base_id,
            version_id=version_id,
        )
        return [dict(record) for record in records]
