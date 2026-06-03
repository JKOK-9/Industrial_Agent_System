from __future__ import annotations

import re
import uuid

from ..storage import Registry, utc_now


class GraphService:
    NODE_TYPES = "graph_node_types"
    RELATION_TYPES = "graph_relation_types"
    RELATION_RULES = "graph_relation_rules"
    TRIPLES = "graph_triples"
    ANALYSIS_TASKS = "graph_analysis_tasks"
    LIBRARIES = "graph_libraries"

    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    def list_node_types(self) -> list[dict]:
        return self.registry.list(self.NODE_TYPES)

    def save_node_types(self, items: list[dict]) -> list[dict]:
        return self.registry.replace_list(self.NODE_TYPES, items)

    def list_relation_types(self) -> list[dict]:
        return self.registry.list(self.RELATION_TYPES)

    def save_relation_types(self, items: list[dict]) -> list[dict]:
        return self.registry.replace_list(self.RELATION_TYPES, items)

    def list_relation_rules(self) -> list[dict]:
        return self.registry.list(self.RELATION_RULES)

    def save_relation_rules(self, items: list[dict]) -> list[dict]:
        return self.registry.replace_list(self.RELATION_RULES, items)

    def list_triples(self) -> list[dict]:
        return self.registry.list(self.TRIPLES)

    def save_triples(self, items: list[dict]) -> list[dict]:
        return self.registry.replace_list(self.TRIPLES, items)

    def list_analysis_tasks(self) -> list[dict]:
        items = self.registry.list(self.ANALYSIS_TASKS)
        return sorted(items, key=lambda item: item.get("updated_at", ""), reverse=True)

    def get_analysis_task(self, task_id: str) -> dict | None:
        return self.registry.get(self.ANALYSIS_TASKS, task_id)

    def delete_analysis_task(self, task_id: str) -> dict | None:
        return self.registry.delete(self.ANALYSIS_TASKS, task_id)

    def create_analysis_task(self, payload: dict) -> dict:
        task = {
            "id": f"graph-task-{uuid.uuid4().hex[:10]}",
            "filename": payload.get("filename", ""),
            "knowledge_base_name": payload.get("knowledge_base_name", ""),
            "analysis_source": payload.get("analysis_source", "heuristic"),
            "headers": payload.get("headers", []),
            "sample_rows": payload.get("sample_rows", []),
            "node_suggestions": payload.get("node_suggestions", []),
            "relation_suggestions": payload.get("relation_suggestions", []),
            "confirmation_rules": payload.get("confirmation_rules", []),
            "status": "待确认",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        return self.registry.add(self.ANALYSIS_TASKS, task)

    def update_analysis_task_confirmation_rules(self, task_id: str, items: list[dict]) -> dict | None:
        status = "已确认" if items and all(item.get("status") == "已确认" for item in items) else "待确认"
        return self.registry.update(
            self.ANALYSIS_TASKS,
            task_id,
            {
                "confirmation_rules": items,
                "status": status,
            },
        )

    def build_triples_preview(self, task_id: str) -> list[dict]:
        task = self.get_analysis_task(task_id)
        if not task:
            raise ValueError("分析任务不存在。")
        sample_rows = task.get("sample_rows", [])
        confirmation_rules = task.get("confirmation_rules", [])
        preview: list[dict] = []
        for rule in confirmation_rules:
            for index, row in enumerate(sample_rows):
                preview.append(
                    {
                        "id": f"{rule.get('id', 'rule')}-{index}",
                        "row_label": f"第 {index + 1} 行",
                        "source": row.get(rule.get("source_header", ""), "-"),
                        "source_type": rule.get("source_type", ""),
                        "relation": rule.get("relation", ""),
                        "target": row.get(rule.get("target_header", ""), "-"),
                        "target_type": rule.get("target_type", ""),
                        "status": rule.get("status", "待确认"),
                    }
                )
        return preview

    def list_libraries(self) -> list[dict]:
        items = self.registry.list(self.LIBRARIES)
        return sorted(items, key=lambda item: item.get("updated_at", ""), reverse=True)

    def build_library_version(self, payload: dict) -> dict:
        task = self.get_analysis_task(payload["task_id"])
        if not task:
            raise ValueError("分析任务不存在。")

        triples = [item for item in self.build_triples_preview(payload["task_id"]) if item.get("status") == "已确认"]
        if not triples:
            raise ValueError("当前分析任务还没有已确认的三元组，无法生成知识库版本。")

        node_map: dict[tuple[str, str], dict] = {}
        edges: list[dict] = []
        source_labels = set()
        source_types = payload.get("source_types") or ["文档", "表格", "图片"]

        for triple in triples:
            source_key = (str(triple.get("source", "")).strip(), str(triple.get("source_type", "")).strip())
            target_key = (str(triple.get("target", "")).strip(), str(triple.get("target_type", "")).strip())
            if source_key[0] and source_key not in node_map:
                node_map[source_key] = {
                    "id": f"node-{uuid.uuid4().hex[:8]}",
                    "label": source_key[0],
                    "type": source_key[1] or "实体",
                }
            if target_key[0] and target_key not in node_map:
                node_map[target_key] = {
                    "id": f"node-{uuid.uuid4().hex[:8]}",
                    "label": target_key[0],
                    "type": target_key[1] or "实体",
                }
            if source_key[0] and target_key[0]:
                edges.append(
                    {
                        "id": f"edge-{uuid.uuid4().hex[:8]}",
                        "source": node_map[source_key]["id"],
                        "target": node_map[target_key]["id"],
                        "relation": triple.get("relation", "关联"),
                    }
                )
            source_labels.add(triple.get("row_label", ""))

        libraries = self.registry.list(self.LIBRARIES)
        knowledge_base_name = payload["knowledge_base_name"].strip()
        library = next((item for item in libraries if item.get("name") == knowledge_base_name), None)
        now = utc_now()
        version_count = len(library.get("versions", [])) if library else 0
        version_id = f"v{version_count + 1}"
        version = {
            "id": version_id,
            "label": f"V{version_count + 1}.0 当前版本",
            "status": "ready",
            "updated_at": now,
            "summary": payload.get("description", "").strip() or f"由分析任务 {payload['task_id']} 构建生成。",
            "metrics": {
                "entities": len(node_map),
                "relations": len(edges),
                "sources": max(1, len(source_labels)),
            },
            "nodes": list(node_map.values()),
            "edges": edges,
            "task_id": payload["task_id"],
        }

        layer_strategy = payload.get("layer_strategy", "")
        layers = [item.strip() for item in layer_strategy.split("+") if item.strip()]
        source_display = " / ".join(source_types)

        if library:
            updated = {
                **library,
                "domain": payload.get("business_domain", "").strip() or library.get("domain", ""),
                "source": source_display or library.get("source", ""),
                "layers": layers or library.get("layers", []),
                "entity_count": len(node_map),
                "relation_count": len(edges),
                "updated_at": now,
                "owner": "知识图谱构建任务",
                "versions": [version, *library.get("versions", [])],
            }
            self.registry.update(self.LIBRARIES, library["id"], updated)
            library_result = updated
        else:
            library_result = {
                "id": f"kg-{uuid.uuid4().hex[:8]}",
                "name": knowledge_base_name,
                "domain": payload.get("business_domain", "").strip(),
                "source": source_display,
                "layers": layers,
                "entity_count": len(node_map),
                "relation_count": len(edges),
                "updated_at": now,
                "owner": "知识图谱构建任务",
                "children": [],
                "versions": [version],
            }
            self.registry.add(self.LIBRARIES, library_result)

        existing_triples = self.registry.list(self.TRIPLES)
        persisted_triples = [
            {
                "id": item["id"],
                "source": item["source"],
                "source_type": item["source_type"],
                "relation": item["relation"],
                "target": item["target"],
                "target_type": item["target_type"],
                "origin": payload["task_id"],
                "status": "已确认",
            }
            for item in triples
        ]
        self.registry.replace_list(self.TRIPLES, [*persisted_triples, *existing_triples])

        self.registry.update(
            self.ANALYSIS_TASKS,
            payload["task_id"],
            {
                "status": "已确认",
                "library_id": library_result["id"],
                "version_id": version_id,
            },
        )
        return library_result

    def build_fusion_library_version(self, payload: dict) -> dict:
        source_graphs = self._resolve_fusion_source_graphs(payload)
        if len(source_graphs) < 2:
            raise ValueError("至少需要两个可用的子知识库图谱才能执行融合。")

        fusion_mode = payload.get("fusion_mode", "").strip() or "实体对齐优先"
        conflict_rule = payload.get("conflict_rule", "").strip()
        knowledge_base_name = payload["knowledge_base_name"].strip()
        now = utc_now()

        node_map: dict[str, dict] = {}
        edge_signatures: set[tuple[str, str, str]] = set()
        edges: list[dict] = []
        source_labels: set[str] = set()
        source_names: list[str] = []
        conflict_items: list[dict] = []
        fusion_trace: list[dict] = []

        for graph in source_graphs:
            version = graph.get("version", {})
            version_nodes = version.get("nodes", [])
            version_edges = version.get("edges", [])
            source_names.append(graph.get("name", ""))
            source_labels.add(graph.get("name", ""))
            local_node_map: dict[str, dict] = {}

            for node in version_nodes:
                normalized = self._normalize_graph_label(node.get("label", ""))
                node_type = str(node.get("type", "")).strip() or "实体"
                dedupe_key = normalized if fusion_mode == "关系完整优先" else f"{normalized}::{node_type}"
                existing = node_map.get(dedupe_key)
                if existing:
                    existing_sources = set(existing.get("sources", []))
                    existing_sources.add(graph.get("name", ""))
                    existing["sources"] = sorted(filter(None, existing_sources))
                    if existing.get("type") != node_type:
                        conflict_items.append(
                            {
                                "id": f"fusion-conflict-{uuid.uuid4().hex[:8]}",
                                "field": "节点类型",
                                "entity": node.get("label", ""),
                                "left": existing.get("type", ""),
                                "right": node_type,
                                "severity": "中",
                                "suggestion": conflict_rule or "保留原类型并记录来源",
                            }
                        )
                    local_node_map[node.get("id")] = existing
                    continue

                created = {
                    "id": f"node-{uuid.uuid4().hex[:8]}",
                    "label": node.get("label", "").strip(),
                    "type": node_type,
                    "sources": [graph.get("name", "")],
                    "properties": {
                        "source_library": graph.get("parent_library_name", ""),
                        "source_subgraph": graph.get("name", ""),
                        "fusion_mode": fusion_mode,
                    },
                }
                node_map[dedupe_key] = created
                local_node_map[node.get("id")] = created

            for edge in version_edges:
                source_node = local_node_map.get(edge.get("source"))
                target_node = local_node_map.get(edge.get("target"))
                if not source_node or not target_node:
                    continue
                relation = str(edge.get("relation", "")).strip() or "关联"
                signature = (source_node["id"], relation, target_node["id"])
                if signature in edge_signatures:
                    continue
                edge_signatures.add(signature)
                edges.append(
                    {
                        "id": f"edge-{uuid.uuid4().hex[:8]}",
                        "source": source_node["id"],
                        "target": target_node["id"],
                        "relation": relation,
                    }
                )
                fusion_trace.append(
                    {
                        "source_graph_id": graph.get("id", ""),
                        "source_graph_name": graph.get("name", ""),
                        "relation": relation,
                        "source_label": source_node["label"],
                        "target_label": target_node["label"],
                    }
                )

        libraries = self.registry.list(self.LIBRARIES)
        library = next((item for item in libraries if item.get("name") == knowledge_base_name), None)
        version_count = len(library.get("versions", [])) if library else 0
        version_id = f"v{version_count + 1}"
        source_domain = " / ".join(sorted({graph.get("domain", "") for graph in source_graphs if graph.get("domain")}))[:120]
        version = {
            "id": version_id,
            "label": f"V{version_count + 1}.0 当前版本",
            "status": "ready",
            "updated_at": now,
            "summary": payload.get("description", "").strip()
            or f"融合 {len(source_graphs)} 个子知识库，采用“{fusion_mode}”策略生成。",
            "metrics": {
                "entities": len(node_map),
                "relations": len(edges),
                "sources": max(1, len(source_labels)),
            },
            "nodes": list(node_map.values()),
            "edges": edges,
            "fusion": {
                "mode": fusion_mode,
                "conflict_rule": conflict_rule,
                "source_graph_ids": [graph.get("id") for graph in source_graphs],
                "source_graphs": [
                    {
                        "id": graph.get("id", ""),
                        "name": graph.get("name", ""),
                        "parent_library_name": graph.get("parent_library_name", ""),
                        "version_label": graph.get("version", {}).get("label", ""),
                    }
                    for graph in source_graphs
                ],
                "conflict_items": conflict_items,
                "trace": fusion_trace[:200],
            },
        }

        if library:
            updated = {
                **library,
                "domain": source_domain or library.get("domain", ""),
                "source": "已有知识图谱融合",
                "layers": library.get("layers", []) or ["实体层", "关系层", "溯源层"],
                "entity_count": len(node_map),
                "relation_count": len(edges),
                "updated_at": now,
                "owner": "图谱融合构建任务",
                "versions": [version, *library.get("versions", [])],
            }
            self.registry.update(self.LIBRARIES, library["id"], updated)
            return updated

        item = {
            "id": f"kg-{uuid.uuid4().hex[:8]}",
            "name": knowledge_base_name,
            "domain": source_domain,
            "source": "已有知识图谱融合",
            "layers": ["实体层", "关系层", "溯源层"],
            "entity_count": len(node_map),
            "relation_count": len(edges),
            "updated_at": now,
            "owner": "图谱融合构建任务",
            "children": [],
            "versions": [version],
        }
        self.registry.add(self.LIBRARIES, item)
        return item

    def _resolve_fusion_source_graphs(self, payload: dict) -> list[dict]:
        selected_ids = payload.get("selected_graph_ids", [])
        source_graph_payload = {item.get("id"): item for item in payload.get("source_graphs", [])}
        libraries = self.registry.list(self.LIBRARIES)
        resolved: list[dict] = []

        for selected_id in selected_ids:
            graph = source_graph_payload.get(selected_id)
            if graph:
                resolved.append(graph)
                continue
            graph = self._find_graph_source_from_registry(libraries, selected_id)
            if graph:
                resolved.append(graph)

        unique: list[dict] = []
        seen_ids: set[str] = set()
        for item in resolved:
            item_id = item.get("id", "")
            if item_id and item_id in seen_ids:
                continue
            if item_id:
                seen_ids.add(item_id)
            unique.append(item)
        return unique

    def _find_graph_source_from_registry(self, libraries: list[dict], selected_id: str) -> dict | None:
        for library in libraries:
            if library.get("id") == selected_id and library.get("versions"):
                return {
                    "id": library.get("id", ""),
                    "parent_library_id": library.get("id", ""),
                    "parent_library_name": library.get("name", ""),
                    "name": library.get("name", ""),
                    "domain": library.get("domain", ""),
                    "scene": library.get("domain", ""),
                    "entity_count": library.get("entity_count", 0),
                    "relation_count": library.get("relation_count", 0),
                    "updated_at": library.get("updated_at", ""),
                    "version": library.get("versions", [])[0],
                }
            for child in library.get("children", []) or []:
                if child.get("id") == selected_id and library.get("versions"):
                    return {
                        "id": child.get("id", ""),
                        "parent_library_id": library.get("id", ""),
                        "parent_library_name": library.get("name", ""),
                        "name": child.get("name", ""),
                        "domain": library.get("domain", ""),
                        "scene": child.get("scene", ""),
                        "entity_count": child.get("entity_count", 0),
                        "relation_count": child.get("relation_count", 0),
                        "updated_at": child.get("updated_at", ""),
                        "version": library.get("versions", [])[0],
                    }
        return None

    @staticmethod
    def _normalize_graph_label(label: str) -> str:
        cleaned = re.sub(r"\s+", "", str(label or "").strip().lower())
        return cleaned
