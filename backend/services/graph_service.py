from __future__ import annotations

import uuid

from ..storage import Registry, utc_now


class GraphService:
    NODE_TYPES = "graph_node_types"
    RELATION_TYPES = "graph_relation_types"
    RELATION_RULES = "graph_relation_rules"
    TRIPLES = "graph_triples"
    ANALYSIS_TASKS = "graph_analysis_tasks"

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
