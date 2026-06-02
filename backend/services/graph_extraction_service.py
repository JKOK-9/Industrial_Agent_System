from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass

import httpx
import pandas as pd

from ..config import GRAPH_LLM_API_KEY, GRAPH_LLM_BASE_URL, GRAPH_LLM_MODEL


@dataclass
class HeaderSuggestion:
    header: str
    sample: str
    role: str
    description: str


class GraphExtractionService:
    def analyze_table(self, file_storage, node_types: list[dict], relation_types: list[dict]) -> dict:
        dataframe = self._read_table(file_storage)
        headers = [str(column).strip() for column in dataframe.columns.tolist()]
        preview_rows = dataframe.head(5).fillna("").astype(str).to_dict(orient="records")

        if GRAPH_LLM_API_KEY and GRAPH_LLM_BASE_URL and GRAPH_LLM_MODEL:
            try:
                suggestions = self._request_llm(headers, preview_rows, node_types, relation_types)
                return {
                    "headers": suggestions["headers"],
                    "sample_rows": preview_rows,
                    "node_suggestions": suggestions["node_suggestions"],
                    "relation_suggestions": suggestions["relation_suggestions"],
                    "confirmation_rules": suggestions["confirmation_rules"],
                    "analysis_source": "llm",
                }
            except Exception:
                pass

        return self._build_heuristic_payload(headers, preview_rows, node_types, relation_types)

    def _read_table(self, file_storage) -> pd.DataFrame:
        filename = (file_storage.filename or "").lower()
        content = file_storage.read()
        file_storage.stream.seek(0)

        if filename.endswith(".csv"):
            return pd.read_csv(io.BytesIO(content))
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            return pd.read_excel(io.BytesIO(content))
        raise ValueError("仅支持 CSV / Excel 文件解析。")

    def _request_llm(self, headers: list[str], preview_rows: list[dict], node_types: list[dict], relation_types: list[dict]) -> dict:
        node_type_names = [item.get("name", "") for item in node_types]
        relation_type_names = [item.get("name", "") for item in relation_types]
        prompt = {
            "headers": headers,
            "preview_rows": preview_rows,
            "available_node_types": node_type_names,
            "available_relation_types": relation_type_names,
            "task": (
                "识别原始业务表中的表头角色，推断哪些表头应映射为节点，哪些表头之间存在关系，"
                "并返回可供用户确认的关系规则。请仅返回 JSON。"
            ),
            "output_schema": {
                "headers": [{"name": "表头", "sample": "样例值", "role": "字段角色", "description": "角色说明"}],
                "node_suggestions": [{"header": "表头", "node_type": "节点类型", "confidence": "高/中/低", "reason": "判断原因"}],
                "relation_suggestions": [
                    {
                        "source_header": "起点表头",
                        "relation": "关系名称",
                        "target_header": "终点表头",
                        "confidence": "高/中/低",
                        "reason": "判断原因",
                    }
                ],
                "confirmation_rules": [
                    {
                        "source_header": "起点表头",
                        "source_type": "起点节点类型",
                        "relation": "关系名称",
                        "target_header": "终点表头",
                        "target_type": "终点节点类型",
                        "status": "待确认",
                    }
                ],
            },
        }

        response = httpx.post(
            f"{GRAPH_LLM_BASE_URL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {GRAPH_LLM_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": GRAPH_LLM_MODEL,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": "你是工业知识图谱抽取助手，只输出 JSON。"},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        content = self._extract_json(content)
        data = json.loads(content)
        return {
            "headers": [
                {
                    "id": f"header-{index + 1}",
                    "name": item["name"],
                    "sample": item.get("sample", ""),
                    "role": item.get("role", ""),
                    "description": item.get("description", ""),
                }
                for index, item in enumerate(data.get("headers", []))
            ],
            "node_suggestions": [
                {"id": f"node-s-{index + 1}", **item}
                for index, item in enumerate(data.get("node_suggestions", []))
            ],
            "relation_suggestions": [
                {"id": f"rel-s-{index + 1}", **item}
                for index, item in enumerate(data.get("relation_suggestions", []))
            ],
            "confirmation_rules": [
                {"id": f"confirm-{index + 1}", **item}
                for index, item in enumerate(data.get("confirmation_rules", []))
            ],
        }

    def _extract_json(self, content: str) -> str:
        fenced = re.search(r"```json\s*(.*?)\s*```", content, re.S)
        if fenced:
            return fenced.group(1)
        return content

    def _build_heuristic_payload(self, headers: list[str], preview_rows: list[dict], node_types: list[dict], relation_types: list[dict]) -> dict:
        header_items: list[dict] = []
        node_suggestions: list[dict] = []
        relation_suggestions: list[dict] = []
        confirmation_rules: list[dict] = []

        node_type_names = {item.get("name", "") for item in node_types}
        relation_type_names = {item.get("name", "") for item in relation_types}

        for index, header in enumerate(headers):
            sample = preview_rows[0].get(header, "") if preview_rows else ""
            role, description, node_type = self._infer_header_role(header, node_type_names)
            header_items.append(
                {
                    "id": f"header-{index + 1}",
                    "name": header,
                    "sample": sample,
                    "role": role,
                    "description": description,
                }
            )
            if node_type:
                node_suggestions.append(
                    {
                        "id": f"node-s-{len(node_suggestions) + 1}",
                        "header": header,
                        "node_type": node_type,
                        "confidence": "高" if role != "属性列" else "中",
                        "reason": description,
                    }
                )

        header_lookup = {item["header"]: item["node_type"] for item in node_suggestions}
        relation_candidates = [
            ("设备名称", "出现", "故障现象"),
            ("故障原因", "导致", "故障现象"),
            ("处理措施", "处理", "故障现象"),
            ("责任班组", "负责", "设备名称"),
            ("设备名称", "位于", "安装位置"),
            ("设备名称", "使用", "备件编码"),
        ]
        for source_header, relation, target_header in relation_candidates:
            if source_header in headers and target_header in headers:
                relation_name = relation if relation in relation_type_names else next(iter(relation_type_names), relation)
                reason = f"{source_header} 与 {target_header} 在工业场景下通常存在 {relation} 关系"
                relation_suggestions.append(
                    {
                        "id": f"rel-s-{len(relation_suggestions) + 1}",
                        "source_header": source_header,
                        "relation": relation_name,
                        "target_header": target_header,
                        "confidence": "高",
                        "reason": reason,
                    }
                )
                confirmation_rules.append(
                    {
                        "id": f"confirm-{len(confirmation_rules) + 1}",
                        "source_header": source_header,
                        "source_type": header_lookup.get(source_header, next(iter(node_type_names), "设备")),
                        "relation": relation_name,
                        "target_header": target_header,
                        "target_type": header_lookup.get(target_header, next(iter(node_type_names), "故障")),
                        "status": "待确认",
                    }
                )

        return {
            "headers": header_items,
            "sample_rows": preview_rows,
            "node_suggestions": node_suggestions,
            "relation_suggestions": relation_suggestions,
            "confirmation_rules": confirmation_rules,
            "analysis_source": "heuristic",
        }

    def _infer_header_role(self, header: str, node_type_names: set[str]) -> tuple[str, str, str | None]:
        header_text = header.lower()
        if "设备" in header or "机组" in header:
            return "主实体列", "字段值稳定指向设备实体", "设备" if "设备" in node_type_names else next(iter(node_type_names), None)
        if "故障" in header and "原因" not in header:
            return "事件列", "字段值多为故障表现或异常现象", "故障" if "故障" in node_type_names else next(iter(node_type_names), None)
        if "原因" in header:
            return "原因列", "字段值多为引发故障的原因对象", "故障" if "故障" in node_type_names else next(iter(node_type_names), None)
        if "措施" in header or "处理" in header or "步骤" in header:
            return "动作列", "字段值描述处理动作或业务步骤", "工艺步骤" if "工艺步骤" in node_type_names else next(iter(node_type_names), None)
        if "班组" in header or "部门" in header or "责任" in header:
            return "组织列", "字段值适合作为责任组织节点", "组织" if "组织" in node_type_names else next(iter(node_type_names), None)
        if "备件" in header or "物料" in header:
            return "物料列", "字段值可作为备件或物料节点", "备件" if "备件" in node_type_names else next(iter(node_type_names), None)
        if "位置" in header or "车间" in header:
            return "位置列", "字段值体现安装位置或空间对象", "位置" if "位置" in node_type_names else next(iter(node_type_names), None)
        if "编码" in header or "编号" in header:
            return "标识列", "字段值更适合作为实体标识或属性", None
        return "属性列", "字段值更适合作为属性补充或辅助信息", None
