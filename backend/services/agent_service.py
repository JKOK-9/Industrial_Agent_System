from __future__ import annotations

import csv
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from ..config import AGENTS_DIR, is_path_inside
from ..schemas import AgentWorkflowRequest
from ..storage import Registry, utc_now
from .model_runtime import GenerationOptions, ModelRuntime
from .rag_service import RagService


THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


class AgentService:
    def __init__(self, registry: Registry, model_runtime: ModelRuntime, rag_service: RagService) -> None:
        self.registry = registry
        self.model_runtime = model_runtime
        self.rag_service = rag_service

    def list_agents(self) -> list[dict]:
        return sorted(self.registry.list("agent_workflows"), key=lambda item: item.get("created_at", ""), reverse=True)

    def get_agent(self, agent_id: str) -> dict | None:
        return self.registry.get("agent_workflows", agent_id)

    def save_agent(self, request: AgentWorkflowRequest) -> dict:
        self._validate_nodes(request.nodes)
        agent_id = uuid.uuid4().hex
        agent_dir = (AGENTS_DIR / agent_id).resolve()
        agent_dir.mkdir(parents=True, exist_ok=True)

        agent = {
            "id": agent_id,
            "name": request.name,
            "description": request.description,
            "nodes": [node.model_dump() for node in request.nodes],
            "status": "ready",
            "node_count": len(request.nodes),
            "path": str(agent_dir / "agent.json"),
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        self._write_agent_file(agent)
        self.registry.add("agent_workflows", agent)
        return agent

    def delete_agent(self, agent_id: str) -> bool:
        agent = self.registry.delete("agent_workflows", agent_id)
        if not agent:
            return False
        path = Path(agent.get("path", "")).resolve()
        agent_dir = path.parent if path.is_file() else path
        if is_path_inside(agent_dir, AGENTS_DIR) and agent_dir.exists():
            shutil.rmtree(agent_dir)
        return True

    def run_agent(self, agent_id: str, input_text: str) -> dict:
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError("智能体不存在。")
        return self._run_nodes(agent, input_text)

    def run_preview(self, request: AgentWorkflowRequest, input_text: str) -> dict:
        self._validate_nodes(request.nodes)
        agent = {
            "id": "preview",
            "name": request.name or "未命名智能体",
            "description": request.description,
            "nodes": [node.model_dump() for node in request.nodes],
        }
        return self._run_nodes(agent, input_text)

    def _validate_nodes(self, nodes: list) -> None:
        if len(nodes) < 2:
            raise ValueError("智能体工作流至少需要两个节点。")
        if nodes[0].type != "input":
            raise ValueError("顺序链路的第一个节点必须是用户输入。")
        if nodes[-1].type != "output":
            raise ValueError("顺序链路的最后一个节点必须是结果输出。")

    def _run_nodes(self, agent: dict, input_text: str) -> dict:
        original_input = input_text.strip()
        current = original_input
        contexts: list[dict[str, str]] = []
        trace: list[dict[str, Any]] = []

        for index, node in enumerate(agent.get("nodes", []), start=1):
            node_type = node.get("type")
            config = node.get("config") or {}
            label = node.get("label") or _node_type_text(node_type)

            if node_type == "input":
                trace.append({"index": index, "node": label, "type": node_type, "output": current})
                continue

            if node_type == "knowledge":
                source = self.registry.get("knowledge_sources", config.get("knowledge_source_id", ""))
                if not source:
                    raise ValueError(f"知识库节点「{label}」未选择有效知识源。")
                content = self._read_knowledge_source(source)
                contexts.append({"name": source.get("name", "知识源"), "content": content})
                trace.append(
                    {
                        "index": index,
                        "node": label,
                        "type": node_type,
                        "output": f"已加载知识源：{source.get('name', '-')}",
                    }
                )
                continue

            if node_type in {"llm", "finetuned"}:
                model, base_model = self._resolve_model(node_type, config)
                prompt, rag_contexts = self._build_prompt_for_node(node_type, model, config, contexts, current, original_input)
                output = self._generate_model_output(node_type, model, base_model, prompt, config)
                if config.get("filter_thinking", True):
                    output = THINK_PATTERN.sub("", output).strip()
                current = output
                contexts = []
                trace.append(
                    {
                        "index": index,
                        "node": label,
                        "type": node_type,
                        "model": model.get("display_name") or model.get("base_model_name"),
                        "prompt": prompt,
                        "contexts": rag_contexts,
                        "output": current,
                    }
                )
                continue

            if node_type == "output":
                trace.append({"index": index, "node": label, "type": node_type, "output": current})
                continue

            raise ValueError(f"暂不支持的节点类型：{node_type}")

        return {
            "agent_id": agent.get("id"),
            "agent_name": agent.get("name"),
            "input": input_text,
            "output": current,
            "trace": trace,
        }

    def _resolve_model(self, node_type: str, config: dict) -> tuple[dict, dict | None]:
        if node_type == "finetuned" and config.get("fine_tuned_model_id"):
            model = self.registry.get("fine_tuned_models", config["fine_tuned_model_id"])
            if not model:
                raise ValueError("微调模型节点选择的模型不存在。")
            base_model = self.registry.get("base_models", model.get("base_model_id", ""))
            if not base_model:
                raise ValueError("微调模型关联的基座模型不存在。")
            if base_model.get("status") != "ready":
                raise ValueError("微调模型关联的基座模型尚未 ready。")
            return model, base_model

        model = self.registry.get("base_models", config.get("model_id", ""))
        if not model:
            raise ValueError("大模型节点未选择有效基座模型。")
        if model.get("status") != "ready":
            raise ValueError("大模型节点选择的基座模型尚未 ready。")
        return model, model

    def _build_prompt_for_node(
        self,
        node_type: str,
        model: dict,
        config: dict,
        contexts: list[dict[str, str]],
        current: str,
        original_input: str,
    ) -> tuple[str, list[dict]]:
        if node_type == "finetuned" and model.get("model_type") == "rag":
            rag_payload = self.rag_service.build_prompt(model["id"], current, int(config.get("top_k") or 4))
            extra_prompt = config.get("prompt", "").strip()
            context_prompt = (
                self._build_model_prompt(extra_prompt, contexts, current, original_input)
                if contexts or extra_prompt or original_input != current
                else ""
            )
            prompt = rag_payload["prompt"]
            if context_prompt:
                prompt = f"{context_prompt}\n\nRAG封装提示：\n{prompt}"
            return prompt, rag_payload.get("contexts", [])

        return self._build_model_prompt(config.get("prompt", ""), contexts, current, original_input), []

    def _build_model_prompt(self, prompt: str, contexts: list[dict[str, str]], current: str, original_input: str) -> str:
        context_text = "\n\n".join(f"[{item['name']}]\n{item['content']}" for item in contexts)
        parts = []
        if prompt:
            parts.append(f"节点提示词：\n{prompt}")
        if original_input and original_input != current:
            parts.append(f"原始用户输入：\n{original_input}")
        if context_text:
            parts.append(f"知识库上下文：\n{context_text}")
        parts.append(f"当前输入：\n{current}")
        return "\n\n".join(parts)

    def _generate_model_output(self, node_type: str, model: dict, base_model: dict | None, prompt: str, config: dict) -> str:
        options = _generation_options_from_config(config)
        if node_type == "llm":
            return self.model_runtime.generate_base(model, prompt, options)

        if model.get("model_type") == "rag":
            if not base_model:
                raise ValueError("RAG 模型关联的基座模型不存在。")
            return self.model_runtime.generate_base(base_model, prompt, options)

        if not base_model:
            raise ValueError("微调模型关联的基座模型不存在。")
        return self.model_runtime.generate_fine_tuned(base_model, model, prompt, options)

    def _read_knowledge_source(self, source: dict) -> str:
        path = Path(source.get("path", ""))
        if not path.exists():
            return source.get("preview", "")
        source_type = source.get("source_type")
        suffix = path.suffix.lower()
        if source_type in {"text", "markdown"}:
            return _limit_text(path.read_text(encoding="utf-8", errors="replace"))
        if source_type == "json":
            if suffix == ".jsonl":
                rows = []
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        stripped = line.strip()
                        if stripped:
                            rows.append(_json_to_text(json.loads(stripped)))
                return _limit_text("\n".join(rows))
            return _limit_text(_json_to_text(json.loads(path.read_text(encoding="utf-8", errors="replace"))))

        delimiter = "\t" if suffix == ".tsv" else ","
        rows = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.reader(handle, delimiter=delimiter):
                rows.append(" | ".join(cell.strip() for cell in row))
        return _limit_text("\n".join(rows))

    def _write_agent_file(self, agent: dict) -> None:
        Path(agent["path"]).write_text(json.dumps(agent, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(_json_to_text(item) for item in value)
    if isinstance(value, dict):
        return "\n".join(f"{key}: {_json_to_text(item)}" for key, item in value.items())
    return str(value)


def _limit_text(text: str, limit: int = 4000) -> str:
    compact = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return compact[:limit]


def _generation_options_from_config(config: dict) -> GenerationOptions:
    return GenerationOptions(
        max_new_tokens=_bounded_int(config.get("max_new_tokens"), default=512, minimum=1, maximum=8192),
        temperature=_bounded_float(config.get("temperature"), default=0.7, minimum=0.0, maximum=2.0),
        top_p=_bounded_float(config.get("top_p"), default=0.9, minimum=0.01, maximum=1.0),
    )


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _bounded_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _node_type_text(node_type: str) -> str:
    return {
        "input": "用户输入",
        "output": "结果输出",
        "llm": "大模型",
        "finetuned": "微调模型",
        "knowledge": "知识库",
    }.get(node_type, node_type)
