from __future__ import annotations

import uuid
from typing import Any

from ..storage import Registry, utc_now
from .agent_service import AgentService


class AgentApiService:
    def __init__(self, registry: Registry, agent_service: AgentService) -> None:
        self.registry = registry
        self.agent_service = agent_service

    def list_services(self) -> list[dict]:
        services = self.registry.list("agent_services")
        return sorted(services, key=lambda item: item.get("updated_at", item.get("created_at", "")), reverse=True)

    def start_service(self, agent_id: str, base_url: str) -> dict:
        agent = self.agent_service.get_agent(agent_id)
        if not agent:
            raise ValueError("智能体不存在。")

        active_service = self._active_service_for_agent(agent_id)
        if active_service:
            url = self._service_url(base_url, active_service["id"])
            return self.registry.update(
                "agent_services",
                active_service["id"],
                {
                    "agent_name": agent.get("name", ""),
                    "agent_description": agent.get("description", ""),
                    "node_count": agent.get("node_count") or len(agent.get("nodes", [])),
                    "url": url,
                    "status": "running",
                },
            ) or active_service

        service_id = uuid.uuid4().hex
        service = {
            "id": service_id,
            "agent_id": agent_id,
            "agent_name": agent.get("name", ""),
            "agent_description": agent.get("description", ""),
            "node_count": agent.get("node_count") or len(agent.get("nodes", [])),
            "url": self._service_url(base_url, service_id),
            "status": "running",
            "invoke_count": 0,
            "last_invoked_at": "",
            "started_at": utc_now(),
            "stopped_at": "",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        return self.registry.add("agent_services", service)

    def stop_service(self, service_id: str) -> dict | None:
        service = self.registry.get("agent_services", service_id)
        if not service:
            return None
        was_running = service.get("status") == "running"
        updated = self.registry.update(
            "agent_services",
            service_id,
            {
                "status": "stopped",
                "stopped_at": utc_now(),
            },
        )
        if was_running:
            released = self._release_unused_model_sessions(service)
            if updated:
                updated["unloaded_model_keys"] = released["unloaded"]
                updated["retained_model_keys"] = released["retained"]
        return updated

    def stop_services_for_agent(self, agent_id: str) -> None:
        for service in self.registry.list("agent_services"):
            if service.get("agent_id") == agent_id and service.get("status") == "running":
                self.stop_service(service["id"])

    def invoke(self, service_id: str, input_text: str, include_trace: bool = True) -> dict[str, Any]:
        service = self.registry.get("agent_services", service_id)
        if not service or service.get("status") != "running":
            raise ValueError("智能体服务不存在或未运行。")

        agent = self.agent_service.get_agent(service.get("agent_id", ""))
        if not agent:
            self.stop_service(service_id)
            raise ValueError("智能体不存在，服务已停止。")

        result = self.agent_service.run_agent(agent["id"], input_text)
        self.registry.update(
            "agent_services",
            service_id,
            {
                "agent_name": agent.get("name", service.get("agent_name", "")),
                "node_count": agent.get("node_count") or len(agent.get("nodes", [])),
                "invoke_count": int(service.get("invoke_count") or 0) + 1,
                "last_invoked_at": utc_now(),
            },
        )
        if not include_trace:
            result = {key: value for key, value in result.items() if key != "trace"}
        return {
            "service_id": service_id,
            "agent_id": agent["id"],
            "agent_name": agent.get("name"),
            **result,
        }

    def _release_unused_model_sessions(self, stopped_service: dict) -> dict[str, list[str]]:
        stopped_keys = self._agent_runtime_keys(stopped_service.get("agent_id", ""))
        if not stopped_keys:
            return {"unloaded": [], "retained": []}

        retained_keys: set[str] = set()
        for service in self.registry.list("agent_services"):
            if service.get("id") == stopped_service.get("id") or service.get("status") != "running":
                continue
            retained_keys.update(self._agent_runtime_keys(service.get("agent_id", "")))

        unload_keys = stopped_keys - retained_keys
        unloaded = self.agent_service.model_runtime.unload_many(unload_keys)
        return {
            "unloaded": unloaded,
            "retained": sorted(stopped_keys & retained_keys),
        }

    def _agent_runtime_keys(self, agent_id: str) -> set[str]:
        agent = self.agent_service.get_agent(agent_id)
        if not agent:
            return set()

        runtime = self.agent_service.model_runtime
        keys: set[str] = set()
        for node in agent.get("nodes", []):
            node_type = node.get("type")
            config = node.get("config") or {}
            if node_type == "llm":
                base_model = self.registry.get("base_models", config.get("model_id", ""))
                if base_model:
                    keys.add(runtime.runtime_key_for_base(base_model))
                continue

            if node_type != "finetuned":
                continue

            fine_tuned_model_id = config.get("fine_tuned_model_id")
            if not fine_tuned_model_id:
                base_model = self.registry.get("base_models", config.get("model_id", ""))
                if base_model:
                    keys.add(runtime.runtime_key_for_base(base_model))
                continue

            fine_tuned_model = self.registry.get("fine_tuned_models", fine_tuned_model_id)
            if not fine_tuned_model:
                continue
            base_model = self.registry.get("base_models", fine_tuned_model.get("base_model_id", ""))
            if not base_model:
                continue
            if fine_tuned_model.get("model_type") == "rag":
                keys.add(runtime.runtime_key_for_base(base_model))
            else:
                keys.add(runtime.runtime_key_for_fine_tuned(base_model, fine_tuned_model))
        return keys

    def _active_service_for_agent(self, agent_id: str) -> dict | None:
        for service in self.registry.list("agent_services"):
            if service.get("agent_id") == agent_id and service.get("status") == "running":
                return service
        return None

    def _service_url(self, base_url: str, service_id: str) -> str:
        return f"{base_url.rstrip('/')}/api/agent-services/{service_id}/invoke"
