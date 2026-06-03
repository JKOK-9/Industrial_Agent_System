from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS
from pydantic import ValidationError

from ..config import CORS_ORIGINS, CORS_SUPPORTS_CREDENTIALS, ensure_runtime_dirs
from ..schemas import AgentWorkflowRequest
from ..storage import Registry, utc_now
from .agent_service import AgentService
from .model_runtime import ModelRuntime
from .rag_service import RagService


def _build_agent_service() -> AgentService:
    ensure_runtime_dirs()
    registry = Registry()
    model_runtime = ModelRuntime()
    rag_service = RagService(registry)
    return AgentService(registry, model_runtime, rag_service)


def _run_payload(payload: dict[str, Any]) -> dict[str, Any]:
    agent_service = _build_agent_service()
    mode = payload.get("mode")
    input_text = str(payload.get("input_text") or "")
    include_trace = bool(payload.get("include_trace", True))

    if mode == "preview":
        agent_request = AgentWorkflowRequest.model_validate(payload.get("agent") or {})
        result = agent_service.run_preview(agent_request, input_text)
    elif mode == "saved":
        agent_id = str(payload.get("agent_id") or "")
        result = agent_service.run_agent(agent_id, input_text)
    else:
        raise ValueError(f"不支持的智能体运行模式：{mode}")

    if not include_trace:
        result = {key: value for key, value in result.items() if key != "trace"}
    return result


def _run_once(payload_path: Path, output_path: Path) -> int:
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        result = _run_payload(payload)
        output = {"ok": True, "result": result}
    except Exception as exc:  # noqa: BLE001
        output = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if output.get("ok") else 1


def _create_service_app(service_id: str, agent_id: str) -> Flask:
    registry = Registry()
    model_runtime = ModelRuntime()
    rag_service = RagService(registry)
    agent_service = AgentService(registry, model_runtime, rag_service)

    app = Flask(__name__)
    CORS(
        app,
        resources={
            r"/*": {
                "origins": CORS_ORIGINS,
                "supports_credentials": CORS_SUPPORTS_CREDENTIALS,
            }
        },
    )

    @app.get("/health")
    def health():
        service = registry.get("agent_services", service_id)
        status = service.get("status") if service else "missing"
        return jsonify({"status": status, "service_id": service_id, "agent_id": agent_id})

    def invoke_service():
        service = registry.get("agent_services", service_id)
        if not service or service.get("status") != "running":
            return jsonify({"detail": "智能体服务不存在或未运行。"}), 404

        payload = request.get_json(silent=True) or {}
        input_text = (
            payload.get("input_text")
            or payload.get("question")
            or payload.get("input")
            or payload.get("message")
            or ""
        )
        include_trace = payload.get("include_trace", True)
        if isinstance(include_trace, str):
            include_trace = include_trace.lower() not in {"false", "0", "no"}

        try:
            result = agent_service.run_agent(agent_id, str(input_text))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"detail": str(exc)}), 500

        updated = registry.update(
            "agent_services",
            service_id,
            {
                "invoke_count": int(service.get("invoke_count") or 0) + 1,
                "last_invoked_at": utc_now(),
            },
        )
        if not include_trace:
            result = {key: value for key, value in result.items() if key != "trace"}
        return jsonify(
            {
                "service_id": service_id,
                "agent_id": agent_id,
                "agent_name": (updated or service).get("agent_name"),
                **result,
            }
        )

    app.add_url_rule("/invoke", "invoke", invoke_service, methods=["POST"])
    app.add_url_rule(f"/api/agent-services/{service_id}/invoke", "invoke_compat", invoke_service, methods=["POST"])
    return app


def _serve(payload_path: Path) -> int:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    service_id = str(payload["service_id"])
    agent_id = str(payload["agent_id"])
    host = str(payload.get("host") or "127.0.0.1")
    port = int(payload["port"])
    app = _create_service_app(service_id, agent_id)
    app.run(host=host, port=port, threaded=True, use_reloader=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print("Usage: python -m backend.services.agent_worker <run-once|serve> ...", file=sys.stderr)
        return 2

    command = argv[0]
    try:
        if command == "run-once" and len(argv) == 3:
            return _run_once(Path(argv[1]).resolve(), Path(argv[2]).resolve())
        if command == "serve" and len(argv) == 2:
            return _serve(Path(argv[1]).resolve())
    except ValidationError as exc:
        print(exc, file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(exc, file=sys.stderr)
        return 1

    print("Invalid agent_worker arguments.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
