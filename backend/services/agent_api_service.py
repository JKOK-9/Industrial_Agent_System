from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import parse, request

from ..config import (
    AGENT_RUN_TIMEOUT_SECONDS,
    AGENT_SERVICE_HOST,
    AGENT_SERVICE_PORT_END,
    AGENT_SERVICE_PORT_START,
    AGENT_SERVICE_PUBLIC_HOST,
    AGENT_SERVICE_PUBLIC_SCHEME,
    AGENT_SERVICE_START_TIMEOUT_SECONDS,
    CONFIGS_DIR,
    LOGS_DIR,
    PROJECT_ROOT,
)
from ..storage import Registry, utc_now
from .agent_service import AgentService


class AgentApiService:
    def __init__(self, registry: Registry, agent_service: AgentService) -> None:
        self.registry = registry
        self.agent_service = agent_service
        self._service_processes: dict[str, subprocess.Popen[str]] = {}

    def list_services(self) -> list[dict]:
        self._sync_service_processes()
        services = self.registry.list("agent_services")
        return sorted(services, key=lambda item: item.get("updated_at", item.get("created_at", "")), reverse=True)

    def start_service(self, agent_id: str, base_url: str) -> dict:
        agent = self.agent_service.get_agent(agent_id)
        if not agent:
            raise ValueError("智能体不存在。")

        active_service = self._active_service_for_agent(agent_id)
        if active_service:
            if self._is_service_healthy(active_service):
                return self.registry.update(
                    "agent_services",
                    active_service["id"],
                    {
                        "agent_name": agent.get("name", ""),
                        "agent_description": agent.get("description", ""),
                        "node_count": agent.get("node_count") or len(agent.get("nodes", [])),
                        "url": active_service.get("url") or self._service_url(base_url, active_service),
                        "status": "running",
                    },
                ) or active_service
            self.stop_service(active_service["id"])

        service_id = uuid.uuid4().hex
        port = self._allocate_port()
        service_payload_path = (CONFIGS_DIR / f"agent-service-{service_id}.json").resolve()
        log_path = (LOGS_DIR / f"agent-service-{service_id}.log").resolve()
        service = {
            "id": service_id,
            "agent_id": agent_id,
            "agent_name": agent.get("name", ""),
            "agent_description": agent.get("description", ""),
            "node_count": agent.get("node_count") or len(agent.get("nodes", [])),
            "host": AGENT_SERVICE_HOST,
            "port": port,
            "url": "",
            "worker_url": f"http://{AGENT_SERVICE_HOST}:{port}/invoke",
            "health_url": f"http://{AGENT_SERVICE_HOST}:{port}/health",
            "pid": "",
            "payload_path": str(service_payload_path),
            "log_path": str(log_path),
            "process_mode": "subprocess",
            "status": "running",
            "invoke_count": 0,
            "last_invoked_at": "",
            "started_at": utc_now(),
            "stopped_at": "",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        service["url"] = self._service_url(base_url, service)

        service_payload_path.write_text(
            json.dumps(
                {
                    "service_id": service_id,
                    "agent_id": agent_id,
                    "host": AGENT_SERVICE_HOST,
                    "port": port,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        service = self.registry.add("agent_services", service)
        try:
            process = self._start_service_process(service)
        except Exception as exc:  # noqa: BLE001
            self.registry.update(
                "agent_services",
                service_id,
                {"status": "stopped", "stopped_at": utc_now(), "error": str(exc)},
            )
            raise
        updated = self.registry.update("agent_services", service_id, {"pid": process.pid, "status": "running"})
        return updated or service

    def stop_service(self, service_id: str) -> dict | None:
        service = self.registry.get("agent_services", service_id)
        if not service:
            return None
        was_running = service.get("status") == "running"
        self._terminate_service_process(service)
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

        if not self._is_service_healthy(service):
            self.registry.update(
                "agent_services",
                service_id,
                {"status": "stopped", "stopped_at": utc_now(), "error": "智能体服务子进程未运行。"},
            )
            raise RuntimeError("智能体服务子进程未运行，请重新开启服务。")

        return self._post_json(
            service.get("worker_url") or f"http://{service.get('host')}:{service.get('port')}/invoke",
            {"input_text": input_text, "include_trace": include_trace},
            timeout=AGENT_RUN_TIMEOUT_SECONDS,
        )

    def run_agent_once(self, agent_id: str, input_text: str, include_trace: bool = True) -> dict[str, Any]:
        if not self.agent_service.get_agent(agent_id):
            raise ValueError("智能体不存在。")
        return self._run_once_subprocess(
            {
                "mode": "saved",
                "agent_id": agent_id,
                "input_text": input_text,
                "include_trace": include_trace,
            }
        )

    def run_preview_once(self, agent_request: Any, input_text: str, include_trace: bool = True) -> dict[str, Any]:
        return self._run_once_subprocess(
            {
                "mode": "preview",
                "agent": agent_request.model_dump(),
                "input_text": input_text,
                "include_trace": include_trace,
            }
        )

    def _run_once_subprocess(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = uuid.uuid4().hex
        payload_path = (CONFIGS_DIR / f"agent-run-{run_id}.json").resolve()
        output_path = (CONFIGS_DIR / f"agent-run-{run_id}.result.json").resolve()
        log_path = (LOGS_DIR / f"agent-run-{run_id}.log").resolve()
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            process = self._popen(
                [sys.executable, "-m", "backend.services.agent_worker", "run-once", str(payload_path), str(output_path)],
                log,
            )
            try:
                return_code = process.wait(timeout=AGENT_RUN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as exc:
                self._terminate_process(process)
                raise RuntimeError("智能体测试运行超时，已停止子进程。") from exc

        try:
            result_payload = json.loads(output_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(f"智能体测试子进程未返回结果，日志：{log_path}") from exc
        finally:
            payload_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

        if return_code != 0 or not result_payload.get("ok"):
            message = result_payload.get("error") or f"智能体测试子进程退出码：{return_code}"
            raise RuntimeError(message)
        return result_payload["result"]

    def _start_service_process(self, service: dict) -> subprocess.Popen[str]:
        log_path = Path(service.get("log_path", "")).resolve()
        payload_path = Path(service.get("payload_path", "")).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8", errors="replace")
        log.write(f"Service ID: {service['id']}\n")
        log.write(f"Agent ID: {service['agent_id']}\n")
        log.write(f"URL: {service['url']}\n\n")
        log.flush()

        process = self._popen([sys.executable, "-m", "backend.services.agent_worker", "serve", str(payload_path)], log)
        self._service_processes[service["id"]] = process
        if not self._wait_for_health(service):
            self._terminate_process(process)
            raise RuntimeError(f"智能体服务子进程启动失败，请查看日志：{log_path}")
        return process

    def _popen(self, command: list[str], log_handle: Any) -> subprocess.Popen[str]:
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        kwargs: dict[str, Any] = {
            "args": command,
            "cwd": str(PROJECT_ROOT),
            "env": env,
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
            "text": True,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        return subprocess.Popen(**kwargs)

    def _wait_for_health(self, service: dict) -> bool:
        deadline = time.monotonic() + AGENT_SERVICE_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            process = self._service_processes.get(service.get("id"))
            if process and process.poll() is not None:
                return False
            if self._is_service_healthy(service):
                return True
            time.sleep(0.5)
        return False

    def _is_service_healthy(self, service: dict) -> bool:
        process = self._service_processes.get(service.get("id"))
        if process and process.poll() is not None:
            return False
        health_url = service.get("health_url") or f"http://{service.get('host')}:{service.get('port')}/health"
        try:
            payload = self._get_json(health_url, timeout=2)
        except Exception:  # noqa: BLE001
            return False
        return payload.get("status") == "running"

    def _sync_service_processes(self) -> None:
        for service in self.registry.list("agent_services"):
            if service.get("status") != "running":
                continue
            if self._is_service_healthy(service):
                continue
            self.registry.update(
                "agent_services",
                service["id"],
                {"status": "stopped", "stopped_at": utc_now(), "error": "智能体服务子进程未运行。"},
            )

    def _terminate_service_process(self, service: dict) -> None:
        process = self._service_processes.pop(service.get("id", ""), None)
        if process:
            self._terminate_process(process)
            return
        pid = service.get("pid")
        if pid:
            self._terminate_pid(pid)

    def _terminate_process(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            self._terminate_pid(process.pid)
        else:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except Exception:  # noqa: BLE001
                process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                self._terminate_pid(process.pid)
            else:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def _terminate_pid(self, pid: Any) -> None:
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            return
        if pid_int <= 0:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid_int), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
            return
        try:
            os.killpg(os.getpgid(pid_int), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            try:
                os.kill(pid_int, signal.SIGTERM)
            except Exception:  # noqa: BLE001
                return

    def _post_json(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(body).get("detail")
            except Exception:  # noqa: BLE001
                detail = body
            raise RuntimeError(detail or f"智能体服务调用失败：HTTP {exc.code}") from exc

    def _get_json(self, url: str, timeout: float) -> dict[str, Any]:
        with request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _allocate_port(self) -> int:
        used_ports = {
            int(service.get("port"))
            for service in self.registry.list("agent_services")
            if service.get("status") == "running" and str(service.get("port") or "").isdigit()
        }
        for port in range(AGENT_SERVICE_PORT_START, AGENT_SERVICE_PORT_END + 1):
            if port in used_ports:
                continue
            if self._port_available(port):
                return port
        raise RuntimeError("没有可用的智能体服务端口，请扩大 AGENT_SERVICE_PORT_START/END 范围。")

    def _port_available(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((AGENT_SERVICE_HOST, port))
            except OSError:
                return False
        return True

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

    def _service_url(self, base_url: str, service: dict) -> str:
        parsed = parse.urlparse(base_url)
        scheme = AGENT_SERVICE_PUBLIC_SCHEME or parsed.scheme or "http"
        host = AGENT_SERVICE_PUBLIC_HOST or parsed.netloc or AGENT_SERVICE_HOST
        return f"{scheme}://{host.rstrip('/')}/api/agent-services/{service.get('id')}/invoke"
