from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .safe_paths import safe_name
from ..config import BASE_MODELS_DIR, CONFIGS_DIR, LOGS_DIR, PROJECT_ROOT, is_path_inside
from ..schemas import BaseModelDownloadRequest
from ..storage import Registry, utc_now


ACTIVE_DOWNLOAD_STATUSES = {"pending", "running", "downloading"}


class ModelService:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self._process_lock = threading.RLock()
        self._download_processes: dict[str, subprocess.Popen] = {}
        self._cancelled_downloads: set[str] = set()

    def list_base_models(self) -> list[dict]:
        return sorted(self.registry.list("base_models"), key=lambda item: item.get("created_at", ""), reverse=True)

    def list_download_jobs(self) -> list[dict]:
        return sorted(self.registry.list("download_jobs"), key=lambda item: item.get("created_at", ""), reverse=True)

    def start_download(self, request: BaseModelDownloadRequest) -> dict:
        if request.source == "local" and not request.local_path:
            raise ValueError("登记本地模型时必须填写本地路径。")

        model_id = uuid.uuid4().hex
        job_id = uuid.uuid4().hex
        destination = (BASE_MODELS_DIR / f"{safe_name(request.display_name)}-{model_id[:8]}").resolve()
        local_path = Path(request.local_path).expanduser().resolve() if request.local_path else None

        model = {
            "id": model_id,
            "display_name": request.display_name,
            "model_id": request.model_id,
            "source": request.source,
            "path": str(local_path if request.source == "local" and local_path else destination),
            "status": "pending",
            "managed": request.source != "local",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        job = {
            "id": job_id,
            "model_record_id": model_id,
            "display_name": request.display_name,
            "status": "pending",
            "log_path": str(LOGS_DIR / f"download-{job_id}.log"),
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        self.registry.add("base_models", model)
        self.registry.add("download_jobs", job)

        if request.source == "local":
            thread = threading.Thread(target=self._download_worker, args=(request, model, job), daemon=True)
            thread.start()
        else:
            self._start_download_process(request, model, job)
        return job

    def delete_base_model(self, model_id: str) -> bool:
        active_statuses = {"pending", "queued", "running"}
        for job in self.registry.list("training_jobs"):
            if job.get("base_model_id") == model_id and job.get("status") in active_statuses:
                raise ValueError("该基座模型仍有关联的运行中训练任务，暂不能删除。")

        model = self.registry.delete("base_models", model_id)
        if not model:
            return False

        path = Path(model.get("path", "")).resolve()
        if model.get("managed") and is_path_inside(path, BASE_MODELS_DIR) and path.exists():
            shutil.rmtree(path)
        return True

    def delete_download_job_history(self, job_id: str) -> bool:
        job = self.registry.get("download_jobs", job_id)
        if not job:
            return False
        if job.get("status") in ACTIVE_DOWNLOAD_STATUSES:
            self._cancel_download_job(job)
            return True
        self.registry.delete("download_jobs", job_id)
        return True

    def _start_download_process(self, request: BaseModelDownloadRequest, model: dict, job: dict) -> None:
        log_path = Path(job["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path = CONFIGS_DIR / f"download-{job['id']}.json"
        payload = {
            "job_id": job["id"],
            "model_record_id": model["id"],
            "source": request.source,
            "repo_id": request.model_id,
            "destination": model["path"],
            "log_path": job["log_path"],
        }
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        command = [sys.executable, "-m", "backend.services.model_download_worker", str(payload_path)]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"Command: {' '.join(command)}\n")
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )

        with self._process_lock:
            self._download_processes[job["id"]] = process

        self.registry.update("base_models", model["id"], {"status": "downloading"})
        self.registry.update(
            "download_jobs",
            job["id"],
            {"status": "running", "pid": process.pid, "payload_path": str(payload_path)},
        )
        thread = threading.Thread(target=self._monitor_download_process, args=(job["id"], model["id"], process), daemon=True)
        thread.start()

    def _monitor_download_process(self, job_id: str, model_id: str, process: subprocess.Popen) -> None:
        return_code = process.wait()
        with self._process_lock:
            self._download_processes.pop(job_id, None)
            cancelled = job_id in self._cancelled_downloads
            if cancelled:
                self._cancelled_downloads.discard(job_id)

        job = self.registry.get("download_jobs", job_id)
        if not job or cancelled:
            return

        model = self.registry.get("base_models", model_id)
        if not model:
            self.registry.update("download_jobs", job_id, {"status": "failed", "error": "模型记录不存在，下载结果已忽略。"})
            return

        log_path = Path(job["log_path"])
        payload_path = Path(job.get("payload_path", ""))
        try:
            if return_code != 0:
                raise RuntimeError(f"下载进程异常退出，退出码：{return_code}")
            destination = Path(model["path"])
            self._ensure_download_has_files(destination)
            with log_path.open("a", encoding="utf-8") as log:
                log.write("Download completed.\n")
            self.registry.update("base_models", model_id, {"status": "ready"})
            self.registry.update("download_jobs", job_id, {"status": "succeeded"})
        except Exception as exc:  # noqa: BLE001
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"ERROR: {exc}\n")
            self.registry.update("base_models", model_id, {"status": "failed", "error": str(exc)})
            self.registry.update("download_jobs", job_id, {"status": "failed", "error": str(exc)})
        finally:
            if payload_path.is_file():
                payload_path.unlink(missing_ok=True)

    def _cancel_download_job(self, job: dict) -> None:
        job_id = job["id"]
        with self._process_lock:
            self._cancelled_downloads.add(job_id)
            process = self._download_processes.get(job_id)

        self._terminate_download_process(process, job.get("pid"))

        model = self.registry.get("base_models", job.get("model_record_id", ""))
        if model:
            self._delete_partial_model(model)
            self.registry.delete("base_models", model["id"])

        payload_path = Path(job.get("payload_path", ""))
        if payload_path.is_file():
            payload_path.unlink(missing_ok=True)

        log_path = Path(job.get("log_path", ""))
        if log_path.is_file() and is_path_inside(log_path, LOGS_DIR):
            log_path.unlink(missing_ok=True)

        self.registry.delete("download_jobs", job_id)

    def _delete_partial_model(self, model: dict) -> None:
        path = Path(model.get("path", "")).resolve()
        if model.get("managed") and is_path_inside(path, BASE_MODELS_DIR) and path.exists():
            shutil.rmtree(path, ignore_errors=True)

    def _terminate_download_process(self, process: subprocess.Popen | None, pid_value: object) -> None:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=8)
            return

        try:
            pid = int(pid_value)
        except (TypeError, ValueError):
            return
        if pid <= 0:
            return

        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return

        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except OSError:
            return
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            return

    def _download_worker(self, request: BaseModelDownloadRequest, model: dict, job: dict) -> None:
        log_path = Path(job["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry.update("base_models", model["id"], {"status": "downloading"})
        self.registry.update("download_jobs", job["id"], {"status": "running"})

        try:
            with log_path.open("a", encoding="utf-8") as log:
                if request.source == "local":
                    path = Path(request.local_path or "").expanduser().resolve()
                    if not path.exists():
                        raise FileNotFoundError(f"本地路径不存在：{path}")
                    log.write(f"Registered local model: {path}\n")
                    self.registry.update("base_models", model["id"], {"path": str(path), "status": "ready"})
                elif request.source == "huggingface":
                    from huggingface_hub import snapshot_download

                    destination = Path(model["path"])
                    destination.mkdir(parents=True, exist_ok=True)
                    log.write(f"Downloading {request.model_id} from Hugging Face to {destination}\n")
                    snapshot_path = snapshot_download(
                        repo_id=request.model_id,
                        local_dir=str(destination),
                    )
                    log.write(f"Snapshot path: {snapshot_path}\n")
                    self._ensure_download_has_files(destination)
                    log.write("Download completed.\n")
                    self.registry.update("base_models", model["id"], {"status": "ready"})
                else:
                    try:
                        from modelscope import snapshot_download as modelscope_snapshot_download
                    except ImportError as exc:
                        raise RuntimeError("使用 ModelScope 下载前，请先安装依赖：pip install modelscope") from exc

                    destination = Path(model["path"])
                    destination.mkdir(parents=True, exist_ok=True)
                    log.write(f"Downloading {request.model_id} from ModelScope to {destination}\n")
                    try:
                        snapshot_path = modelscope_snapshot_download(request.model_id, local_dir=str(destination))
                    except TypeError:
                        snapshot_path = modelscope_snapshot_download(request.model_id, cache_dir=str(destination))
                    log.write(f"Snapshot path: {snapshot_path}\n")
                    self._copy_snapshot_to_destination(Path(snapshot_path), destination)
                    self._ensure_download_has_files(destination)
                    log.write("Download completed.\n")
                    self.registry.update("base_models", model["id"], {"status": "ready"})

            self.registry.update("download_jobs", job["id"], {"status": "succeeded"})
        except Exception as exc:  # noqa: BLE001
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"ERROR: {exc}\n")
            self.registry.update("base_models", model["id"], {"status": "failed", "error": str(exc)})
            self.registry.update("download_jobs", job["id"], {"status": "failed", "error": str(exc)})

    def _copy_snapshot_to_destination(self, snapshot_path: Path, destination: Path) -> None:
        if snapshot_path.resolve() == destination.resolve():
            return
        if not snapshot_path.exists():
            return
        has_destination_files = any(item.is_file() for item in destination.rglob("*"))
        if has_destination_files:
            return
        shutil.copytree(snapshot_path, destination, dirs_exist_ok=True)

    def _ensure_download_has_files(self, destination: Path) -> None:
        if not destination.exists() or not any(item.is_file() for item in destination.rglob("*")):
            raise RuntimeError(f"模型下载目录为空：{destination}")
