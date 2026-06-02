from __future__ import annotations

import os
import signal
import shlex
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

import yaml

from .dataset_service import DatasetValidationError, UploadedFile, remove_dataset_dir, save_and_prepare_dataset
from .safe_paths import safe_name
from ..config import CONFIGS_DIR, DATASETS_DIR, LLAMAFACTORY_CLI, LLAMAFACTORY_WORKDIR, LOGS_DIR, OUTPUTS_DIR, PROJECT_ROOT, is_path_inside
from ..schemas import TrainingRequest
from ..storage import Registry, utc_now


ACTIVE_TRAINING_STATUSES = {"queued", "running"}


class TrainingService:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._cancelled_jobs: set[str] = set()
        self._lock = threading.Lock()
        self._mark_interrupted_jobs()

    def list_jobs(self) -> list[dict]:
        return sorted(self.registry.list("training_jobs"), key=lambda item: item.get("created_at", ""), reverse=True)

    def list_fine_tuned_models(self) -> list[dict]:
        return sorted(self.registry.list("fine_tuned_models"), key=lambda item: item.get("created_at", ""), reverse=True)

    def submit_training(self, request: TrainingRequest, dataset_file: UploadedFile) -> dict:
        base_model = self.registry.get("base_models", request.model_id)
        if not base_model:
            raise ValueError("基座模型不存在。")
        if base_model.get("status") != "ready":
            raise ValueError("基座模型尚未 ready。")

        job_id = uuid.uuid4().hex
        safe_output_name = safe_name(request.output_name)
        dataset_name = f"dataset_{job_id[:8]}"
        dataset_dir = (DATASETS_DIR / job_id).resolve()
        output_dir = (OUTPUTS_DIR / f"{safe_output_name}-{job_id[:8]}").resolve()
        config_path = (CONFIGS_DIR / f"{job_id}.yaml").resolve()
        log_path = (LOGS_DIR / f"train-{job_id}.log").resolve()

        try:
            dataset_meta = save_and_prepare_dataset(
                upload=dataset_file,
                dataset_dir=dataset_dir,
                dataset_name=dataset_name,
                dataset_format=request.dataset_format,
            )
        except DatasetValidationError:
            remove_dataset_dir(dataset_dir)
            raise

        train_config = self._build_llamafactory_config(
            request=request,
            base_model_path=base_model["path"],
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
        )
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(train_config, handle, sort_keys=False, allow_unicode=True)

        job = {
            "id": job_id,
            "status": "queued",
            "output_name": request.output_name,
            "domain": request.domain,
            "base_model_id": base_model["id"],
            "base_model_name": base_model["display_name"],
            "training_method": request.training_method,
            "dataset": dataset_meta,
            "config_path": str(config_path),
            "output_dir": str(output_dir),
            "log_path": str(log_path),
            "params": request.model_dump(),
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        self.registry.add("training_jobs", job)

        thread = threading.Thread(target=self._training_worker, args=(job,), daemon=True)
        thread.start()
        return job

    def delete_fine_tuned_model(self, model_id: str) -> bool:
        model = self.registry.delete("fine_tuned_models", model_id)
        if not model:
            return False
        output_dir = Path(model.get("path", "")).resolve()
        if is_path_inside(output_dir, OUTPUTS_DIR) and output_dir.exists():
            shutil.rmtree(output_dir)
        return True

    def read_logs(self, job_id: str) -> str:
        job = self.registry.get("training_jobs", job_id)
        if not job:
            return ""
        path = Path(job.get("log_path", ""))
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")[-20000:]

    def stop_training_job(self, job_id: str) -> dict | None:
        job = self.registry.get("training_jobs", job_id)
        if not job:
            return None
        if job.get("status") not in ACTIVE_TRAINING_STATUSES:
            return job

        self._mark_cancelled(job_id)
        self._terminate_training_process(job)
        return self.registry.update(
            "training_jobs",
            job_id,
            {
                "status": "interrupted",
                "finished_at": utc_now(),
                "updated_at": utc_now(),
                "error": "训练任务已手动停止。",
            },
        )

    def delete_job_history(self, job_id: str) -> bool:
        job = self.registry.get("training_jobs", job_id)
        if not job:
            return False
        status = job.get("status")
        if status in ACTIVE_TRAINING_STATUSES:
            self._mark_cancelled(job_id)
            self._terminate_training_process(job)
            self._cleanup_training_artifacts(job, remove_output=True, remove_log=True)
            self._delete_models_from_job(job_id, remove_files=True)
        elif status in {"failed", "interrupted"}:
            self._cleanup_training_artifacts(job, remove_output=True, remove_log=True)
            self._delete_models_from_job(job_id, remove_files=True)
        self.registry.delete("training_jobs", job_id)
        return True

    def _build_llamafactory_config(
        self,
        request: TrainingRequest,
        base_model_path: str,
        dataset_name: str,
        dataset_dir: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        return {
            "model_name_or_path": base_model_path,
            "stage": "sft",
            "do_train": True,
            "finetuning_type": request.training_method,
            "lora_target": request.lora_target,
            "dataset": dataset_name,
            "dataset_dir": str(dataset_dir),
            "template": request.template,
            "cutoff_len": request.cutoff_len,
            "overwrite_cache": True,
            "preprocessing_num_workers": 1,
            "output_dir": str(output_dir),
            "logging_steps": request.logging_steps,
            "save_steps": request.save_steps,
            "plot_loss": True,
            "overwrite_output_dir": True,
            "per_device_train_batch_size": request.per_device_train_batch_size,
            "gradient_accumulation_steps": request.gradient_accumulation_steps,
            "learning_rate": request.learning_rate,
            "num_train_epochs": request.num_train_epochs,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "bf16": request.bf16,
            "fp16": request.fp16 and not request.bf16,
            "lora_rank": request.lora_rank,
            "lora_alpha": request.lora_alpha,
            "lora_dropout": request.lora_dropout,
            "report_to": "none",
        }

    def _training_worker(self, job: dict) -> None:
        job_id = job["id"]
        if self._consume_cancelled(job_id):
            return

        log_path = Path(job["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output_dir = Path(job["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        self.registry.update("training_jobs", job_id, {"status": "running", "started_at": utc_now(), "updated_at": utc_now()})
        if self._consume_cancelled(job_id):
            return

        command = _llamafactory_command(job["config_path"])
        cwd = Path(LLAMAFACTORY_WORKDIR).resolve() if LLAMAFACTORY_WORKDIR else PROJECT_ROOT
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")

        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write(f"Command: {' '.join(command)}\n")
            log.write(f"Python executable: {sys.executable}\n")
            log.write(f"Working directory: {cwd}\n\n")
            log.flush()
            try:
                if self._consume_cancelled(job_id):
                    return

                popen_kwargs: dict[str, Any] = {
                    "args": command,
                    "cwd": str(cwd),
                    "env": env,
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.STDOUT,
                    "text": True,
                    "encoding": "utf-8",
                    "errors": "replace",
                }
                if os.name == "nt":
                    popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
                        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                    )
                else:
                    popen_kwargs["start_new_session"] = True

                process = subprocess.Popen(**popen_kwargs)
                with self._lock:
                    self._processes[job_id] = process
                    cancelled_after_start = job_id in self._cancelled_jobs
                self.registry.update("training_jobs", job_id, {"pid": process.pid, "updated_at": utc_now()})
                if cancelled_after_start:
                    self._terminate_training_process({**job, "pid": process.pid})
                    self._consume_cancelled(job_id)
                    return

                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()

                return_code = process.wait()
                with self._lock:
                    self._processes.pop(job_id, None)
                if self._consume_cancelled(job_id):
                    return

                if return_code == 0:
                    fine_tuned_model = {
                        "id": uuid.uuid4().hex,
                        "display_name": job["output_name"],
                        "domain": job.get("domain", ""),
                        "base_model_id": job["base_model_id"],
                        "base_model_name": job["base_model_name"],
                        "training_job_id": job_id,
                        "training_method": job["training_method"],
                        "path": str(output_dir),
                        "status": "ready",
                        "created_at": utc_now(),
                        "updated_at": utc_now(),
                    }
                    self.registry.add("fine_tuned_models", fine_tuned_model)
                    self.registry.update(
                        "training_jobs",
                        job_id,
                        {"status": "succeeded", "finished_at": utc_now(), "updated_at": utc_now()},
                    )
                else:
                    self.registry.update(
                        "training_jobs",
                        job_id,
                        {
                            "status": "failed",
                            "finished_at": utc_now(),
                            "updated_at": utc_now(),
                            "error": f"llamafactory-cli exited with {return_code}",
                        },
                    )
            except FileNotFoundError:
                if self._consume_cancelled(job_id):
                    return
                message = f"找不到 LLaMA-Factory 命令：{' '.join(command)}。请安装 llamafactory 或设置 LLAMAFACTORY_CLI。"
                log.write(f"\nERROR: {message}\n")
                self.registry.update(
                    "training_jobs",
                    job_id,
                    {"status": "failed", "finished_at": utc_now(), "updated_at": utc_now(), "error": message},
                )
                with self._lock:
                    self._processes.pop(job_id, None)
            except Exception as exc:  # noqa: BLE001
                if self._consume_cancelled(job_id):
                    return
                log.write(f"\nERROR: {exc}\n")
                self.registry.update(
                    "training_jobs",
                    job_id,
                    {"status": "failed", "finished_at": utc_now(), "updated_at": utc_now(), "error": str(exc)},
                )
                with self._lock:
                    self._processes.pop(job_id, None)

    def _mark_cancelled(self, job_id: str) -> None:
        with self._lock:
            self._cancelled_jobs.add(job_id)

    def _consume_cancelled(self, job_id: str) -> bool:
        with self._lock:
            if job_id not in self._cancelled_jobs:
                return False
            self._cancelled_jobs.discard(job_id)
            return True

    def _terminate_training_process(self, job: dict) -> None:
        job_id = job.get("id")
        process: subprocess.Popen[str] | None = None
        with self._lock:
            if job_id:
                process = self._processes.pop(job_id, None)

        pid = getattr(process, "pid", None) or job.get("pid")
        if process and process.poll() is None:
            if os.name == "nt" and pid:
                self._terminate_pid(pid)
            else:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except Exception:  # noqa: BLE001
                    process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "nt" and pid:
                    self._terminate_pid(pid)
                else:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except Exception:  # noqa: BLE001
                        process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            return

        if pid:
            self._terminate_pid(pid)

    def _terminate_pid(self, pid: Any) -> None:
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            return
        if pid_int <= 0:
            return

        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid_int), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                )
            except Exception:  # noqa: BLE001
                return
            return

        try:
            os.killpg(os.getpgid(pid_int), signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:  # noqa: BLE001
            try:
                os.kill(pid_int, signal.SIGTERM)
            except Exception:  # noqa: BLE001
                return

    def _cleanup_training_artifacts(self, job: dict, *, remove_output: bool, remove_log: bool) -> None:
        dataset_dir = Path((job.get("dataset") or {}).get("dataset_dir", "")).resolve()
        self._safe_remove_tree(dataset_dir, DATASETS_DIR)

        config_path = Path(job.get("config_path", "")).resolve()
        self._safe_unlink(config_path, CONFIGS_DIR)

        if remove_log:
            log_path = Path(job.get("log_path", "")).resolve()
            self._safe_unlink(log_path, LOGS_DIR)

        if remove_output:
            output_dir = Path(job.get("output_dir", "")).resolve()
            self._safe_remove_tree(output_dir, OUTPUTS_DIR)

    def _delete_models_from_job(self, job_id: str, *, remove_files: bool) -> None:
        for model in list(self.registry.list("fine_tuned_models")):
            if model.get("training_job_id") != job_id:
                continue
            if remove_files:
                self.delete_fine_tuned_model(model["id"])
            else:
                self.registry.delete("fine_tuned_models", model["id"])

    def _safe_remove_tree(self, path: Path, root: Path) -> None:
        if is_path_inside(path, root) and path.exists():
            shutil.rmtree(path, ignore_errors=True)

    def _safe_unlink(self, path: Path, root: Path) -> None:
        if is_path_inside(path, root) and path.exists() and path.is_file():
            path.unlink(missing_ok=True)

    def _mark_interrupted_jobs(self) -> None:
        for job in self.registry.list("training_jobs"):
            if job.get("status") in ACTIVE_TRAINING_STATUSES:
                self.registry.update(
                    "training_jobs",
                    job["id"],
                    {"status": "interrupted", "updated_at": utc_now(), "error": "服务重启后无法继续追踪该训练进程。"},
                )


def _llamafactory_command(config_path: str) -> list[str]:
    configured = (LLAMAFACTORY_CLI or "").strip()
    if configured and configured != "llamafactory-cli":
        return [*shlex.split(configured), "train", config_path]

    cli = _find_executable("llamafactory-cli") or _find_executable("lmf")
    if cli:
        return [cli, "train", config_path]

    return [sys.executable, "-m", "llamafactory.cli", "train", config_path]


def _find_executable(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found

    executable = Path(sys.executable).resolve()
    candidates = [
        executable.parent / name,
        executable.parent / f"{name}.exe",
        executable.parent / "Scripts" / name,
        executable.parent / "Scripts" / f"{name}.exe",
        executable.parent.parent / "Scripts" / name,
        executable.parent.parent / "Scripts" / f"{name}.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None
