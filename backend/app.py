from __future__ import annotations

from flask import Flask, jsonify, request
from flask_cors import CORS
from pydantic import ValidationError

from .config import ensure_runtime_dirs
from .schemas import BaseModelDownloadRequest, TrainingRequest
from .services.dataset_service import DatasetValidationError
from .services.model_service import ModelService
from .services.training_service import TrainingService
from .storage import Registry


registry = Registry()
model_service = ModelService(registry)
training_service = TrainingService(registry)


def create_app() -> Flask:
    ensure_runtime_dirs()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    @app.errorhandler(ValidationError)
    def handle_validation_error(error: ValidationError):
        return jsonify({"detail": error.errors(include_context=False, include_url=False)}), 422

    @app.errorhandler(DatasetValidationError)
    def handle_dataset_error(error: DatasetValidationError):
        return jsonify({"detail": str(error)}), 422

    @app.errorhandler(ValueError)
    def handle_value_error(error: ValueError):
        return jsonify({"detail": str(error)}), 400

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/models/base")
    def list_base_models():
        return jsonify({"items": model_service.list_base_models()})

    @app.post("/api/models/base/download")
    def download_base_model():
        payload = request.get_json(silent=True) or {}
        download_request = BaseModelDownloadRequest.model_validate(payload)
        job = model_service.start_download(download_request)
        return jsonify({"job": job}), 202

    @app.delete("/api/models/base/<model_id>")
    def delete_base_model(model_id: str):
        try:
            deleted = model_service.delete_base_model(model_id)
        except ValueError as exc:
            return jsonify({"detail": str(exc)}), 409
        if not deleted:
            return jsonify({"detail": "基座模型不存在。"}), 404
        return jsonify({"ok": True})

    @app.get("/api/download-jobs")
    def list_download_jobs():
        return jsonify({"items": model_service.list_download_jobs()})

    @app.get("/api/training/jobs")
    def list_training_jobs():
        return jsonify({"items": training_service.list_jobs()})

    @app.post("/api/training/jobs")
    def submit_training():
        dataset_file = request.files.get("dataset_file")
        if not dataset_file:
            return jsonify({"detail": "请上传微调数据文件。"}), 422

        payload = {
            "model_id": request.form.get("model_id"),
            "output_name": request.form.get("output_name"),
            "dataset_format": request.form.get("dataset_format", "alpaca"),
            "training_method": request.form.get("training_method", "lora"),
            "template": request.form.get("template", "default"),
            "cutoff_len": request.form.get("cutoff_len", 2048),
            "num_train_epochs": request.form.get("num_train_epochs", 3),
            "learning_rate": request.form.get("learning_rate", 0.0002),
            "per_device_train_batch_size": request.form.get("per_device_train_batch_size", 1),
            "gradient_accumulation_steps": request.form.get("gradient_accumulation_steps", 8),
            "logging_steps": request.form.get("logging_steps", 10),
            "save_steps": request.form.get("save_steps", 500),
            "lora_rank": request.form.get("lora_rank", 8),
            "lora_alpha": request.form.get("lora_alpha", 16),
            "lora_dropout": request.form.get("lora_dropout", 0.05),
            "lora_target": request.form.get("lora_target", "all"),
            "fp16": request.form.get("fp16", "true"),
            "bf16": request.form.get("bf16", "false"),
        }
        training_request = TrainingRequest.model_validate(payload)
        job = training_service.submit_training(training_request, dataset_file)
        return jsonify({"job": job}), 202

    @app.get("/api/training/jobs/<job_id>/logs")
    def training_logs(job_id: str):
        return training_service.read_logs(job_id), 200, {"Content-Type": "text/plain; charset=utf-8"}

    @app.get("/api/models/finetuned")
    def list_fine_tuned_models():
        return jsonify({"items": training_service.list_fine_tuned_models()})

    @app.delete("/api/models/finetuned/<model_id>")
    def delete_fine_tuned_model(model_id: str):
        if not training_service.delete_fine_tuned_model(model_id):
            return jsonify({"detail": "微调模型不存在。"}), 404
        return jsonify({"ok": True})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
