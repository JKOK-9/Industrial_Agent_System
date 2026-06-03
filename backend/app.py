from __future__ import annotations

from flask import Flask, jsonify, request
from flask_cors import CORS
from pydantic import ValidationError

from .config import CORS_ORIGINS, CORS_SUPPORTS_CREDENTIALS, ensure_runtime_dirs
from .schemas import (
    AgentRunRequest,
    AgentWorkflowRequest,
    BaseModelDownloadRequest,
    BatchDeleteRequest,
    KnowledgeSourceRequest,
    PromptAssetRequest,
    RagConfigRequest,
    RagPromptRequest,
    TrainingRequest,
)
from .services.dataset_service import DatasetValidationError
from .services.agent_api_service import AgentApiService
from .services.agent_service import AgentService
from .services.model_service import ModelService
from .services.model_runtime import ModelRuntime
from .services.rag_service import RagService
from .services.resource_service import ResourceService
from .services.training_service import TrainingService
from .storage import Registry


registry = Registry()
model_service = ModelService(registry)
training_service = TrainingService(registry)
rag_service = RagService(registry)
resource_service = ResourceService(registry)
model_runtime = ModelRuntime()
agent_service = AgentService(registry, model_runtime, rag_service)
agent_api_service = AgentApiService(registry, agent_service)


def create_app() -> Flask:
    ensure_runtime_dirs()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": CORS_ORIGINS,
                "supports_credentials": CORS_SUPPORTS_CREDENTIALS,
            }
        },
    )

    @app.errorhandler(ValidationError)
    def handle_validation_error(error: ValidationError):
        return jsonify({"detail": error.errors(include_context=False, include_url=False)}), 422

    @app.errorhandler(DatasetValidationError)
    def handle_dataset_error(error: DatasetValidationError):
        return jsonify({"detail": str(error)}), 422

    @app.errorhandler(ValueError)
    def handle_value_error(error: ValueError):
        return jsonify({"detail": str(error)}), 400

    @app.errorhandler(RuntimeError)
    def handle_runtime_error(error: RuntimeError):
        return jsonify({"detail": str(error)}), 500

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/models/base")
    def list_base_models():
        return jsonify({"items": model_service.list_base_models()})

    @app.get("/api/models/runtime")
    def list_loaded_model_sessions():
        return jsonify({"loaded_models": model_runtime.loaded_models()})

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

    @app.get("/api/resources/knowledge-sources")
    def list_knowledge_sources():
        return jsonify({"items": resource_service.list_knowledge_sources()})

    @app.post("/api/resources/knowledge-sources")
    def create_knowledge_source():
        knowledge_file = request.files.get("knowledge_file")
        if not knowledge_file:
            return jsonify({"detail": "请上传知识源文件。"}), 422
        payload = {
            "name": request.form.get("name"),
            "source_type": request.form.get("source_type", "text"),
            "description": request.form.get("description", ""),
        }
        source_request = KnowledgeSourceRequest.model_validate(payload)
        item = resource_service.create_knowledge_source(source_request, knowledge_file)
        return jsonify({"item": item}), 201

    @app.delete("/api/resources/knowledge-sources/<item_id>")
    def delete_knowledge_source(item_id: str):
        if not resource_service.delete_knowledge_source(item_id):
            return jsonify({"detail": "知识源不存在。"}), 404
        return jsonify({"ok": True})

    @app.post("/api/resources/knowledge-sources/batch-delete")
    def batch_delete_knowledge_sources():
        payload = request.get_json(silent=True) or {}
        batch_request = BatchDeleteRequest.model_validate(payload)
        return jsonify(resource_service.batch_delete_knowledge_sources(batch_request.ids))

    @app.get("/api/resources/prompts")
    def list_prompt_assets():
        return jsonify({"items": resource_service.list_prompt_assets()})

    @app.post("/api/resources/prompts")
    def create_prompt_asset():
        payload = {
            "name": request.form.get("name"),
            "prompt_type": request.form.get("prompt_type", "system"),
            "description": request.form.get("description", ""),
            "content": request.form.get("content", ""),
        }
        prompt_request = PromptAssetRequest.model_validate(payload)
        item = resource_service.create_prompt_asset(prompt_request, request.files.get("prompt_file"))
        return jsonify({"item": item}), 201

    @app.delete("/api/resources/prompts/<item_id>")
    def delete_prompt_asset(item_id: str):
        if not resource_service.delete_prompt_asset(item_id):
            return jsonify({"detail": "提示词不存在。"}), 404
        return jsonify({"ok": True})

    @app.post("/api/resources/prompts/batch-delete")
    def batch_delete_prompt_assets():
        payload = request.get_json(silent=True) or {}
        batch_request = BatchDeleteRequest.model_validate(payload)
        return jsonify(resource_service.batch_delete_prompt_assets(batch_request.ids))

    @app.get("/api/agents")
    def list_agents():
        return jsonify({"items": agent_service.list_agents()})

    @app.post("/api/agents")
    def save_agent():
        payload = request.get_json(silent=True) or {}
        agent_request = AgentWorkflowRequest.model_validate(payload)
        return jsonify({"item": agent_service.save_agent(agent_request)}), 201

    @app.get("/api/agents/<agent_id>")
    def get_agent(agent_id: str):
        agent = agent_service.get_agent(agent_id)
        if not agent:
            return jsonify({"detail": "智能体不存在。"}), 404
        return jsonify({"item": agent})

    @app.delete("/api/agents/<agent_id>")
    def delete_agent(agent_id: str):
        agent_api_service.stop_services_for_agent(agent_id)
        if not agent_service.delete_agent(agent_id):
            return jsonify({"detail": "智能体不存在。"}), 404
        return jsonify({"ok": True})

    @app.get("/api/agent-services")
    def list_agent_services():
        return jsonify({"items": agent_api_service.list_services()})

    @app.post("/api/agents/<agent_id>/service")
    def start_agent_service(agent_id: str):
        service = agent_api_service.start_service(agent_id, request.url_root)
        return jsonify({"service": service}), 201

    @app.delete("/api/agent-services/<service_id>")
    def stop_agent_service(service_id: str):
        service = agent_api_service.stop_service(service_id)
        if not service:
            return jsonify({"detail": "智能体服务不存在。"}), 404
        return jsonify({"service": service})

    @app.post("/api/agent-services/<service_id>/stop")
    def stop_agent_service_post(service_id: str):
        service = agent_api_service.stop_service(service_id)
        if not service:
            return jsonify({"detail": "智能体服务不存在。"}), 404
        return jsonify({"service": service})

    @app.post("/api/agent-services/<service_id>/invoke")
    def invoke_agent_service(service_id: str):
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
        return jsonify(agent_api_service.invoke(service_id, str(input_text), bool(include_trace)))

    @app.post("/api/agents/<agent_id>/run")
    def run_agent(agent_id: str):
        payload = request.get_json(silent=True) or {}
        run_request = AgentRunRequest.model_validate(payload)
        return jsonify(agent_api_service.run_agent_once(agent_id, run_request.input_text))

    @app.post("/api/agents/preview-run")
    def run_agent_preview():
        payload = request.get_json(silent=True) or {}
        agent_request = AgentWorkflowRequest.model_validate(payload.get("agent") or {})
        run_request = AgentRunRequest.model_validate({"input_text": payload.get("input_text", "")})
        return jsonify(agent_api_service.run_preview_once(agent_request, run_request.input_text))

    @app.delete("/api/download-jobs/<job_id>")
    def delete_download_job_history(job_id: str):
        try:
            deleted = model_service.delete_download_job_history(job_id)
        except ValueError as exc:
            return jsonify({"detail": str(exc)}), 409
        if not deleted:
            return jsonify({"detail": "下载任务记录不存在。"}), 404
        return jsonify({"ok": True})

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
            "domain": request.form.get("domain", ""),
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

    @app.post("/api/training/jobs/<job_id>/stop")
    def stop_training_job(job_id: str):
        job = training_service.stop_training_job(job_id)
        if not job:
            return jsonify({"detail": "训练任务记录不存在。"}), 404
        return jsonify({"job": job})

    @app.post("/api/rag/models")
    def create_rag_model():
        knowledge_file = request.files.get("knowledge_file")
        if not knowledge_file:
            return jsonify({"detail": "请上传 RAG 数据文件。"}), 422

        payload = {
            "model_id": request.form.get("model_id"),
            "display_name": request.form.get("display_name"),
            "domain": request.form.get("domain", ""),
            "prompt": request.form.get("prompt"),
        }
        rag_request = RagConfigRequest.model_validate(payload)
        model = rag_service.create_rag_model(rag_request, knowledge_file)
        return jsonify({"model": model}), 201

    @app.post("/api/rag/models/<model_id>/prompt")
    def build_rag_prompt(model_id: str):
        payload = request.get_json(silent=True) or {}
        prompt_request = RagPromptRequest.model_validate(payload)
        return jsonify(rag_service.build_prompt(model_id, prompt_request.question, prompt_request.top_k))

    @app.delete("/api/training/jobs/<job_id>")
    def delete_training_job_history(job_id: str):
        try:
            deleted = training_service.delete_job_history(job_id)
        except ValueError as exc:
            return jsonify({"detail": str(exc)}), 409
        if not deleted:
            return jsonify({"detail": "训练任务记录不存在。"}), 404
        return jsonify({"ok": True})

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
