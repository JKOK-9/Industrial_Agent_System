from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseModelDownloadRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    display_name: str = Field(min_length=1, max_length=120)
    model_id: str = Field(min_length=1, max_length=300)
    source: Literal["huggingface", "modelscope", "local", "api"] = "huggingface"
    local_path: str | None = None
    api_base_url: str | None = Field(default=None, max_length=1000)
    api_key: str | None = Field(default=None, max_length=4000)
    api_model: str | None = Field(default=None, max_length=300)
    api_provider: str = Field(default="openai_compatible", max_length=80)


class TrainingRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    output_name: str = Field(min_length=1, max_length=120)
    domain: str = Field(default="", max_length=120)
    dataset_format: Literal["alpaca", "sharegpt", "openai"] = "alpaca"
    training_method: Literal["lora"] = "lora"
    template: str = "default"
    cutoff_len: int = Field(default=2048, ge=128, le=32768)
    num_train_epochs: float = Field(default=3, gt=0, le=100)
    learning_rate: float = Field(default=2e-4, gt=0, le=1)
    per_device_train_batch_size: int = Field(default=1, ge=1, le=512)
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=1024)
    logging_steps: int = Field(default=10, ge=1, le=10000)
    save_steps: int = Field(default=500, ge=1, le=100000)
    lora_rank: int = Field(default=8, ge=1, le=1024)
    lora_alpha: int = Field(default=16, ge=1, le=4096)
    lora_dropout: float = Field(default=0.05, ge=0, le=1)
    lora_target: str = "all"
    fp16: bool = True
    bf16: bool = False


class RagConfigRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    display_name: str = Field(min_length=1, max_length=120)
    domain: str = Field(default="", max_length=120)
    prompt: str = Field(min_length=1, max_length=5000)


class RagPromptRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=4, ge=1, le=12)


class KnowledgeSourceRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    source_type: Literal["text", "table", "json", "markdown"] = "text"
    description: str = Field(default="", max_length=300)


class PromptAssetRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    prompt_type: Literal["system", "instruction", "template", "other"] = "system"
    description: str = Field(default="", max_length=300)
    content: str = Field(default="", max_length=20000)


class BatchDeleteRequest(BaseModel):
    ids: list[str] = Field(default_factory=list, min_length=1)


class WorkflowNode(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    type: Literal["input", "output", "llm", "finetuned", "knowledge"]
    label: str = Field(min_length=1, max_length=80)
    config: dict = Field(default_factory=dict)


class AgentWorkflowRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=300)
    nodes: list[WorkflowNode] = Field(default_factory=list, min_length=2)


class AgentRunRequest(BaseModel):
    input_text: str = Field(default="", max_length=8000)
