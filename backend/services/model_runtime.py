from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MODEL_RUNTIME_MAX_NEW_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("MODEL_RUNTIME_TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.getenv("MODEL_RUNTIME_TOP_P", "0.9"))


@dataclass
class GenerationOptions:
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    repetition_penalty: float = 1.05


class ModelSession:
    def __init__(self, runtime_key: str, tokenizer: Any, model: Any, torch_module: Any) -> None:
        self.runtime_key = runtime_key
        self.tokenizer = tokenizer
        self.model = model
        self.torch = torch_module
        self._lock = threading.Lock()

    def generate(self, prompt: str, options: GenerationOptions | None = None) -> str:
        options = options or GenerationOptions()
        tokenizer = self.tokenizer
        model = self.model
        torch = self.torch

        rendered_prompt = self._render_prompt(prompt)
        inputs = tokenizer(rendered_prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": options.max_new_tokens,
            "do_sample": options.temperature > 0,
            "top_p": options.top_p,
            "repetition_penalty": options.repetition_penalty,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if options.temperature > 0:
            generation_kwargs["temperature"] = options.temperature

        with self._lock:
            with torch.inference_mode():
                output_ids = model.generate(**inputs, **generation_kwargs)

        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_length:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _render_prompt(self, prompt: str) -> str:
        if os.getenv("MODEL_RUNTIME_DISABLE_CHAT_TEMPLATE", "").lower() in {"1", "true", "yes"}:
            return prompt
        if not getattr(self.tokenizer, "chat_template", None):
            return prompt
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # noqa: BLE001
            return prompt


class ModelRuntime:
    def __init__(self) -> None:
        self._sessions: dict[str, ModelSession] = {}
        self._lock = threading.RLock()

    def generate_base(self, base_model: dict, prompt: str, options: GenerationOptions | None = None) -> str:
        runtime_key = self._base_runtime_key(base_model)
        session = self._get_or_load(runtime_key=runtime_key, model_path=base_model.get("path", ""))
        return session.generate(prompt, options)

    def generate_fine_tuned(self, base_model: dict, fine_tuned_model: dict, prompt: str, options: GenerationOptions | None = None) -> str:
        adapter_path = Path(fine_tuned_model.get("path", "")).resolve()
        if (adapter_path / "adapter_config.json").is_file():
            runtime_key = self._adapter_runtime_key(base_model, fine_tuned_model)
            session = self._get_or_load(runtime_key=runtime_key, model_path=base_model.get("path", ""), adapter_path=str(adapter_path))
        else:
            runtime_key = self._full_model_runtime_key(fine_tuned_model)
            session = self._get_or_load(runtime_key=runtime_key, model_path=str(adapter_path))
        return session.generate(prompt, options)

    def loaded_models(self) -> list[str]:
        with self._lock:
            return sorted(self._sessions)

    def _get_or_load(self, runtime_key: str, model_path: str, adapter_path: str | None = None) -> ModelSession:
        with self._lock:
            existing = self._sessions.get(runtime_key)
            if existing:
                return existing

            session = self._load_session(runtime_key=runtime_key, model_path=model_path, adapter_path=adapter_path)
            self._sessions[runtime_key] = session
            return session

    def _load_session(self, runtime_key: str, model_path: str, adapter_path: str | None = None) -> ModelSession:
        path = Path(model_path).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"模型路径不存在，无法启动真实推理：{path}")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "真实模型推理依赖尚未安装。请在 agent 环境安装："
                "pip install torch transformers accelerate peft sentencepiece safetensors"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        force_cpu = os.getenv("MODEL_RUNTIME_FORCE_CPU", "").lower() in {"1", "true", "yes"}
        if torch.cuda.is_available() and not force_cpu:
            load_kwargs["torch_dtype"] = "auto"
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(str(path), **load_kwargs)
        if "device_map" not in load_kwargs:
            model.to("cpu")

        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("加载 LoRA 微调模型需要安装 peft：pip install peft") from exc
            model = PeftModel.from_pretrained(model, adapter_path)

        model.eval()
        return ModelSession(runtime_key=runtime_key, tokenizer=tokenizer, model=model, torch_module=torch)

    def _base_runtime_key(self, base_model: dict) -> str:
        return f"base:{_model_identifier(base_model)}"

    def _adapter_runtime_key(self, base_model: dict, fine_tuned_model: dict) -> str:
        return f"adapter:{_model_identifier(base_model)}:{_model_identifier(fine_tuned_model)}"

    def _full_model_runtime_key(self, fine_tuned_model: dict) -> str:
        return f"full:{_model_identifier(fine_tuned_model)}"


def _model_identifier(model: dict) -> str:
    identifier = model.get("model_id") or model.get("runtime_id") or model.get("path") or model.get("id")
    if identifier:
        return str(identifier).strip()
    return str(Path(model.get("path", "")).resolve())
