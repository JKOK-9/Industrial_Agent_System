from __future__ import annotations

import json
import os
import threading
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MODEL_RUNTIME_MAX_NEW_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("MODEL_RUNTIME_TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.getenv("MODEL_RUNTIME_TOP_P", "0.9"))
DEFAULT_API_TIMEOUT_SECONDS = float(os.getenv("MODEL_API_TIMEOUT_SECONDS", "3600"))


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
        self._disposed = False

    def generate(self, prompt: str, options: GenerationOptions | None = None) -> str:
        options = options or GenerationOptions()
        with self._lock:
            if self._disposed or self.model is None or self.tokenizer is None:
                raise RuntimeError("模型会话已卸载，请重新发起调用以加载模型。")

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

            with torch.inference_mode():
                output_ids = model.generate(**inputs, **generation_kwargs)

            prompt_length = inputs["input_ids"].shape[-1]
            generated_ids = output_ids[0][prompt_length:]
            return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def dispose(self) -> None:
        with self._lock:
            if self._disposed:
                return
            torch = self.torch
            model = self.model
            tokenizer = self.tokenizer
            self.model = None
            self.tokenizer = None
            self._disposed = True

        del model
        del tokenizer
        gc.collect()
        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

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
        if base_model.get("source") == "api":
            return self._generate_api_base(base_model, prompt, options)

        runtime_key = self.runtime_key_for_base(base_model)
        session = self._get_or_load(runtime_key=runtime_key, model_path=base_model.get("path", ""))
        return session.generate(prompt, options)

    def generate_fine_tuned(self, base_model: dict, fine_tuned_model: dict, prompt: str, options: GenerationOptions | None = None) -> str:
        adapter_path = Path(fine_tuned_model.get("path", "")).resolve()
        if (adapter_path / "adapter_config.json").is_file():
            runtime_key = self.runtime_key_for_fine_tuned(base_model, fine_tuned_model)
            session = self._get_or_load(runtime_key=runtime_key, model_path=base_model.get("path", ""), adapter_path=str(adapter_path))
        else:
            runtime_key = self.runtime_key_for_fine_tuned(base_model, fine_tuned_model)
            session = self._get_or_load(runtime_key=runtime_key, model_path=str(adapter_path))
        return session.generate(prompt, options)

    def loaded_models(self) -> list[str]:
        with self._lock:
            return sorted(self._sessions)

    def runtime_key_for_base(self, base_model: dict) -> str:
        return self._base_runtime_key(base_model)

    def runtime_key_for_fine_tuned(self, base_model: dict, fine_tuned_model: dict) -> str:
        adapter_path = Path(fine_tuned_model.get("path", "")).resolve()
        if (adapter_path / "adapter_config.json").is_file():
            return self._adapter_runtime_key(base_model, fine_tuned_model)
        return self._full_model_runtime_key(fine_tuned_model)

    def unload_many(self, runtime_keys: set[str] | list[str] | tuple[str, ...]) -> list[str]:
        unloaded: list[str] = []
        for runtime_key in sorted(set(runtime_keys)):
            if self.unload(runtime_key):
                unloaded.append(runtime_key)
        return unloaded

    def unload(self, runtime_key: str) -> bool:
        with self._lock:
            session = self._sessions.pop(runtime_key, None)
        if not session:
            return False
        session.dispose()
        return True

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

    def _generate_api_base(self, base_model: dict, prompt: str, options: GenerationOptions | None = None) -> str:
        options = options or GenerationOptions()
        api_config = base_model.get("api_config") or {}
        provider = str(api_config.get("provider") or "openai_compatible")
        if provider != "openai_compatible":
            raise RuntimeError(f"暂不支持的 API 模型服务类型：{provider}")

        endpoint = _chat_completions_endpoint(str(api_config.get("base_url") or ""))
        api_model = str(api_config.get("model") or base_model.get("model_id") or "").strip()
        if not endpoint:
            raise RuntimeError("API 接入模型缺少 API 地址。")
        if not api_model:
            raise RuntimeError("API 接入模型缺少模型名称。")

        payload = {
            "model": api_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": options.max_new_tokens,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "stream": False,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = str(api_config.get("api_key") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib_request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib_request.urlopen(req, timeout=DEFAULT_API_TIMEOUT_SECONDS) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API 模型调用失败：HTTP {exc.code} {body[:1000]}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"API 模型调用失败：{exc.reason}") from exc

        return _extract_api_text(response_payload)

    def _base_runtime_key(self, base_model: dict) -> str:
        if base_model.get("source") == "api":
            api_config = base_model.get("api_config") or {}
            return f"api:{api_config.get('base_url')}:{api_config.get('model') or base_model.get('model_id')}"
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


def _chat_completions_endpoint(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    if not cleaned:
        return ""
    if "chat/completions" in cleaned:
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return f"{cleaned}/v1/chat/completions"


def _extract_api_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] or {}
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif isinstance(item, str):
                        parts.append(item)
                if parts:
                    return "\n".join(parts).strip()
        text = first.get("text")
        if isinstance(text, str):
            return text.strip()
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text.strip()
    raise RuntimeError("API 模型响应中未找到可用文本。")
