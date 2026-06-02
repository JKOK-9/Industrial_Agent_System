from __future__ import annotations

import os
from pathlib import Path


def load_local_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env.local"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_local_env()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = Path(os.getenv("LLM_PLATFORM_RUNTIME_DIR", PROJECT_ROOT / "runtime")).resolve()
BASE_MODELS_DIR = RUNTIME_DIR / "base_models"
DATASETS_DIR = RUNTIME_DIR / "datasets"
CONFIGS_DIR = RUNTIME_DIR / "configs"
OUTPUTS_DIR = RUNTIME_DIR / "outputs"
LOGS_DIR = RUNTIME_DIR / "logs"
RESOURCES_DIR = RUNTIME_DIR / "resources"
KNOWLEDGE_DIR = RESOURCES_DIR / "knowledge"
PROMPTS_DIR = RESOURCES_DIR / "prompts"
AGENTS_DIR = RUNTIME_DIR / "agents"
REGISTRY_PATH = RUNTIME_DIR / "registry.json"

LLAMAFACTORY_CLI = os.getenv("LLAMAFACTORY_CLI", "llamafactory-cli")
LLAMAFACTORY_WORKDIR = os.getenv("LLAMAFACTORY_WORKDIR")
GRAPH_LLM_API_KEY = os.getenv("GRAPH_LLM_API_KEY", "")
GRAPH_LLM_BASE_URL = os.getenv("GRAPH_LLM_BASE_URL", "")
GRAPH_LLM_MODEL = os.getenv("GRAPH_LLM_MODEL", "")


def ensure_runtime_dirs() -> None:
    for path in [
        RUNTIME_DIR,
        BASE_MODELS_DIR,
        DATASETS_DIR,
        CONFIGS_DIR,
        OUTPUTS_DIR,
        LOGS_DIR,
        RESOURCES_DIR,
        KNOWLEDGE_DIR,
        PROMPTS_DIR,
        AGENTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def is_path_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
