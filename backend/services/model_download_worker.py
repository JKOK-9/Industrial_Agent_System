from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m backend.services.model_download_worker <payload.json>")
        return 2

    payload_path = Path(sys.argv[1]).resolve()
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    source = payload["source"]
    repo_id = payload["repo_id"]
    destination = Path(payload["destination"]).resolve()
    log_path = Path(payload["log_path"]).resolve()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)

    try:
        with log_path.open("a", encoding="utf-8") as log:
            if source == "huggingface":
                from huggingface_hub import snapshot_download

                log.write(f"Downloading {repo_id} from Hugging Face to {destination}\n")
                snapshot_path = snapshot_download(repo_id=repo_id, local_dir=str(destination))
                log.write(f"Snapshot path: {snapshot_path}\n")
            elif source == "modelscope":
                try:
                    from modelscope import snapshot_download as modelscope_snapshot_download
                except ImportError as exc:
                    raise RuntimeError("使用 ModelScope 下载前，请先安装依赖：pip install modelscope") from exc

                log.write(f"Downloading {repo_id} from ModelScope to {destination}\n")
                try:
                    snapshot_path = modelscope_snapshot_download(repo_id, local_dir=str(destination))
                except TypeError:
                    snapshot_path = modelscope_snapshot_download(repo_id, cache_dir=str(destination))
                log.write(f"Snapshot path: {snapshot_path}\n")
                _copy_snapshot_to_destination(Path(snapshot_path), destination)
            else:
                raise RuntimeError(f"不支持的模型来源：{source}")

            _ensure_download_has_files(destination)
        return 0
    except Exception as exc:  # noqa: BLE001
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"ERROR: {exc}\n")
        return 1


def _copy_snapshot_to_destination(snapshot_path: Path, destination: Path) -> None:
    if snapshot_path.resolve() == destination.resolve():
        return
    if not snapshot_path.exists():
        return
    has_destination_files = any(item.is_file() for item in destination.rglob("*"))
    if has_destination_files:
        return
    shutil.copytree(snapshot_path, destination, dirs_exist_ok=True)


def _ensure_download_has_files(destination: Path) -> None:
    if not destination.exists() or not any(item.is_file() for item in destination.rglob("*")):
        raise RuntimeError(f"模型下载目录为空：{destination}")


if __name__ == "__main__":
    raise SystemExit(main())
