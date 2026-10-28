from huggingface_hub import snapshot_download


# snapshot_download(
#     repo_id="Qwen/Qwen3-8B",
#     local_dir=r"./models/Qwen3-8B",
#     local_dir_use_symlinks=False
# )


import json
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_JSON = Path(__file__).resolve().parent / "models_to_test.json"


def download_models(json_path: str | Path = MODELS_JSON) -> dict[str, Path]:
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"Fichier JSON non trouvé : {json_file}")

    with json_file.open("r", encoding="utf-8") as f:
        models = json.load(f)
    if not isinstance(models, dict):
        raise ValueError("Le fichier JSON doit contenir un objet {nom: repo_id}.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    local_paths = {}

    for name, repo_id in models.items():
        path = MODELS_DIR / name
        local_paths[name] = path

        if (path / "config.json").exists():
            print(f"[SKIP] {name} déjà présent : {path}")
            continue

        print(f"[DOWNLOAD] {name} ({repo_id}) -> {path}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(path),
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"],
            ignore_patterns=["original/*"],
        )
        print(f"[OK] {name}\n")

    return local_paths


download_models()