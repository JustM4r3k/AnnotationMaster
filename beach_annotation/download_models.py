#!/usr/bin/env python3
"""Download the official Hugging Face model repositories for local inference.

No paid inference API is used. Internet access is needed for this download.
Afterward, annotate.py uses only local model files.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-qwen", action="store_true", help="Also download the optional 7B classifier.")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.models_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "model_versions.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    keys = ["detector", "segmenter"] + (["classifier"] if args.with_qwen else [])
    for key in keys:
        spec = config["models"][key]
        destination = (root / spec["folder"]).resolve()
        if root not in destination.parents:
            raise ValueError("Model folders must be relative subdirectories of --models-dir.")
        old = manifest.get(key)
        if old and old["repo_id"] != spec["repo_id"]:
            raise ValueError(f"The {key} model changed. Use a new --models-dir to avoid mixing model files.")
        # Resolve and record the exact revision on the first run. Subsequent
        # downloads reuse it rather than silently switching to newer weights.
        revision = old["revision"] if old else HfApi().model_info(spec["repo_id"]).sha
        print(f"Downloading {spec['repo_id']} @ {revision} into {destination}", flush=True)
        snapshot_download(
            repo_id=spec["repo_id"], revision=revision, local_dir=str(destination),
            # Include converted Transformers weights and tokenizer/processor
            # files; avoid also downloading original-format duplicate weights.
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "README.md", "LICENSE*"],
        )
        if not (destination / "config.json").is_file() or not list(destination.glob("*.safetensors")):
            raise RuntimeError(f"Incomplete model download at {destination}")
        manifest[key] = {"repo_id": spec["repo_id"], "revision": revision, "folder": spec["folder"]}
        temp = manifest_path.with_suffix(".tmp")
        temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temp.replace(manifest_path)
    print("Done. Model revisions recorded in models/model_versions.json.")


if __name__ == "__main__":
    main()
