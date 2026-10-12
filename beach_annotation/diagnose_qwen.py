#!/usr/bin/env python3
"""Qwen-only visual diagnostic. Run beside the existing models/ directory.

Five selected regions x four independent prompts = 20 generations. No detector,
SAM, training labels, remote inference, or new model download is involved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
MODES = ("describe_exact", "describe_square", "classify_pair_no_examples", "classify_pair_original")
TITLES = ("A. Describe the exact rectangle", "B. Describe the pipeline square",
          "C. Classify the pair, no examples", "D. Classify the pair, original examples")
LABELS = {"can", "plastic_bottle", "glass", "metal_other", "paper_cardboard",
          "plastic_other", "other_trash", "uncertain", "not_trash"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def crop_window(box, width, height, margin, minimum):
    """Frozen copy of the tested pipeline's square crop policy."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = max(minimum, max(box[2] - box[0], box[3] - box[1]) * (1 + 2 * margin))
    return (max(0, math.floor(cx - side / 2)), max(0, math.floor(cy - side / 2)),
            min(width, math.ceil(cx + side / 2)), min(height, math.ceil(cy + side / 2)))


def prepare_views(image, box):
    exact = image.crop(box)
    tight = crop_window(box, image.width, image.height, .15, 24)
    context = crop_window(box, image.width, image.height, .50, 96)
    marked = image.crop(context)
    draw = ImageDraw.Draw(marked)
    draw.rectangle((max(0, box[0] - context[0] - 2), max(0, box[1] - context[1] - 2),
                    min(marked.width - 1, box[2] - context[0] + 2),
                    min(marked.height - 1, box[3] - context[1] + 2)), outline=(255, 0, 0), width=2)
    return {"exact": exact, "square": image.crop(tight), "context": marked}


def load_cases(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    result, seen = [], set()
    for case in manifest["cases"]:
        if not re.fullmatch(r"case_[0-9]+", case["id"]) or case["id"] in seen:
            raise ValueError("Case IDs must be distinct case_NN identifiers.")
        seen.add(case["id"])
        path = (manifest_path.parent / case["image"]).resolve()
        if not path.is_relative_to(manifest_path.parent) or not path.is_file():
            raise ValueError(f"Missing or unsafe case image: {case['image']}")
        if sha256(path) != case["source_sha256"]:
            raise ValueError(f"Photo bytes changed: {case['image']}")
        with Image.open(path) as opened:
            if opened.getexif().get(274, 1) not in (None, 1):
                raise ValueError("Case photos must use stored pixel orientation.")
            image = opened.convert("RGB")
        box = case["box_xyxy"]
        if (len(box) != 4 or not all(type(x) is int for x in box) or
                not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height)):
            raise ValueError(f"Invalid case box: {case['id']}")
        result.append((case, prepare_views(image, box)))
    if not result:
        raise ValueError("No diagnostic cases.")
    return manifest, result


def parse_label(answer):
    """Report format validity without retries, corrections or prose guessing."""
    try:
        text = answer.strip()
        if text.startswith("```") and text.endswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        obj = json.loads(text)
        if isinstance(obj, list) and len(obj) == 1:
            obj = obj[0]
        if not isinstance(obj, dict) or not isinstance(obj.get("label"), str) or obj["label"] not in LABELS:
            raise ValueError("Invalid label")
        return {"json_label_valid": True, "label": obj["label"]}
    except (ValueError, TypeError, IndexError):
        return {"json_label_valid": False, "label": None}


def reconstruct_pixels(flat, grids, processor):
    """Undo Qwen's pinned patch packing to inspect the actual processed pixels.

    Verified against Transformers 4.57.1 Qwen2VLImageProcessor._preprocess:
    reshape -> transpose(0,3,6,4,7,2,1,5,8) -> flatten. These images show
    resizing/normalization inputs, not the model's internal visual features.
    """
    p, t, m = processor.patch_size, processor.temporal_patch_size, processor.merge_size
    images, offset = [], 0
    for gt, gh, gw in grids:
        gt, gh, gw = int(gt), int(gh), int(gw)
        if gt != 1 or gh % m or gw % m:
            raise ValueError("Unexpected still-image grid layout.")
        count = gt * gh * gw
        block = flat[offset:offset + count]
        packed = block.reshape(gt, gh // m, gw // m, m, m, 3, t, p, p)
        frames = packed.transpose(0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(gt * t, 3, gh * p, gw * p)
        pixels = frames[0].transpose(1, 2, 0).copy()
        if processor.do_normalize:
            pixels = pixels * np.array(processor.image_std) + np.array(processor.image_mean)
        if processor.do_rescale:
            pixels = pixels / processor.rescale_factor
        images.append(Image.fromarray(np.clip(np.rint(pixels), 0, 255).astype(np.uint8)))
        offset += count
    if offset != len(flat):
        raise ValueError("Image grids and pixel rows disagree.")
    return images


class QwenRunner:
    def __init__(self, model_dir, device, dtype_name="auto"):
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if transformers.__version__ != "4.57.1":
            raise RuntimeError("Use the existing transformers==4.57.1 environment for a controlled comparison.")
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(f"Qwen model missing: {model_dir}. Use --model-dir with your existing Qwen folder.")
        if read_json(model_dir / "config.json").get("model_type") != "qwen2_5_vl":
            raise ValueError("Expected the Qwen2.5-VL model, not a text-only Qwen or another model.")
        self.torch, self.device = torch, torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment.")
        dtype = torch.float32
        if dtype_name != "auto":
            dtype = getattr(torch, dtype_name)
        elif self.device.type == "cuda":
            with torch.cuda.device(self.device):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Loading only Qwen from {model_dir} ({device}, {dtype})...", flush=True)
        self.processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True,
                                                       min_pixels=128 * 28 * 28, max_pixels=512 * 28 * 28)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir), local_files_only=True, use_safetensors=True,
            torch_dtype=dtype, attn_implementation="sdpa").to(self.device).eval()
        self.info = {"device": str(self.device), "dtype": str(dtype), "model_class": type(self.model).__name__,
                     "processor_class": type(self.processor).__name__,
                     "image_processor_class": type(self.processor.image_processor).__name__,
                     "image_processor_config": self.processor.image_processor.to_dict(),
                     "model_config_sha256": sha256(model_dir / "config.json"),
                     "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None}
        version_file = model_dir.parent / "model_versions.json"
        if version_file.is_file():
            self.info["download_manifest_classifier"] = read_json(version_file).get("classifier")

    def generate(self, images, prompt, trial_dir):
        # Every call has a fresh conversation. No filename, reviewer note,
        # previous caption or previous predicted label is included in messages.
        messages = [{"role": "user", "content": [{"type": "image"} for _ in images] +
                     [{"type": "text", "text": prompt}]}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        (trial_dir / "chat_template.txt").write_text(rendered, encoding="utf-8")
        inputs = self.processor(text=[rendered], images=images, padding=True, return_tensors="pt")
        if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
            raise RuntimeError("Processor did not produce image inputs.")
        flat = inputs["pixel_values"].detach().cpu().float().numpy()
        grids = inputs["image_grid_thw"].cpu().tolist()
        if len(grids) != len(images) or not np.isfinite(flat).all():
            raise RuntimeError("Wrong image count or non-finite processed pixels.")
        expected = sum(int(np.prod(grid)) // self.processor.image_processor.merge_size ** 2 for grid in grids)
        actual = int((inputs["input_ids"] == self.model.config.image_token_id).sum().item())
        if actual != expected or actual == 0:
            raise RuntimeError(f"Image token mismatch: got {actual}, expected {expected}.")
        metadata = {"image_grid_thw": grids, "image_tokens": actual, "expected_image_tokens": expected,
                    "input_tokens": int(inputs["input_ids"].shape[1]), "pixel_shape": list(flat.shape),
                    "pixel_min": float(flat.min()), "pixel_max": float(flat.max()),
                    "pixel_std": float(flat.std()), "pixel_sha256": hashlib.sha256(flat.tobytes()).hexdigest()}
        # A debug rendering failure should not prevent observing the raw model
        # answer; the warning remains prominent in processor.json/report.json.
        try:
            previews = reconstruct_pixels(flat, grids, self.processor.image_processor)
            for index, preview in enumerate(previews, 1):
                preview.save(trial_dir / f"processed_image_{index}.png")
            metadata["processed_previews"] = [f"processed_image_{i}.png" for i in range(1, len(previews) + 1)]
        except (ValueError, AttributeError) as exc:
            metadata["processed_preview_error"] = str(exc)
        write_json(trial_dir / "processor.json", metadata)
        started = time.perf_counter()
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        new_ids = ids[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        metadata.update({"seconds": round(time.perf_counter() - started, 3),
                         "generated_tokens": int(new_ids.shape[1]), "hit_token_limit": new_ids.shape[1] >= 256})
        return answer, metadata


def save_report(output, report):
    write_json(output / "report.json", report)
    escape = html.escape
    sections = []
    for case in report["cases"]:
        cid = case["id"]
        previews = ''.join(f'<figure><a href="inputs/{cid}/{key}.png"><img src="inputs/{cid}/{key}.png"></a>'
                           f'<figcaption>{title}</figcaption></figure>' for key, title in
                           (("exact", "Exact rectangle"), ("square", "Pipeline square"), ("context", "Marked context")))
        rows = []
        for mode, title in zip(MODES, TITLES):
            result = next((r for r in report["results"] if r["case_id"] == cid and r["mode"] == mode), None)
            answer = escape(result["answer"]) if result else "Not run."
            parsed = ""
            if result and "json_label_valid" in result:
                parsed = f'<p>Valid label JSON: {result["json_label_valid"]}; label: {escape(str(result["label"]))}</p>'
            links = ""
            if result:
                meta = result["processor"]
                links = " ".join(f'<a href="trials/{cid}/{mode}/{name}">Processed image {i}</a>'
                                 for i, name in enumerate(meta.get("processed_previews", []), 1))
                if meta.get("processed_preview_error"):
                    links += '<p>Processed preview error: ' + escape(meta["processed_preview_error"]) + '</p>'
            rows.append(f'<article><h3>{title}</h3><pre>{answer}</pre>{parsed}<p>{links}</p></article>')
        sections.append(f'<section><h2>{cid}</h2><p>{escape(case["reviewer_note"])} '
                        f'<strong>This note is never sent to Qwen.</strong></p><div class="images">{previews}</div>'
                        f'<div class="answers">{"".join(rows)}</div></section>')
    doc = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
    doc += '<title>Qwen visual diagnostic</title><style>body{font:16px system-ui,sans-serif;background:#f3f5f7;color:#172333;max-width:1200px;margin:32px auto;padding:0 20px}section{background:white;border:1px solid #d8e0e8;border-radius:12px;padding:22px;margin:24px 0}.images{display:flex;gap:18px;flex-wrap:wrap}figure{flex:1;min-width:200px;margin:0}img{width:100%;height:210px;object-fit:contain;background:#edf1f5}figcaption{margin:8px 0;color:#45566a}.answers{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}article{background:#f5f7fa;padding:14px;border-radius:8px}h3{font-size:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px system-ui}a{color:#1555aa}</style><body>'
    doc += f'<h1>Qwen visual diagnostic</h1><p>Status: <strong>{escape(report["status"])}</strong>. '
    doc += f'{len(report["results"])}/{report["planned_generations"]} generations completed.</p>'
    doc += '<p>A and B use the same neutral prompt with different crops. C and D use the same two images and differ only in prompt examples. All calls start fresh. Format validity is not a measure of recognition accuracy.</p>'
    if report.get("error"):
        doc += '<p>Error: ' + escape(report["error"]) + '</p>'
    (output / "report.html").write_text(doc + ''.join(sections) + '</body></html>', encoding="utf-8")


def pack_results(output):
    destination = output.parent / (output.name + ".zip")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output.parent).as_posix())
    return destination


def run(args, runner_factory=QwenRunner):
    output = args.output.expanduser().resolve()
    if (output.exists() and any(output.iterdir())) or (output.parent / (output.name + ".zip")).exists():
        raise ValueError("Output already exists. Choose a new --output directory to preserve previous results.")
    manifest, cases = load_cases(args.cases)
    prompts = read_json(args.cases.parent / "prompts.json")
    if set(prompts) != set(MODES) or prompts[MODES[0]] != prompts[MODES[1]]:
        raise ValueError("Unexpected diagnostic prompts or mismatched description prompts.")
    if prompts[MODES[2]] != prompts[MODES[3]].split("Format examples only (choose based on the actual images): ")[0].rstrip():
        raise ValueError("The classification prompt comparison must differ only by examples.")
    report = {"status": "PREPARED", "created_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": sha256(__file__), "reference_annotate_sha256": manifest["reference_annotate_sha256"],
              "cases": [case for case, _ in cases], "prompts": prompts, "results": [],
              "planned_generations": len(cases) * len(MODES), "runtime_versions": {}}
    for package in ("torch", "torchvision", "transformers", "numpy", "Pillow"):
        try:
            report["runtime_versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["runtime_versions"][package] = "not installed"
    for case, views in cases:
        directory = output / "inputs" / case["id"]
        directory.mkdir(parents=True, exist_ok=True)
        for key, view in views.items():
            view.save(directory / (key + ".png"))
    save_report(output, report)
    if args.dry_run:
        print(f"Prepared {len(cases)} cases; no models loaded. Inspect {output / 'report.html'}", flush=True)
        return report
    failed = None
    try:
        runner = runner_factory(args.model_dir.expanduser().resolve(), args.device, args.dtype)
        report["model"] = runner.info
        for case, views in cases:
            for mode in MODES:
                selected = [views["exact"]] if mode == "describe_exact" else (
                    [views["square"]] if mode == "describe_square" else [views["square"], views["context"]])
                trial = output / "trials" / case["id"] / mode
                trial.mkdir(parents=True)
                number = len(report["results"]) + 1
                print(f"[{number}/{report['planned_generations']}] {case['id']} / {mode}", flush=True)
                answer, metadata = runner.generate(selected, prompts[mode], trial)
                result = {"case_id": case["id"], "mode": mode, "answer": answer, "processor": metadata}
                if mode.startswith("classify"):
                    result.update(parse_label(answer))
                report["results"].append(result)
                report["status"] = "RUNNING"
                save_report(output, report)
                print("  " + answer.replace("\n", " ")[:400], flush=True)
        report["status"] = "COMPLETE_REQUIRES_REVIEW"
    except (Exception, KeyboardInterrupt) as exc:
        failed = exc
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
    finally:
        save_report(output, report)
        archive = pack_results(output)
        print(f"Saved report: {output / 'report.html'}\nUpload this ZIP: {archive}", flush=True)
    if failed is not None:
        raise RuntimeError(f"Diagnostic stopped; raw results and error are saved in {archive}") from failed
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "qwen_diagnostic")
    parser.add_argument("--cases", type=Path, default=ROOT / "diagnostic_cases" / "cases.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Prepare case images/report without loading Qwen.")
    run(parser.parse_args())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
