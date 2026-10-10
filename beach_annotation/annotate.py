#!/usr/bin/env python3
"""Draft beach-litter instance annotations: Grounding DINO -> SAM 2.1 -> COCO.

Optional Qwen2.5-VL suggestions and existing COCO masks are supported.
Run `python annotate.py --help` and read README.md before the first run.
The output is for human review in CVAT, not automatically approved training data.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import zipfile
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = (
    "can", "plastic_bottle", "glass", "metal_other",
    "paper_cardboard", "plastic_other", "other_trash",
)
CATEGORIES = [
    {"id": i + 1, "name": name, "supercategory": "litter"}
    for i, name in enumerate(CLASS_NAMES)
]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
COLORS = [(245, 91, 76), (47, 173, 235), (129, 101, 239), (220, 164, 38),
          (207, 115, 56), (56, 189, 132), (226, 98, 185)]


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path, value):
    """Replace only a complete JSON file; interruption cannot leave half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    temp.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_name(name):
    """Use COCO's relative file names; never silently flatten subdirectories."""
    name = str(name).replace("\\", "/")
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or not p.parts or ":" in p.parts[0]:
        raise ValueError(f"Expected a relative image filename, got {name!r}")
    return p.as_posix()


def mask_api():
    try:
        from pycocotools import mask
        return mask
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt first (pycocotools is missing).") from exc


def encode_mask(mask):
    """COCO RLE preserves holes and disconnected pieces of ONE object."""
    rle = mask_api().encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def segmentation_rle(segmentation, height, width):
    api = mask_api()
    if isinstance(segmentation, list):
        if not segmentation:
            return None
        return api.merge(api.frPyObjects(segmentation, height, width))
    if isinstance(segmentation, dict):
        if list(segmentation.get("size", [])) != [height, width]:
            raise ValueError("RLE dimensions do not match the original image.")
        if isinstance(segmentation.get("counts"), list):
            return api.frPyObjects(segmentation, height, width)
        return segmentation
    raise ValueError("Segmentation must be a COCO polygon list or RLE dictionary.")


def box_iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-12)


def xywh(box):
    return [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]


def xyxy(box):
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def tile_windows(width, height, tile_size, overlap):
    """Always scan the full image, then overlapping tiles if the image is larger."""
    yield (0, 0, width, height)
    if tile_size == 0 or (width <= tile_size and height <= tile_size):
        return

    def starts(length):
        last = max(0, length - tile_size)
        step = max(1, round(tile_size * (1 - overlap)))
        return sorted(set(list(range(0, last + 1, step)) + [last]))

    for y in starts(height):
        for x in starts(width):
            yield (x, y, min(width, x + tile_size), min(height, y + tile_size))


def crop_window(box, width, height, margin, minimum):
    """Context around a box, clipped to the image; all coordinates stay in pixels."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = max(minimum, max(box[2] - box[0], box[3] - box[1]) * (1 + 2 * margin))
    return (max(0, math.floor(cx - side / 2)), max(0, math.floor(cy - side / 2)),
            min(width, math.ceil(cx + side / 2)), min(height, math.ceil(cy + side / 2)))


def merge_candidates(candidates, threshold):
    """Remove near-identical boxes; keep competing class suggestions as metadata."""
    kept = []
    for item in sorted(candidates, key=lambda x: x["score"], reverse=True):
        duplicate = next((old for old in kept if box_iou(old["box"], item["box"]) >= threshold), None)
        if duplicate is None:
            item = copy.deepcopy(item)
            item["candidate_classes"] = [item["category_id"]]
            kept.append(item)
        else:
            duplicate["candidate_classes"] = sorted(set(
                duplicate["candidate_classes"] + [item["category_id"]]))
    return kept


def validate_config(config):
    prompts = config["prompts"]
    if set(prompts) != set(CLASS_NAMES):
        raise ValueError("config.json must define prompts for exactly the seven classes.")
    for name, phrases in prompts.items():
        if not phrases or not all(isinstance(p, str) and p.strip() for p in phrases):
            raise ValueError(f"Empty or invalid prompts for {name}")
    det = config["detection"]
    if det["tile_size"] < 0 or not 0 <= det["tile_overlap"] < 1:
        raise ValueError("Use tile_size >= 0 and 0 <= tile_overlap < 1.")
    for key in ("box_threshold", "text_threshold", "box_merge_iou", "mask_merge_iou"):
        if not 0 < det[key] <= 1:
            raise ValueError(f"Invalid detection.{key}")
    sam = config["segmentation"]
    if sam["context_margin"] < 0 or sam["minimum_crop_size"] < 1:
        raise ValueError("Invalid SAM crop settings.")


class ExistingAnnotations:
    """Import source COCO without changing its segmentation geometry.

    A category map uses original NAMES, never assumed original numeric IDs.
    Exact relative filename matching is deliberate: accidental cross-image masks
    are much worse than an actionable error asking for the correct image root.
    """
    def __init__(self, path=None, category_map=None):
        self.images, self.annotations, self.category_map = {}, {}, {}
        self.path = str(path) if path else None
        self.info, self.licenses = {}, []
        if path is None:
            return
        data = read_json(path)
        self.info, self.licenses = data.get("info", {}), data.get("licenses", [])
        name_map = read_json(category_map) if category_map else {n: n for n in CLASS_NAMES}
        source_names = {}
        for category in data["categories"]:
            original = category["name"]
            target = name_map.get(original)
            if target not in CLASS_NAMES:
                raise ValueError(f"No valid category mapping for {original!r}. Supply --category-map.")
            if category["id"] in source_names:
                raise ValueError("Duplicate source category ID.")
            source_names[category["id"]] = original
            self.category_map[category["id"]] = CLASS_NAMES.index(target) + 1
        self.source_names = source_names
        source_ids = set()
        for image in data["images"]:
            name = normalized_name(image["file_name"])
            if name in self.images or image["id"] in source_ids:
                raise ValueError("Duplicate source image filename or ID.")
            self.images[name] = image
            source_ids.add(image["id"])
            self.annotations[image["id"]] = []
        annotation_ids = set()
        for ann in data["annotations"]:
            if ann["id"] in annotation_ids:
                raise ValueError("Duplicate source annotation ID.")
            annotation_ids.add(ann["id"])
            if ann["image_id"] not in source_ids or ann["category_id"] not in self.category_map:
                raise ValueError("Source annotation references an unknown image or class.")
            self.annotations[ann["image_id"]].append(ann)

    def for_image(self, relative_name, width, height):
        if self.path is None:
            return []
        if relative_name not in self.images:
            raise ValueError(
                f"{relative_name!r} is not in the existing COCO file. Set --images to the "
                "directory relative to which COCO file_name paths are written.")
        source_image = self.images[relative_name]
        if (source_image["width"], source_image["height"]) != (width, height):
            raise ValueError(f"Source COCO/image dimensions disagree: {relative_name}")
        result = []
        for original in self.annotations[source_image["id"]]:
            ann = copy.deepcopy(original)
            attrs = ann.setdefault("attributes", {})
            attrs.update({"source": "existing_coco", "review_status": "needs_review",
                          "original_annotation_id": str(original["id"]),
                          "original_image_id": str(original["image_id"]),
                          "original_category": self.source_names[original["category_id"]]})
            ann["category_id"] = self.category_map[original["category_id"]]
            # Preserve polygons/RLE, bbox, area, and iscrowd from the source.
            ann.pop("id", None)
            ann.pop("image_id", None)
            result.append(ann)
        return result


def parse_qwen_response(text):
    """Malformed or out-of-taxonomy output must never silently create a label."""
    try:
        start = text.index("{")
        result, _ = json.JSONDecoder().raw_decode(text[start:])
        status = result["status"]
        label = result.get("class_name")
        if status not in ("suggestion", "uncertain", "not_trash"):
            raise ValueError("Unknown status")
        if status == "suggestion" and label not in CLASS_NAMES:
            raise ValueError("Unknown class")
        return {"status": status, "class_name": label if status == "suggestion" else None,
                "reason": str(result.get("reason", ""))[:600]}
    except (ValueError, KeyError, TypeError):
        return {"status": "parse_error", "class_name": None, "reason": text[:600]}


class Models:
    """The GPU-specific code lives here; export and tests do not load models."""
    def __init__(self, config, model_root, device, use_qwen=False):
        import torch
        import transformers
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

        if transformers.__version__ != "4.57.1":
            raise RuntimeError("Use transformers==4.57.1 from requirements.txt; this script pins its APIs.")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Check nvidia-smi and your PyTorch GPU installation.")
        self.torch, self.device, self.config = torch, device, config

        def location(key):
            path = model_root / config["models"][key]["folder"]
            if not (path / "config.json").is_file():
                raise FileNotFoundError(f"Model missing at {path}. Run download_models.py first.")
            return str(path)

        print("Loading Grounding DINO and SAM 2.1...", flush=True)
        self.det_processor = AutoProcessor.from_pretrained(location("detector"), local_files_only=True)
        # Built-in PyTorch operations avoid compiling Grounding DINO CUDA extensions.
        self.det_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            location("detector"), local_files_only=True, use_safetensors=True,
            disable_custom_kernels=True).to(device).eval()
        self.sam_processor = Sam2Processor.from_pretrained(location("segmenter"), local_files_only=True)
        self.sam_model = Sam2Model.from_pretrained(
            location("segmenter"), local_files_only=True, use_safetensors=True).to(device).eval()
        self.qwen = None
        if use_qwen:
            from transformers import Qwen2_5_VLForConditionalGeneration
            dtype = torch.float32
            if device.startswith("cuda"):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print("Loading optional Qwen2.5-VL-7B...", flush=True)
            self.qwen_processor = AutoProcessor.from_pretrained(
                location("classifier"), local_files_only=True,
                min_pixels=128 * 28 * 28, max_pixels=512 * 28 * 28)
            self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                location("classifier"), local_files_only=True, use_safetensors=True,
                torch_dtype=dtype, attn_implementation="sdpa").to(device).eval()

    def detect(self, image, classes):
        settings = self.config["detection"]
        candidates = []
        for window in tile_windows(image.width, image.height, settings["tile_size"], settings["tile_overlap"]):
            crop = image.crop(window)
            # Separate prompt groups make the target-class assignment explicit.
            # The returned phrases can be incomplete, so we retain them as evidence,
            # not as dictionary keys on which class conversion depends.
            for name in classes:
                prompt = ". ".join(p.strip().lower().rstrip(".") for p in self.config["prompts"][name]) + "."
                inputs = self.det_processor(images=crop, text=prompt, return_tensors="pt").to(self.device)
                with self.torch.inference_mode():
                    outputs = self.det_model(**inputs)
                # This pinned Transformers release uses `threshold`, not the older
                # `box_threshold` keyword visible in some online examples.
                detected = self.det_processor.post_process_grounded_object_detection(
                    outputs, input_ids=inputs["input_ids"], threshold=settings["box_threshold"],
                    text_threshold=settings["text_threshold"], target_sizes=[(crop.height, crop.width)])[0]
                for box, score, phrase in zip(detected["boxes"].cpu().tolist(),
                                               detected["scores"].cpu().tolist(), detected["text_labels"]):
                    x0, y0, x1, y1 = box
                    local = [max(0., x0), max(0., y0), min(float(crop.width), x1), min(float(crop.height), y1)]
                    if not all(math.isfinite(v) for v in box + [score]) or local[2] <= local[0] or local[3] <= local[1]:
                        continue
                    touches_edge = window != (0, 0, image.width, image.height) and (
                        (window[0] > 0 and local[0] < 2) or (window[1] > 0 and local[1] < 2) or
                        (window[2] < image.width and local[2] > crop.width - 2) or
                        (window[3] < image.height and local[3] > crop.height - 2))
                    candidates.append({"box": [local[0] + window[0], local[1] + window[1],
                                                local[2] + window[0], local[3] + window[1]],
                                       "score": float(score), "phrase": phrase,
                                       "category_id": CLASS_NAMES.index(name) + 1,
                                       "tile_edge": touches_edge})
        return merge_candidates(candidates, settings["box_merge_iou"])

    def segment(self, image, box):
        cfg = self.config["segmentation"]
        window = crop_window(box, image.width, image.height, cfg["context_margin"], cfg["minimum_crop_size"])
        crop = image.crop(window)
        local_box = [box[0] - window[0], box[1] - window[1], box[2] - window[0], box[3] - window[1]]
        inputs = self.sam_processor(images=crop, input_boxes=[[local_box]], return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            result = self.sam_model(**inputs, multimask_output=False)
        local_mask = self.sam_processor.post_process_masks(
            result.pred_masks.cpu(), inputs["original_sizes"].cpu())[0][0, 0].numpy().astype(bool)
        if local_mask.shape != (crop.height, crop.width):
            raise RuntimeError(f"Unexpected SAM mask shape: {local_mask.shape}")
        # Paste into ORIGINAL image coordinates before exporting; CVAT sees the
        # original image, not the crop used internally by SAM.
        mask = np.zeros((image.height, image.width), dtype=bool)
        mask[window[1]:window[3], window[0]:window[2]] = local_mask
        quality = float(result.iou_scores.flatten()[0].cpu())
        return mask, quality

    def classify(self, image, box):
        if self.qwen is None:
            return None
        window = crop_window(box, image.width, image.height, 0.20, 96)
        crop = image.crop(window)
        target_box = [round(box[0] - window[0]), round(box[1] - window[1]),
                      round(box[2] - window[0]), round(box[3] - window[1])]
        instructions = (
            f"Classify the target object at pixel box {target_box} in this {crop.width}x{crop.height} image. "
            "Consider object type AND material. Use only this taxonomy: "
            "can=metal drink/food/aerosol cans, including crushed cans; "
            "plastic_bottle=recognisable plastic bottles; glass=glass bottles/jars/fragments; "
            "metal_other=metal excluding cans; paper_cardboard=paper/cardboard/paper cups; "
            "plastic_other=plastic excluding bottles, including bags/wrappers/foam; "
            "other_trash=cigarette butts/filters, rubber, textiles, other manufactured litter outside those classes. "
            "Shells, stones, seaweed and driftwood are not target litter. "
            "If material or object identity cannot be determined, return uncertain. "
            "Do not use other_trash as a synonym for uncertain. "
            'Return JSON only: {"status":"suggestion|uncertain|not_trash",'
            '"class_name":"one of the seven class names, or null", "reason":"short explanation"}.'
        )
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instructions}]}]
        text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.qwen_processor(text=[text], images=[crop], padding=True, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            ids = self.qwen.generate(**inputs, max_new_tokens=200, do_sample=False)
        answer = self.qwen_processor.batch_decode(
            ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return parse_qwen_response(answer)


def draft_annotation(image, candidate, models, config):
    mask, quality = models.segment(image, candidate["box"])
    attrs = {"source": "ai_proposal", "review_status": "needs_review",
             "detector_class": CLASS_NAMES[candidate["category_id"] - 1],
             "detector_phrase": candidate["phrase"], "detector_score": round(candidate["score"], 6),
             "sam_quality": round(quality, 6),
             "candidate_classes": ",".join(CLASS_NAMES[i - 1] for i in candidate["candidate_classes"])}
    flags = []
    if len(candidate["candidate_classes"]) > 1:
        flags.append("class_disagreement")
    if candidate.get("tile_edge"):
        flags.append("tile_edge_check_whole_object")
    category = candidate["category_id"]
    # Qwen is a second opinion. Uncertain/not_trash responses are retained for a
    # reviewer, with the detector's provisional class, rather than erased.
    suggestion = models.classify(image, candidate["box"])
    if suggestion:
        attrs.update({"qwen_status": suggestion["status"], "qwen_reason": suggestion["reason"]})
        if suggestion["status"] == "suggestion":
            category = CLASS_NAMES.index(suggestion["class_name"]) + 1
            if category != candidate["category_id"]:
                flags.append("qwen_detector_disagreement")
        else:
            flags.append("qwen_" + suggestion["status"])
    area = int(mask.sum())
    if area:
        rle = encode_mask(mask)
        bbox = mask_api().toBbox(rle).astype(float).tolist()
        if area < config["segmentation"]["small_mask_pixels"]:
            flags.append("tiny_mask")
    else:
        # A failed mask must stay visible to the reviewer. CVAT imports empty
        # segmentation as a box that must be replaced by a manually drawn mask.
        rle, bbox = [], xywh(candidate["box"])
        area = bbox[2] * bbox[3]
        flags.append("missing_mask_draw_in_cvat")
    if quality < config["segmentation"]["review_quality_below"]:
        flags.append("low_sam_quality")
    attrs["flags"] = ";".join(flags)
    return {"category_id": category, "segmentation": rle, "bbox": bbox, "area": area,
            "iscrowd": 0, "attributes": attrs}


def merge_masks(proposals, threshold, height, width):
    """Merge nearly identical AI masks, recording alternate classes for review."""
    kept, rles = [], []
    for ann in proposals:
        rle = segmentation_rle(ann["segmentation"], height, width)
        duplicate = None
        if rle is not None:
            for i, old_rle in enumerate(rles):
                if old_rle is not None and box_iou(xyxy(ann["bbox"]), xyxy(kept[i]["bbox"])) > 0:
                    if float(mask_api().iou([rle], [old_rle], [0])[0, 0]) >= threshold:
                        duplicate = kept[i]
                        break
        if duplicate is None:
            kept.append(ann)
            rles.append(rle)
        else:
            old, new = duplicate["attributes"], ann["attributes"]
            old["candidate_classes"] = ",".join(sorted(set(
                old["candidate_classes"].split(",") + new["candidate_classes"].split(",") +
                [CLASS_NAMES[ann["category_id"] - 1], CLASS_NAMES[duplicate["category_id"] - 1]])))
            flags = set(filter(None, (old["flags"] + ";" + new["flags"]).split(";")))
            if len(old["candidate_classes"].split(",")) > 1:
                flags.add("class_disagreement")
            old["flags"] = ";".join(sorted(flags))
            old["merged_proposals"] = int(old.get("merged_proposals", 0)) + 1
    return kept


def flag_existing_overlaps(existing, proposals, height, width):
    """Keep existing human masks unchanged, including in classification conflicts."""
    for proposal in proposals:
        conflicts = []
        for old in existing:
            if box_iou(xyxy(old["bbox"]), xyxy(proposal["bbox"])) >= 0.25:
                conflicts.append(old["attributes"]["original_annotation_id"])
        if conflicts:
            attrs = proposal["attributes"]
            attrs["overlaps_existing_ids"] = ",".join(conflicts)
            attrs["flags"] = ";".join(filter(None, [attrs["flags"], "overlap_existing_check_duplicate"]))


def validate_coco(data):
    """Catch corrupted geometry or IDs before delivering an import to CVAT."""
    api = mask_api()
    if {c["id"]: c["name"] for c in data["categories"]} != {c["id"]: c["name"] for c in CATEGORIES}:
        raise ValueError("Expected the seven canonical COCO categories with IDs 1–7.")
    images, names = {}, set()
    for im in data["images"]:
        if im["id"] in images or im["file_name"] in names:
            raise ValueError("Duplicate output image ID or filename.")
        normalized_name(im["file_name"])
        if im["width"] <= 0 or im["height"] <= 0:
            raise ValueError("Invalid image dimensions.")
        images[im["id"]] = im
        names.add(im["file_name"])
    seen, missing_masks = set(), 0
    for ann in data["annotations"]:
        if ann["id"] in seen or ann["image_id"] not in images or ann["category_id"] not in range(1, 8):
            raise ValueError("Duplicate annotation ID or broken image/category reference.")
        seen.add(ann["id"])
        im = images[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        if not all(math.isfinite(float(v)) for v in (x, y, w, h, ann["area"])):
            raise ValueError("Non-finite annotation geometry.")
        if x < -1e-3 or y < -1e-3 or w <= 0 or h <= 0 or x + w > im["width"] + 1e-3 or y + h > im["height"] + 1e-3:
            raise ValueError(f"Box outside image bounds: annotation {ann['id']}")
        rle = segmentation_rle(ann["segmentation"], im["height"], im["width"])
        if rle is None:
            missing_masks += 1
            continue
        actual_area = float(api.area(rle))
        if actual_area <= 0:
            raise ValueError("An annotation has an empty mask.")
        # Original polygon areas may have been calculated continuously rather
        # than rasterized. Do not rewrite the source's geometry or reject for
        # that small difference. Newly created RLE masks must agree exactly.
        if ann.get("attributes", {}).get("source") == "ai_proposal":
            if abs(actual_area - ann["area"]) > 0.01:
                raise ValueError("Generated RLE mask area disagrees with COCO area.")
            if not np.allclose(api.toBbox(rle), ann["bbox"], atol=0.01):
                raise ValueError("Generated RLE mask box disagrees with COCO bbox.")
    return {"images": len(images), "annotations": len(seen), "boxes_missing_masks": missing_masks}


def export_records(records, output, source=None, pending=None):
    data = {"info": {"description": "DRAFT beach-litter annotations. All images require full human review.",
                     "date_created": datetime.now(timezone.utc).isoformat(),
                     "review_status": "needs_review"},
            "licenses": source.licenses if source else [], "images": [], "annotations": [],
            "categories": copy.deepcopy(CATEGORIES)}
    rows = []
    for image_id, record in enumerate(sorted(records, key=lambda r: r["file_name"]), 1):
        image_entry = {"id": image_id, "file_name": record["file_name"], "width": record["width"],
                       "height": record["height"], "review_status": "needs_review"}
        if source and record["file_name"] in source.images:
            original_image = source.images[record["file_name"]]
            for key in ("license", "coco_url", "flickr_url", "date_captured"):
                if key in original_image:
                    image_entry[key] = original_image[key]
        data["images"].append(image_entry)
        if not record["annotations"]:
            rows.append([record["file_name"], "", "", "needs_review", "no_annotations_not_a_verified_negative"])
        for original in record["annotations"]:
            ann = copy.deepcopy(original)
            ann["id"], ann["image_id"] = len(data["annotations"]) + 1, image_id
            data["annotations"].append(ann)
            rows.append([record["file_name"], ann["id"], CLASS_NAMES[ann["category_id"] - 1],
                         "needs_review", ann.get("attributes", {}).get("flags", "")])
    if source and source.info:
        data["info"]["original_dataset_info"] = source.info
    summary = validate_coco(data)
    summary.update({"status": "DRAFT", "pending_images": pending or [],
                    "all_images_require_review": True})
    write_json(output / "annotations_draft.json", data)
    write_json(output / "summary.json", summary)
    with (output / "review.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "annotation_id", "provisional_class", "review_status", "flags"])
        writer.writerows(rows)
    return summary


def save_preview(image, annotations, path):
    overlay = np.array(image, dtype=np.float32)
    for ann in annotations:
        rle = segmentation_rle(ann["segmentation"], image.height, image.width)
        if rle is not None:
            selected = mask_api().decode(rle).astype(bool)
            color = np.array(COLORS[ann["category_id"] - 1])
            overlay[selected] = 0.60 * overlay[selected] + 0.40 * color
    preview = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(preview)
    for ann in annotations:
        color = COLORS[ann["category_id"] - 1]
        draw.rectangle(xyxy(ann["bbox"]), outline=color, width=2)
        draw.text((ann["bbox"][0], max(0, ann["bbox"][1] - 12)), CLASS_NAMES[ann["category_id"] - 1], fill=color)
    draw.rectangle((0, 0, 295, 20), fill=(20, 20, 20))
    draw.text((5, 4), "DRAFT - check all objects and boundaries", fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(path)


def pack_images(records, image_root, destination):
    """A task upload containing exactly the processed photos, with matching names."""
    temporary = destination.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for record in records:
            name = normalized_name(record["file_name"])
            archive.write(image_root / name, arcname=name)
    temporary.replace(destination)


def load_image(path):
    with Image.open(path) as opened:
        # Existing COCO is in stored pixel coordinates. Automatic EXIF rotation
        # here would silently move masks; ask the caller to normalize first.
        if opened.getexif().get(274, 1) not in (None, 1):
            raise ValueError(f"Nontrivial EXIF orientation in {path.name}. Normalize the image AND existing masks first.")
        return opened.convert("RGB")


def run(args):
    config = read_json(args.config)
    validate_config(config)
    image_root = args.images.expanduser().resolve()
    if not image_root.is_dir():
        raise ValueError(f"Image folder not found: {image_root}")
    files = sorted((p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS),
                   key=lambda p: p.relative_to(image_root).as_posix())
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise ValueError("No photographs found in --images.")
    names = [normalized_name(p.relative_to(image_root).as_posix()) for p in files]
    source = ExistingAnnotations(args.existing_coco, args.category_map)
    if args.existing_coco:
        missing = [name for name in names if name not in source.images]
        if missing:
            raise ValueError(f"Image paths do not match existing COCO (first: {missing[0]}). See README filename guidance.")
    if args.category_map and not args.existing_coco:
        raise ValueError("--category-map requires --existing-coco.")
    classes = args.classes.split(",") if args.classes else list(CLASS_NAMES)
    if not classes or any(name not in CLASS_NAMES for name in classes) or len(set(classes)) != len(classes):
        raise ValueError("--classes must be comma-separated canonical class names, without spaces or duplicates.")
    if args.dry_run:
        for path, name in zip(files, names):
            im = load_image(path)
            anns = source.for_image(name, im.width, im.height)
            # Also validates imported source geometry, without loading any model.
            data = {"images": [{"id": 1, "file_name": name, "width": im.width, "height": im.height}],
                    "categories": CATEGORIES, "annotations": [dict(a, id=i + 1, image_id=1) for i, a in enumerate(anns)]}
            validate_coco(data)
        print(f"Dry run passed for {len(files)} photographs. No models loaded and no annotations written.")
        return
    output = args.output.expanduser().resolve()
    if output == image_root or image_root in output.parents:
        raise ValueError("Put --output outside --images so previews are never re-annotated as inputs.")
    model_root = args.models_dir.expanduser().resolve()
    versions_path = model_root / "model_versions.json"
    if not versions_path.is_file():
        raise ValueError("Run download_models.py first to download models and record their revisions.")
    versions = read_json(versions_path)
    required = ["detector", "segmenter"] + (["classifier"] if args.qwen else [])
    for key in required:
        if (key not in versions or versions[key]["repo_id"] != config["models"][key]["repo_id"]
                or versions[key]["folder"] != config["models"][key]["folder"]):
            raise ValueError(f"Downloaded {key} does not match config.json. Run download_models.py with this config.")
    signature = {"config": config, "images_directory": str(image_root), "files": names,
                 "existing_coco_sha256": file_hash(args.existing_coco) if args.existing_coco else None,
                 "category_map_sha256": file_hash(args.category_map) if args.category_map else None,
                 "qwen": args.qwen, "classes": classes, "device": args.device,
                 "model_versions": {k: versions[k] for k in required}, "script_sha256": file_hash(__file__)}
    run_file = output / "run.json"
    if run_file.exists():
        if not args.resume:
            raise ValueError("Output already contains a run. Use --resume or choose a new output folder.")
        if read_json(run_file) != signature:
            raise ValueError("Resume inputs/settings/models changed. Use a NEW output directory.")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is nonempty and is not a resumable run.")
        output.mkdir(parents=True, exist_ok=True)
        write_json(run_file, signature)
    versions_runtime = {}
    for package in ("torch", "torchvision", "transformers", "numpy", "Pillow", "pycocotools"):
        try:
            versions_runtime[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions_runtime[package] = "not installed"
    write_json(output / "runtime_versions.json", versions_runtime)
    models, records, failed = None, [], None
    try:
        for index, (path, name) in enumerate(zip(files, names), 1):
            print(f"[{index}/{len(files)}] {name}", flush=True)
            checkpoint_key = hashlib.sha256(name.encode()).hexdigest()
            checkpoint = output / "checkpoints" / (checkpoint_key + ".json")
            current_hash = file_hash(path)
            if checkpoint.exists():
                record = read_json(checkpoint)
                if record["source_sha256"] != current_hash or record["file_name"] != name:
                    raise ValueError(f"Image changed since its checkpoint: {name}. Use a new output folder.")
                records.append(record)
                print("  Reusing completed checkpoint.", flush=True)
                continue
            image = load_image(path)
            existing = source.for_image(name, image.width, image.height)
            if models is None:
                models = Models(config, model_root, args.device, args.qwen)
            candidates = models.detect(image, classes)
            print(f"  {len(candidates)} candidate objects; creating masks...", flush=True)
            proposals = [draft_annotation(image, item, models, config) for item in candidates]
            proposals = merge_masks(proposals, config["detection"]["mask_merge_iou"], image.height, image.width)
            flag_existing_overlaps(existing, proposals, image.height, image.width)
            record = {"file_name": name, "width": image.width, "height": image.height,
                      "source_sha256": current_hash, "review_status": "needs_review",
                      "annotations": existing + proposals}
            validate_coco({"categories": CATEGORIES,
                           "images": [{"id": 1, "file_name": name, "width": image.width, "height": image.height}],
                           "annotations": [dict(a, id=i + 1, image_id=1) for i, a in enumerate(record["annotations"])]})
            write_json(checkpoint, record)
            records.append(record)
            if index <= args.previews:
                # Keep the original extension in the name, preventing collisions
                # between e.g. scene.jpg and scene.png in the same input folder.
                save_preview(image, record["annotations"], output / "previews" / (name + ".png"))
    except (Exception, KeyboardInterrupt) as exc:
        failed = exc
    finally:
        done = {r["file_name"] for r in records}
        summary = export_records(records, output, source, [n for n in names if n not in done])
        print(f"Saved {summary['images']} images / {summary['annotations']} annotations. "
              f"Pending images: {len(summary['pending_images'])}.", flush=True)
        print(f"Draft annotations: {output / 'annotations_draft.json'}", flush=True)
    if failed is not None:
        raise RuntimeError(f"Run stopped: {failed}. Completed images were saved. Rerun the same command with --resume.") from failed
    if args.pack_images:
        pack_images(records, image_root, output / "images_for_cvat.zip")
        print(f"CVAT image upload: {output / 'images_for_cvat.zip'}", flush=True)
    print("Review EVERY image in CVAT. Zero detections are not verified negatives.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="Create draft COCO instance annotations.")
    p.add_argument("--images", type=Path, required=True, help="Photographs only; subfolders are preserved.")
    p.add_argument("--output", type=Path, required=True, help="New result folder, separate from the input images.")
    p.add_argument("--config", type=Path, default=ROOT / "config.json")
    p.add_argument("--models-dir", type=Path, default=ROOT / "models")
    p.add_argument("--device", default="cuda", help="cuda, cuda:0, etc.; cpu is possible but very slow.")
    p.add_argument("--qwen", action="store_true", help="Enable optional local Qwen classification suggestions.")
    p.add_argument("--existing-coco", type=Path, help="Preserve and supplement these COCO annotations.")
    p.add_argument("--category-map", type=Path, help="JSON mapping from source category names to our seven names.")
    p.add_argument("--classes", help="Only prompt these comma-separated canonical class names.")
    p.add_argument("--limit", type=int, default=0, help="Process the first N sorted images; 0 means all.")
    p.add_argument("--previews", type=int, default=20, help="Save previews for the first N images; 0 disables.")
    p.add_argument("--pack-images", action="store_true", help="Also ZIP the processed original photos for CVAT upload.")
    p.add_argument("--resume", action="store_true", help="Reuse per-image checkpoints with the same inputs/settings.")
    p.add_argument("--dry-run", action="store_true", help="Check image/source paths and geometry without model inference.")
    p = sub.add_parser("validate", help="Validate the canonical seven-class COCO export.")
    p.add_argument("--coco", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        if args.limit < 0 or args.previews < 0:
            parser.error("--limit and --previews must be nonnegative.")
        run(args)
    else:
        print(json.dumps(validate_coco(read_json(args.coco)), indent=2))
        print("File structure/geometry checked. This does not approve annotation accuracy.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
