"""Meaningful data-integrity tests; no pretrained models or GPU needed.

Run from the project folder: python -m unittest discover -s tests -v
These are not tests of real model accuracy or GPU compatibility.
"""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

import annotate as a


class PipelineTests(unittest.TestCase):
    def test_rle_preserves_holes_and_disconnected_parts(self):
        mask = np.zeros((31, 47), dtype=bool)
        mask[2:20, 3:22] = True
        mask[6:10, 9:15] = False
        mask[24:28, 39:44] = True
        encoded = a.encode_mask(mask)
        roundtrip = json.loads(json.dumps(encoded))
        np.testing.assert_array_equal(a.mask_api().decode(roundtrip), mask)
        self.assertEqual(int(a.mask_api().area(roundtrip)), int(mask.sum()))

    def test_tiles_cover_edges_and_keep_full_view(self):
        windows = list(a.tile_windows(1817, 1241, 1024, 0.25))
        self.assertEqual(windows[0], (0, 0, 1817, 1241))
        coverage = np.zeros((1241, 1817), dtype=bool)
        for x0, y0, x1, y1 in windows[1:]:
            coverage[y0:y1, x0:x1] = True
        self.assertTrue(coverage.all())
        self.assertEqual(len(list(a.tile_windows(100, 80, 1024, 0.25))), 1)

    def test_crop_contains_box_at_image_boundary(self):
        for box in ([0, 2, 7, 15], [95, 64, 100, 70], [5, 3, 99, 69]):
            x0, y0, x1, y1 = a.crop_window(box, 100, 70, 0.35, 30)
            self.assertTrue(0 <= x0 <= box[0] < box[2] <= x1 <= 100)
            self.assertTrue(0 <= y0 <= box[1] < box[3] <= y1 <= 70)

    def test_duplicate_boxes_keep_conflicting_class_evidence(self):
        def candidate(box, cls, score):
            return {"box": box, "category_id": cls, "score": score, "phrase": "bottle"}
        out = a.merge_candidates([candidate([1, 1, 11, 11], 2, .9),
                                  candidate([1, 1, 11, 11], 3, .8),
                                  candidate([20, 1, 30, 11], 2, .7)], .85)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["candidate_classes"], [2, 3])

    def test_qwen_unsupported_labels_do_not_become_other_trash(self):
        for text in ['{"status":"suggestion","class_name":"unknown"}', 'not JSON', '{}']:
            result = a.parse_qwen_response(text)
            self.assertEqual(result["status"], "parse_error")
            self.assertIsNone(result["class_name"])
        result = a.parse_qwen_response('```json\n{"status":"uncertain","class_name":null}\n```')
        self.assertEqual(result["status"], "uncertain")

    def test_path_traversal_and_absolute_names_rejected(self):
        for name in ["../photo.jpg", "/photo.jpg", "C:\\photos\\test.jpg"]:
            with self.assertRaises(ValueError):
                a.normalized_name(name)
        self.assertEqual(a.normalized_name("batch\\photo.jpg"), "batch/photo.jpg")

    def test_preserve_geometry_and_resolve_source_ids_by_names(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            polygon = [[1, 2, 12, 2, 12, 18, 1, 18]]
            original = {"images": [{"id": 400, "file_name": "a.jpg", "width": 30, "height": 40}],
                        "categories": [{"id": 92, "name": "pet_bottle"}],
                        "annotations": [{"id": 910, "image_id": 400, "category_id": 92,
                                         "segmentation": polygon, "bbox": [1, 2, 11, 16],
                                         "area": 176, "iscrowd": 0}]}
            a.write_json(path / "source.json", original)
            a.write_json(path / "map.json", {"pet_bottle": "plastic_bottle"})
            source = a.ExistingAnnotations(path / "source.json", path / "map.json")
            ann = source.for_image("a.jpg", 30, 40)[0]
            self.assertEqual(ann["category_id"], 2)
            self.assertEqual(ann["segmentation"], polygon)
            self.assertEqual(ann["attributes"]["original_annotation_id"], "910")
            self.assertEqual(a.read_json(path / "source.json"), original)
            with self.assertRaises(ValueError):
                source.for_image("a.jpg", 40, 30)
            with self.assertRaises(ValueError):
                source.for_image("other/a.jpg", 30, 40)

    def test_empty_images_are_review_items_not_verified_negatives(self):
        with tempfile.TemporaryDirectory() as temp:
            records = [{"file_name": "empty.jpg", "width": 30, "height": 20, "annotations": []}]
            out = Path(temp)
            a.export_records(records, out)
            result = a.read_json(out / "annotations_draft.json")
            self.assertEqual(result["images"][0]["review_status"], "needs_review")
            self.assertIn("not_a_verified_negative", (out / "review.csv").read_text())

    def test_empty_sam_and_not_trash_suggestions_remain_reviewable(self):
        class FailedMask:
            def segment(self, image, box):
                return np.zeros((image.height, image.width), bool), 0.2
            def classify(self, image, box):
                return {"status": "not_trash", "reason": "Possibly a shell", "class_name": None}
        config = a.read_json(a.ROOT / "config.json")
        candidate = {"box": [2, 2, 10, 10], "score": .8, "phrase": "plastic fragment",
                     "category_id": 6, "candidate_classes": [6]}
        ann = a.draft_annotation(Image.new("RGB", (20, 20)), candidate, FailedMask(), config)
        self.assertEqual(ann["category_id"], 6)
        self.assertEqual(ann["segmentation"], [])
        self.assertIn("qwen_not_trash", ann["attributes"]["flags"])
        self.assertIn("missing_mask", ann["attributes"]["flags"])

    def test_validator_catches_bad_area_and_duplicate_ids(self):
        mask = np.zeros((20, 20), bool)
        mask[2:10, 3:11] = True
        data = {"categories": a.CATEGORIES, "images": [{"id": 1, "file_name": "a.jpg", "width": 20, "height": 20}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": a.encode_mask(mask),
                                 "bbox": [3, 2, 8, 8], "area": 64, "iscrowd": 0,
                                 "attributes": {"source": "ai_proposal"}}]}
        self.assertEqual(a.validate_coco(data)["annotations"], 1)
        damaged = copy.deepcopy(data)
        damaged["annotations"][0]["area"] = 1
        with self.assertRaises(ValueError):
            a.validate_coco(damaged)
        damaged = copy.deepcopy(data)
        damaged["annotations"].append(copy.deepcopy(damaged["annotations"][0]))
        with self.assertRaises(ValueError):
            a.validate_coco(damaged)

    def test_end_to_end_fake_inference_resume_and_image_zip(self):
        class FakeModels:
            calls = 0
            def __init__(self, *args):
                pass
            def detect(self, image, classes):
                FakeModels.calls += 1
                if image.getpixel((0, 0))[0] == 0:
                    return []
                return [{"box": [3, 4, 13, 15], "score": .9, "phrase": "cigarette butt",
                         "category_id": 7, "candidate_classes": [7]}]
            def segment(self, image, box):
                mask = np.zeros((image.height, image.width), bool)
                mask[4:15, 3:13] = True
                return mask, .95
            def classify(self, image, box):
                return None
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "images" / "nested").mkdir(parents=True)
            Image.new("RGB", (30, 20), (255, 0, 0)).save(root / "images" / "nested" / "a.png")
            Image.new("RGB", (30, 20)).save(root / "images" / "b.png")
            config = a.read_json(a.ROOT / "config.json")
            a.write_json(root / "models" / "model_versions.json", {
                key: {"repo_id": spec["repo_id"], "folder": spec["folder"], "revision": "test-only"}
                for key, spec in config["models"].items()})
            args = argparse.Namespace(images=root / "images", output=root / "results", config=a.ROOT / "config.json",
                                      models_dir=root / "models", device="cpu", qwen=False, existing_coco=None,
                                      category_map=None, classes=None, limit=0, previews=2, resume=False,
                                      dry_run=False, pack_images=True)
            with patch.object(a, "Models", FakeModels), contextlib.redirect_stdout(io.StringIO()):
                a.run(args)
                args.resume = True
                a.run(args)
                self.assertEqual(FakeModels.calls, 2, "Resume must not repeat inference.")
                # Images changing under an existing run must invalidate cached labels.
                Image.new("RGB", (30, 20), (0, 255, 0)).save(root / "images" / "b.png")
                with self.assertRaisesRegex(RuntimeError, "Image changed"):
                    a.run(args)
            with zipfile.ZipFile(root / "results" / "images_for_cvat.zip") as archive:
                self.assertEqual(set(archive.namelist()), {"b.png", "nested/a.png"})
            self.assertTrue((root / "results" / "previews" / "nested" / "a.png.png").is_file())


if __name__ == "__main__":
    unittest.main()
