"""Regressions from the first five BeachLitter photos, plus mocked failure paths.

The fixture contains saved predictions, not ground truth. These tests verify
data handling and error reporting; they cannot measure pretrained accuracy.
"""
import argparse
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import annotate as a


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = a.read_json(a.ROOT / "config.json")
        cls.pilot = a.read_json(Path(__file__).parent / "fixtures" / "pilot_regression.json")

    def test_pilot_images_now_receive_overlapping_tiles(self):
        cfg = self.config["detection"]
        for image in self.pilot["images"]:
            w, h = image["width"], image["height"]
            windows = list(a.tile_windows(w, h, cfg["tile_size"], cfg["tile_overlap"]))
            self.assertGreater(len(windows), 1)
            coverage = np.zeros((h, w), bool)
            for x0, y0, x1, y1 in windows[1:]:
                coverage[y0:y1, x0:x1] = True
                self.assertLessEqual(x1 - x0, 384)
                self.assertLessEqual(y1 - y0, 384)
            self.assertTrue(coverage.all())

    def test_old_literal_enum_and_null_labels_are_rejected(self):
        responses = self.pilot["qwen_raw_responses"]
        self.assertEqual(len(responses), 30)
        for text in responses:
            parsed = a.parse_qwen_response(text)
            if 'suggestion|uncertain|not_trash' in text or '"label": "null"' in text:
                self.assertEqual(parsed["status"], "parse_error")
                self.assertIsNone(parsed["class_name"])
        # Do not salvage a label from a reason such as "the object is water".
        self.assertEqual(a.parse_qwen_response('{"reason":"The object is water, not trash."}')["status"],
                         "parse_error")

    def test_explicit_labels_and_wrappers_normalize_without_extra_classes(self):
        for label in (*a.CLASS_NAMES, "uncertain", "not_trash"):
            payload = json.dumps({"label": label, "reason": "test"})
            for response in (payload, '[' + payload + ']', '```json\n' + payload + '\n```'):
                parsed = a.parse_qwen_response(response)
                self.assertEqual(parsed["status"], "suggestion" if label in a.CLASS_NAMES else label)
                self.assertEqual(parsed["class_name"], label if label in a.CLASS_NAMES else None)
        for text in ('{"label":"unknown"}', '{"label":"uncertain","class_name":"can"}',
                     '{"status":"suggestion","class_name":"glass","label":"can"}',
                     '[{"label":"can"},{"label":"glass"}]'):
            self.assertEqual(a.parse_qwen_response(text)["status"], "parse_error")

    def test_retry_can_recover_but_never_silently_skip_classifier(self):
        attempts = []
        def recover(previous):
            attempts.append(previous)
            return 'bad JSON' if previous is None else '{"label":"other_trash","reason":"Shoe"}'
        with contextlib.redirect_stdout(io.StringIO()):
            result = a.classify_with_retry(recover, 2)
            self.assertEqual(result["class_name"], "other_trash")
            self.assertEqual(result["attempts"], 2)
            self.assertEqual(attempts, [None, 'bad JSON'])
            with self.assertRaises(a.QwenClassificationError) as caught:
                a.classify_with_retry(lambda previous: 'bad JSON', 2)
            self.assertEqual(caught.exception.attempts, ['bad JSON', 'bad JSON'])

    def test_context_mark_does_not_overwrite_original_or_tight_crop(self):
        image = Image.new("RGB", (200, 150), (20, 30, 40))
        original = np.array(image)
        tight, context = a.qwen_crops(image, [60, 40, 69, 47])
        self.assertLess(tight.width, context.width)
        self.assertFalse(np.any(np.all(np.array(tight) == [255, 0, 0], axis=-1)))
        self.assertTrue(np.any(np.all(np.array(context) == [255, 0, 0], axis=-1)))
        np.testing.assert_array_equal(np.array(image), original)

    def test_observed_sea_sand_regions_go_to_background_review_without_deletion(self):
        images = {im["id"]: im for im in self.pilot["images"]}
        annotations = copy.deepcopy(self.pilot["annotations"])
        for ann in annotations:
            im = images[ann["image_id"]]
            before = copy.deepcopy(ann["segmentation"])
            a.assign_review_group(ann, im["width"], im["height"], self.config)
            self.assertEqual(ann["segmentation"], before)
            if ann["id"] in (1, 6, 9, 21):
                self.assertEqual(ann["attributes"]["review_group"], "background_review")
        self.assertEqual(len(annotations), 30)

    def test_pilot_near_duplicates_merge_but_bottle_part_is_retained(self):
        merged = []
        original = copy.deepcopy(self.pilot)
        for im in original["images"]:
            anns = [x for x in original["annotations"] if x["image_id"] == im["id"]]
            merged.extend(a.merge_masks(anns, self.config["detection"]["mask_merge_iou"],
                                        im["height"], im["width"], self.config))
        by_id = {ann["id"]: ann for ann in merged}
        self.assertEqual(len(merged), 27)
        self.assertTrue({5, 8, 28}.isdisjoint(by_id))
        for number in (15, 18):
            self.assertIn(number, by_id)
            self.assertIn("contained_overlap", by_id[number]["attributes"]["flags"])
        for number in (2, 3, 23):
            self.assertTrue(json.loads(by_id[number]["attributes"]["merged_evidence"]))

    def test_review_exports_keep_all_groups_and_preview_ids_match_csv(self):
        im = self.pilot["images"][0]
        anns = copy.deepcopy(self.pilot["annotations"][:8])
        for ann in anns:
            a.assign_review_group(ann, im["width"], im["height"], self.config)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            a.export_records([dict(im, annotations=anns)], output)
            master = a.read_json(output / "annotations_draft.json")
            queues = a.read_json(output / "review_groups.json")["groups"]
            self.assertEqual(len(master["annotations"]), 8)
            self.assertEqual(sorted(n for ids in queues.values() for n in ids), list(range(1, 9)))
            with (output / "review.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([int(r["image_object_id"]) for r in rows], list(range(1, 9)))
            self.assertEqual({r["review_group"] for r in rows}, {"candidates", "uncertain", "background_review"})

    def test_qwen_failure_stops_run_and_writes_raw_error_with_pending_image(self):
        class InvalidQwen:
            def __init__(self, *args):
                pass
            def detect(self, image, classes):
                return [{"box": [3, 3, 8, 8], "score": .8, "phrase": "can",
                         "category_id": 1, "candidate_classes": [1]}]
            def segment(self, image, box):
                mask = np.zeros((image.height, image.width), bool)
                mask[3:8, 3:8] = True
                return mask, .9
            def classify(self, image, box):
                return a.classify_with_retry(lambda previous: 'invalid JSON', 2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "images").mkdir()
            Image.new("RGB", (30, 20)).save(root / "images" / "photo.png")
            a.write_json(root / "models" / "model_versions.json", {
                k: dict(v, revision="test-only") for k, v in self.config["models"].items()})
            args = argparse.Namespace(images=root / "images", output=root / "out", config=a.ROOT / "config.json",
                                      models_dir=root / "models", device="cpu", qwen=True, existing_coco=None,
                                      category_map=None, classes=None, limit=0, previews=0, resume=False,
                                      dry_run=False, pack_images=False)
            with patch.object(a, "Models", InvalidQwen), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "Qwen returned invalid"):
                    a.run(args)
            error = a.read_json(root / "out" / "qwen_error.json")
            self.assertEqual(error["raw_attempts"], ['invalid JSON', 'invalid JSON'])
            self.assertEqual(error["target_box_xyxy"], [3, 3, 8, 8])
            summary = a.read_json(root / "out" / "summary.json")
            self.assertEqual(summary["pending_images"], ['photo.png'])
            self.assertEqual(summary["annotations"], 0)


if __name__ == "__main__":
    unittest.main()
