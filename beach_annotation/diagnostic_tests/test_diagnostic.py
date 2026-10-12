"""Offline checks for case selection, pixel reconstruction and failure reporting.

These tests do not establish Qwen's visual accuracy or run GPU inference.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

import numpy as np
from PIL import Image

import diagnose_qwen as d


class DiagnosticTests(unittest.TestCase):
    def args(self, root, dry_run=False):
        return argparse.Namespace(cases=d.ROOT / "diagnostic_cases/cases.json", output=root / "out",
                                  model_dir=root / "models", device="cpu", dtype="auto", dry_run=dry_run)

    def test_wide_sea_rectangle_and_pipeline_square_are_different_inputs(self):
        _, cases = d.load_cases(d.ROOT / "diagnostic_cases/cases.json")
        self.assertEqual(len(cases), 5)
        _, views = cases[0]
        self.assertEqual(views["exact"].size, (765, 105))
        self.assertEqual(views["square"].size, (765, 574))
        self.assertTrue(np.any(np.all(np.array(views["context"]) == [255, 0, 0], axis=-1)))

    def test_prepare_views_keeps_source_pixels_and_coordinates(self):
        pixels = np.arange(120 * 160 * 3, dtype=np.uint8).reshape(120, 160, 3)
        image = Image.fromarray(pixels)
        views = d.prepare_views(image, [45, 35, 64, 44])
        np.testing.assert_array_equal(np.array(views["exact"]), pixels[35:44, 45:64])
        np.testing.assert_array_equal(np.array(image), pixels)

    def test_processed_pixel_previews_preserve_colors_orientation_and_image_order(self):
        # Independent patch-by-patch oracle: iterate the documented Qwen merge
        # blocks instead of using the reconstruction's reshape/transpose logic.
        processor = SimpleNamespace(patch_size=14, temporal_patch_size=2, merge_size=2,
                                    do_normalize=True, do_rescale=True, rescale_factor=1 / 255,
                                    image_mean=[.48, .45, .40], image_std=[.26, .25, .27])
        rng = np.random.default_rng(42)
        originals = [rng.integers(0, 256, (56, 28, 3), dtype=np.uint8),
                     rng.integers(0, 256, (28, 56, 3), dtype=np.uint8)]
        rows, grids = [], []
        for image in originals:
            height, width = image.shape[:2]
            norm = (image / 255 - processor.image_mean) / processor.image_std
            grids.append([1, height // 14, width // 14])
            for block_y in range(0, height, 28):
                for block_x in range(0, width, 28):
                    for offset_y in (0, 14):
                        for offset_x in (0, 14):
                            patch = norm[block_y + offset_y:block_y + offset_y + 14,
                                         block_x + offset_x:block_x + offset_x + 14]
                            rows.append(np.stack([patch[:, :, c] for c in range(3) for _ in range(2)]).ravel())
        reconstructed = d.reconstruct_pixels(np.array(rows, dtype=np.float32), grids, processor)
        for actual, expected in zip(reconstructed, originals):
            np.testing.assert_array_equal(np.array(actual), expected)
        with self.assertRaises(ValueError):
            d.reconstruct_pixels(np.array(rows), grids[:1], processor)

    def test_dry_run_prepares_crops_without_loading_models(self):
        def forbidden(*args):
            self.fail("Dry run attempted to load a model")
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            report = d.run(self.args(root, True), forbidden)
            self.assertEqual(report["planned_generations"], 20)
            self.assertEqual(report["results"], [])
            self.assertTrue((root / "out/inputs/case_05/context.png").is_file())
            self.assertIn("Not run.", (root / "out/report.html").read_text())

    def test_controlled_comparisons_and_report_do_not_leak_reviewer_notes(self):
        calls = []
        class FakeRunner:
            info = {"test_double": True}
            def __init__(self, *args):
                pass
            def generate(self, images, prompt, trial):
                calls.append(([hash(image.tobytes()) for image in images], prompt))
                return '<script>untrusted model text</script>', {}
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            report = d.run(self.args(root), FakeRunner)
            self.assertEqual(len(calls), 20)
            for i in range(0, 20, 4):
                self.assertEqual(calls[i][1], calls[i + 1][1])
                self.assertEqual(calls[i + 2][0], calls[i + 3][0])
                for _, prompt in calls[i:i + 4]:
                    for case in report["cases"]:
                        self.assertNotIn(case["reviewer_note"], prompt)
                        self.assertNotIn(case["image"], prompt)
            page = (root / "out/report.html").read_text()
            self.assertNotIn('<script>untrusted', page)
            self.assertIn('&lt;script&gt;', page)
            self.assertTrue((root / "out.zip").is_file())
            with self.assertRaisesRegex(ValueError, "Output already exists"):
                d.run(self.args(root), FakeRunner)

    def test_failure_saves_completed_answers_and_packages_the_error(self):
        class FailingRunner:
            info = {"test_double": True}
            def __init__(self, *args):
                self.calls = 0
            def generate(self, *args):
                self.calls += 1
                if self.calls == 3:
                    raise RuntimeError("Synthetic GPU failure")
                return "A visible scene.", {}
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            with self.assertRaisesRegex(RuntimeError, "Diagnostic stopped"):
                d.run(self.args(root), FailingRunner)
            report = d.read_json(root / "out/report.json")
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(len(report["results"]), 2)
            self.assertIn("Synthetic GPU failure", report["error"])
            with zipfile.ZipFile(root / "out.zip") as archive:
                saved = json.loads(archive.read("out/report.json"))
            self.assertEqual(saved, report)

    def test_format_check_does_not_guess_labels_from_prose(self):
        for text in ('Looks like a bottle', '{"label":"unknown"}', '{"label":null}'):
            self.assertFalse(d.parse_label(text)["json_label_valid"])
        self.assertEqual(d.parse_label('{"label":"not_trash","reason":"Water"}')["label"], "not_trash")


if __name__ == "__main__":
    unittest.main()
