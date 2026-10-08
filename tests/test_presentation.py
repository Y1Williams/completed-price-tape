"""Demo rendering checks without plotting or optimization."""
import base64
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tape_example.presentation import demo


class DemoRenderingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.source = ROOT / "data/reference"
        self.data = json.loads(
            (self.source / "continuum_results.json").read_text(encoding="utf-8")
        )
        self.grid = max(self.data["numerical_grid_screen"], key=lambda item: item["n"])

    def fake_plot(self, input_dir, output_dir, font):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image = output_dir / "figure1.png"
        image.write_bytes(b"rendered-figure")
        return image

    def read_payload(self, destination):
        html = destination.read_text(encoding="utf-8")
        match = re.search(r'<script type="application/json" id="results">(.*?)</script>', html)
        self.assertIsNotNone(match)
        return json.loads(match.group(1))

    def test_original_23_points_are_preserved(self):
        with patch("tape_example.presentation.plot", side_effect=self.fake_plot):
            destination = demo(self.source, self.output)
        payload = self.read_payload(destination)
        self.assertEqual(set(payload), {"points", "n"})
        self.assertEqual(payload["n"], 64)
        self.assertEqual(len(payload["points"]), 23)
        self.assertEqual(
            [round(point["u"], 10) for point in payload["points"]],
            [value / 10 for value in range(1, 21)] + [3, 4, 6],
        )
        rows = [row for row in self.grid["rows"] if row["U"] > 0]
        for point, row in zip(payload["points"], rows):
            with self.subTest(u=point["u"]):
                self.assertEqual(point["u"], row["U"])
                self.assertEqual(point["tape"], row["tape"]["value"])
                self.assertEqual(point["cost"], self.grid["cost"]["value"])
                self.assertEqual(point["ratio"], point["cost"] / point["tape"])
                self.assertEqual(point["multiple"], point["tape"] / point["cost"])

    def test_baseline_values_and_returned_image_are_embedded(self):
        image = self.output / "rendered.png"
        image.write_bytes(b"baseline-figure")
        with patch("tape_example.presentation.plot", return_value=image) as render:
            destination = demo(self.source, self.output)
        render.assert_called_once_with(self.source, self.output, font="STIXGeneral")
        baseline = next(point for point in self.read_payload(destination)["points"] if point["u"] == 2)
        self.assertAlmostEqual(baseline["cost"], 0.002535925537346756, places=15)
        self.assertAlmostEqual(baseline["tape"], 0.008026864475963645, places=15)
        self.assertAlmostEqual(baseline["ratio"], 0.31592978116680015, places=15)
        html = destination.read_text(encoding="utf-8")
        self.assertIn(base64.b64encode(b"baseline-figure").decode("ascii"), html)

    def test_changed_data_is_rendered_and_embedded(self):
        fresh = self.output / "fresh-data"
        fresh.mkdir()
        self.grid["cost"]["value"] *= 2
        row = next(row for row in self.grid["rows"] if row["U"] == 2)
        row["tape"]["value"] *= 3
        (fresh / "continuum_results.json").write_text(json.dumps(self.data), encoding="utf-8")
        with patch("tape_example.presentation.plot", side_effect=self.fake_plot) as render:
            destination = demo(fresh, self.output)
        render.assert_called_once_with(fresh, self.output, font="STIXGeneral")
        point = next(point for point in self.read_payload(destination)["points"] if point["u"] == 2)
        self.assertEqual(point["cost"], self.grid["cost"]["value"])
        self.assertEqual(point["tape"], row["tape"]["value"])
        self.assertEqual(point["ratio"], point["cost"] / point["tape"])
        self.assertEqual(point["multiple"], point["tape"] / point["cost"])
        self.assertIn(
            base64.b64encode(b"rendered-figure").decode("ascii"),
            destination.read_text(encoding="utf-8"),
        )

    def test_requested_font_is_forwarded(self):
        with patch("tape_example.presentation.plot", side_effect=self.fake_plot) as render:
            demo(self.source, self.output, "Times New Roman")
        render.assert_called_once_with(self.source, self.output, font="Times New Roman")

    def test_stale_output_is_overwritten_on_every_call(self):
        image = self.output / "figure1.png"
        image.write_bytes(b"old-figure")
        destination = self.output / "demo.html"
        destination.write_text("old-demo", encoding="utf-8")
        with patch("tape_example.presentation.plot", side_effect=self.fake_plot) as render:
            for _ in range(2):
                self.assertEqual(demo(self.source, self.output), destination)
        self.assertEqual(render.call_count, 2)
        self.assertEqual(image.read_bytes(), b"rendered-figure")
        html = destination.read_text(encoding="utf-8")
        self.assertNotIn("old-demo", html)
        self.assertNotIn(base64.b64encode(b"old-figure").decode("ascii"), html)
        self.assertIn(base64.b64encode(b"rendered-figure").decode("ascii"), html)

    def test_new_output_folder_is_created(self):
        output = self.output / "new" / "nested-output"
        self.assertFalse(output.exists())
        with patch("tape_example.presentation.plot", side_effect=self.fake_plot) as render:
            destination = demo(self.source, output)
        render.assert_called_once_with(self.source, output, font="STIXGeneral")
        self.assertEqual(destination, output / "demo.html")
        self.assertTrue(destination.is_file())
        self.assertEqual((output / "figure1.png").read_bytes(), b"rendered-figure")
        self.assertEqual(len(self.read_payload(destination)["points"]), 23)


if __name__ == "__main__":
    unittest.main(verbosity=2)
