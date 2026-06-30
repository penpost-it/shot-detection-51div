"""Tests for the shot-grouping analysis core.

Pure-Python core (no numpy/cv2), so these run with the standard library alone:

    cd python_poc && python3 -m unittest grouping.test_grouping -v

They are also collected as-is by pytest.
"""

import math
import unittest

from grouping.grouping import analyze_grouping, min_enclosing_circle


def det(cx, cy, size=2.0):
    """Bbox dict whose center is (cx, cy)."""
    half = size / 2.0
    return {
        "x1": cx - half,
        "y1": cy - half,
        "x2": cx + half,
        "y2": cy + half,
        "confidence": 0.9,
        "class_name": "bullet_hole",
    }


class _ObjDet:
    """Detection-like object exposing x1..y2 as attributes (like the YOLO Detection)."""

    def __init__(self, cx, cy, size=2.0):
        half = size / 2.0
        self.x1, self.y1, self.x2, self.y2 = cx - half, cy - half, cx + half, cy + half
        self.confidence, self.class_name = 0.9, "bullet_hole"


class TestMinEnclosingCircle(unittest.TestCase):
    def test_two_points_form_a_diameter(self):
        center, radius = min_enclosing_circle([(0.0, 0.0), (10.0, 0.0)])
        self.assertAlmostEqual(center[0], 5.0, places=5)
        self.assertAlmostEqual(center[1], 0.0, places=5)
        self.assertAlmostEqual(radius, 5.0, places=5)

    def test_obtuse_triangle_uses_longest_side(self):
        # (5,5) lies exactly on the circle with diameter (0,0)-(10,0), so r stays 5.
        center, radius = min_enclosing_circle([(0.0, 0.0), (10.0, 0.0), (5.0, 5.0)])
        self.assertAlmostEqual(radius, 5.0, places=5)
        self.assertAlmostEqual(center[0], 5.0, places=5)
        self.assertAlmostEqual(center[1], 0.0, places=5)

    def test_acute_triangle_uses_circumcircle(self):
        center, radius = min_enclosing_circle([(0.0, 0.0), (2.0, 0.0), (1.0, math.sqrt(3))])
        self.assertAlmostEqual(radius, 2.0 / math.sqrt(3), places=5)
        self.assertAlmostEqual(center[0], 1.0, places=5)
        self.assertAlmostEqual(center[1], math.sqrt(3) / 3.0, places=5)

    def test_single_point_has_zero_radius(self):
        center, radius = min_enclosing_circle([(3.0, 4.0)])
        self.assertEqual(center, (3.0, 4.0))
        self.assertEqual(radius, 0.0)


class TestAnalyzeGrouping(unittest.TestCase):
    def test_no_shots(self):
        r = analyze_grouping([], (1000, 1000))
        self.assertEqual(r.status, "no_shots")
        self.assertIsNone(r.formed)
        self.assertEqual(r.n_shots, 0)

    def test_insufficient_shots(self):
        r = analyze_grouping([det(10, 10)], (1000, 1000), min_shots=2)
        self.assertEqual(r.status, "insufficient_shots")
        self.assertIsNone(r.formed)
        self.assertEqual(r.n_shots, 1)

    def test_two_points_within_threshold_are_formed(self):
        r = analyze_grouping([det(100, 100), det(180, 100)], (1000, 1000), threshold_px=100)
        self.assertEqual(r.n_shots, 2)
        self.assertAlmostEqual(r.diameter_px, 80.0, places=4)
        self.assertTrue(r.formed)
        self.assertEqual(r.status, "formed")

    def test_two_points_exceeding_threshold_are_not_formed(self):
        r = analyze_grouping([det(100, 100), det(180, 100)], (1000, 1000), threshold_px=50)
        self.assertFalse(r.formed)
        self.assertEqual(r.status, "not_formed")

    def test_diameter_fraction_is_normalized_by_canvas_diagonal(self):
        # centers 100 px apart; canvas 600x800 -> diagonal 1000 -> frac 0.1
        r = analyze_grouping([det(100, 100), det(200, 100)], (600, 800))
        self.assertAlmostEqual(r.canvas_diagonal_px, 1000.0, places=4)
        self.assertAlmostEqual(r.diameter_px, 100.0, places=4)
        self.assertAlmostEqual(r.diameter_frac, 0.1, places=4)

    def test_threshold_px_overrides_fraction(self):
        r = analyze_grouping(
            [det(100, 100), det(180, 100)], (1000, 1000), threshold_frac=0.0, threshold_px=100
        )
        self.assertAlmostEqual(r.threshold_px, 100.0, places=4)
        self.assertTrue(r.formed)

    def test_centroid_is_mean_of_bbox_centers(self):
        r = analyze_grouping([det(0, 0), det(10, 0), det(5, 10)], (1000, 1000), threshold_px=1e9)
        self.assertAlmostEqual(r.centroid[0], 5.0, places=4)
        self.assertAlmostEqual(r.centroid[1], 10.0 / 3.0, places=4)

    def test_object_detections_match_dict_detections(self):
        rd = analyze_grouping([det(100, 100), det(180, 100)], (1000, 1000), threshold_px=100)
        ro = analyze_grouping([_ObjDet(100, 100), _ObjDet(180, 100)], (1000, 1000), threshold_px=100)
        self.assertAlmostEqual(rd.diameter_px, ro.diameter_px, places=6)
        self.assertEqual(rd.status, ro.status)

    def test_tight_cluster_is_formed_with_default_threshold(self):
        # diagonal ~1414, default frac 0.12 -> ~169 px; spread ~14 px -> formed
        pts = [det(500, 500), det(510, 500), det(505, 510)]
        self.assertTrue(analyze_grouping(pts, (1000, 1000)).formed)

    def test_wide_spread_is_not_formed_with_default_threshold(self):
        pts = [det(100, 100), det(900, 100), det(500, 800)]
        self.assertFalse(analyze_grouping(pts, (1000, 1000)).formed)


class TestPublicApi(unittest.TestCase):
    def test_public_symbols_importable_from_package_root(self):
        import grouping

        for name in ("analyze_grouping", "GroupingResult", "min_enclosing_circle", "draw_grouping"):
            self.assertTrue(hasattr(grouping, name), f"grouping.{name} should be exported")


if __name__ == "__main__":
    unittest.main()
