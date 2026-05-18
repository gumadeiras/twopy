"""Pure tests for napari status text helpers.

Inputs: counts and noun labels.
Outputs: short status phrases without importing Qt.
"""

import unittest

from twopy.napari.text import counted_noun


class NapariTextTest(unittest.TestCase):
    """Tests text helpers used by the napari adapter."""

    def test_counted_noun_formats_status_text(self) -> None:
        """Confirm napari status text uses normal singular and plural words.

        Inputs: singular and plural counts.
        Outputs: phrases with full singular or plural nouns.
        """
        self.assertEqual(counted_noun(1, "file"), "1 file")
        self.assertEqual(counted_noun(2, "file"), "2 files")
        self.assertEqual(counted_noun(1, "ROI", "ROIs"), "1 ROI")
        self.assertEqual(counted_noun(2, "ROI", "ROIs"), "2 ROIs")


if __name__ == "__main__":
    unittest.main()
