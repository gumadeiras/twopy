"""Tests for pixel-size calibration loading and resolution.

Inputs: small CSV fixtures and measured microscope calibration rows.
Outputs: direct, interpolated, and extrapolated microns-per-pixel estimates.
"""

import tempfile
import unittest
from datetime import date
from pathlib import Path

from twopy import (
    DEFAULT_PIXEL_CALIBRATION_PATH,
    PixelCalibrationRow,
    load_pixel_calibrations,
    resolve_pixel_size_um,
)


class PixelCalibrationTest(unittest.TestCase):
    """Tests the pure pixel calibration resolver."""

    def test_default_registry_is_tracked_and_loadable(self) -> None:
        """Confirm twopy ships a default dated calibration registry.

        Inputs: the public default calibration path.
        Outputs: parsed calibration rows from the checked-in registry.
        """
        rows = load_pixel_calibrations(DEFAULT_PIXEL_CALIBRATION_PATH)

        self.assertEqual(len(rows), 20)
        self.assertEqual(rows[0].measured_on, date(2023, 12, 14))
        self.assertEqual(rows[-1].mode, 6)
        self.assertEqual(rows[-1].measured_on, date(2023, 12, 14))

    def test_loads_csv_registry_rows(self) -> None:
        """Confirm calibration CSV rows become typed values.

        Inputs: a temporary CSV with one measured row.
        Outputs: one ``PixelCalibrationRow`` with numeric mode, zoom, and pixel
        size.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pixel_size.csv"
            path.write_text(
                "rig,mode,scanner,zoom,pixel_size_um,measured_on\n"
                "day,2,galvo,9,0.2392344498,2023-12-14\n",
                encoding="utf-8",
            )

            rows = load_pixel_calibrations(path)

        self.assertEqual(
            rows,
            (
                PixelCalibrationRow(
                    rig="day",
                    mode=2,
                    scanner="galvo",
                    zoom=9.0,
                    pixel_size_um=0.2392344498,
                    measured_on=date(2023, 12, 14),
                ),
            ),
        )

    def test_resolves_direct_measured_pixel_size(self) -> None:
        """Confirm exact calibration rows are used without fitting.

        Inputs: measured day-rig mode-2 galvo calibration rows.
        Outputs: direct resolution for zoom 10.
        """
        rows = _day_mode2_galvo_rows()

        resolution = resolve_pixel_size_um(
            rows,
            rig="Day",
            mode=2,
            scanner="Galvo",
            zoom=10,
        )

        self.assertEqual(resolution.method, "direct")
        self.assertEqual(resolution.reference_zooms, (10.0,))
        self.assertEqual(resolution.reference_dates, (date(2023, 12, 14),))
        self.assertAlmostEqual(resolution.pixel_size_um, 0.2139037433)

    def test_interpolates_pixels_per_micron_within_same_group(self) -> None:
        """Confirm missing zooms interpolate in pixels-per-micron space.

        Inputs: measured zooms 9 and 10 from the same rig/mode/scanner group.
        Outputs: zoom 9.5 estimate from the linear pixels-per-micron fit.
        """
        rows = _day_mode2_galvo_rows()

        resolution = resolve_pixel_size_um(
            rows,
            rig="day",
            mode=2,
            scanner="galvo",
            zoom=9.5,
        )

        expected_pixels_per_um = (1.0 / 0.2392344498 + 1.0 / 0.2139037433) / 2.0
        self.assertEqual(resolution.method, "interpolated")
        self.assertEqual(resolution.reference_zooms, (9.0, 10.0))
        self.assertEqual(
            resolution.reference_dates,
            (date(2023, 12, 14), date(2023, 12, 14)),
        )
        self.assertAlmostEqual(resolution.pixel_size_um, 1.0 / expected_pixels_per_um)

    def test_requires_explicit_extrapolation_outside_measured_range(self) -> None:
        """Confirm outside-range zooms fail unless the caller opts in.

        Inputs: measured zooms 9 and 10, then requested zoom 12.
        Outputs: validation error by default and extrapolated value when
        explicitly allowed.
        """
        rows = _day_mode2_galvo_rows()

        with self.assertRaisesRegex(ValueError, "outside measured range"):
            resolve_pixel_size_um(
                rows,
                rig="day",
                mode=2,
                scanner="galvo",
                zoom=12,
            )

        resolution = resolve_pixel_size_um(
            rows,
            rig="day",
            mode=2,
            scanner="galvo",
            zoom=12,
            allow_extrapolation=True,
        )

        self.assertEqual(resolution.method, "extrapolated")
        self.assertEqual(resolution.reference_zooms, (10.0, 11.0))

    def test_does_not_borrow_between_rigs(self) -> None:
        """Confirm group keys are hard boundaries for calibration lookup.

        Inputs: day-rig rows only and a night-rig request.
        Outputs: clear missing-group error instead of cross-rig fallback.
        """
        rows = _day_mode2_galvo_rows()

        with self.assertRaisesRegex(ValueError, "No pixel calibration rows"):
            resolve_pixel_size_um(
                rows,
                rig="night",
                mode=2,
                scanner="galvo",
                zoom=10,
            )

    def test_load_rejects_conflicting_duplicate_rows(self) -> None:
        """Confirm duplicate measurements cannot silently disagree.

        Inputs: a CSV with one repeated rig/mode/scanner/zoom key.
        Outputs: clear validation error naming the duplicate conflict.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pixel_size.csv"
            path.write_text(
                "rig,mode,scanner,zoom,pixel_size_um,measured_on\n"
                "day,2,galvo,9,0.2392344498,2023-12-14\n"
                "day,2,galvo,9,0.2,2023-12-14\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
                load_pixel_calibrations(path)

    def test_load_rejects_invalid_measurement_date(self) -> None:
        """Confirm calibration dates must be explicit ISO dates.

        Inputs: a CSV row with a non-date ``measured_on`` value.
        Outputs: clear validation error naming the invalid date format.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pixel_size.csv"
            path.write_text(
                "rig,mode,scanner,zoom,pixel_size_um,measured_on\n"
                "day,2,galvo,9,0.2392344498,Dec 14 2023\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
                load_pixel_calibrations(path)


def _day_mode2_galvo_rows() -> tuple[PixelCalibrationRow, ...]:
    """Return measured day-rig mode-2 galvo calibration rows.

    Inputs: none.
    Outputs: tuple matching the lab calibration registry values.
    """
    return (
        PixelCalibrationRow("day", 2, "galvo", 9.0, 0.2392344498, date(2023, 12, 14)),
        PixelCalibrationRow(
            "day",
            2,
            "galvo",
            10.0,
            0.2139037433,
            date(2023, 12, 14),
        ),
        PixelCalibrationRow(
            "day",
            2,
            "galvo",
            11.0,
            0.1939393939,
            date(2023, 12, 14),
        ),
    )


if __name__ == "__main__":
    unittest.main()
