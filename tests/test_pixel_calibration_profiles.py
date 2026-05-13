"""Tests for calibration-profile metadata and mapping resolution.

Inputs: small converted-metadata dictionaries and profile-mapping fixtures.
Outputs: resolved rig/mode/scanner profile fields and conservative group
selection decisions.
"""

import unittest
from datetime import date

from twopy.pixel_calibration import PixelCalibrationRow
from twopy.pixel_calibration_profiles import (
    DEFAULT_PIXEL_CALIBRATION_PROFILE_PATH,
    PixelCalibrationProfile,
    load_pixel_calibration_profile_mappings,
    resolve_pixel_calibration_profile,
    select_pixel_calibration_group,
)


class PixelCalibrationProfileTests(unittest.TestCase):
    """Tests for calibration-profile resolution."""

    def test_loads_default_profile_mappings(self) -> None:
        """Confirm twopy ships explicit config-to-mode mappings.

        Inputs: the public default profile-mapping CSV.
        Outputs: parsed rows for the historical ScanImage configs Gustavo
        identified.
        """
        mappings = load_pixel_calibration_profile_mappings(
            DEFAULT_PIXEL_CALIBRATION_PROFILE_PATH,
        )

        by_config = {mapping.config_name: mapping for mapping in mappings}
        self.assertEqual(by_config["128x128_1ms_6.5Hz"].mode, 2)
        self.assertEqual(by_config["512x512_4ms_zStack"].scanner, "galvo")
        self.assertEqual(
            by_config["256x128_0.5ms_fastAcquisition"].acquisition_fields[
                "lines_per_frame"
            ],
            128,
        )

    def test_resolves_direct_metadata_before_mapping(self) -> None:
        """Confirm direct metadata can identify a complete profile.

        Inputs: run metadata with a day-rig name and acquisition metadata with
        direct scanner and mode fields.
        Outputs: a metadata-sourced profile without relying on config mapping.
        """
        profile = resolve_pixel_calibration_profile(
            {
                "scanimage.imagingSystem": "Galvo",
                "acq.mode": 2,
                "configName": "unmapped_config",
            },
            {"rig_name": "day rig"},
            mappings=(),
        )

        self.assertEqual(profile.rig, "day")
        self.assertEqual(profile.mode, 2)
        self.assertEqual(profile.scanner, "galvo")
        self.assertEqual(profile.source, "metadata")

    def test_mapping_supplies_mode_when_scanimage_lacks_it(self) -> None:
        """Confirm config mappings fill the missing lab mode field.

        Inputs: old ScanImage state metadata with config name and acquisition
        timing fields, but no explicit calibration mode.
        Outputs: mapped mode and scanner with no guessed rig.
        """
        profile = resolve_pixel_calibration_profile(
            {
                "SI.hConfigurationSaver.cfgFilename": (
                    r"C:\Users\Scientifica\Desktop\128x128_1ms_6.5Hz.cfg"
                ),
                "SI.hRoiManager.linesPerFrame": 128,
                "SI.hRoiManager.pixelsPerLine": 128,
                "acq.pixelTime": 0.0000064,
                "SI.hRoiManager.linePeriod": 0.0012,
                "acq.scanAngleMultiplierFast": 1,
                "acq.scanAngleMultiplierSlow": 0.57744,
            },
            {"rig_name": "UnknownRig"},
            mappings=load_pixel_calibration_profile_mappings(
                DEFAULT_PIXEL_CALIBRATION_PROFILE_PATH,
            ),
        )

        self.assertIsNone(profile.rig)
        self.assertEqual(profile.mode, 2)
        self.assertEqual(profile.scanner, "galvo")
        self.assertEqual(profile.source, "metadata+mapping")

    def test_odorrig_run_metadata_maps_to_night_rig(self) -> None:
        """Confirm historical OdorRig metadata selects night calibration.

        Inputs: run metadata from older ScanImage recordings.
        Outputs: night rig so the UI does not fall back to the first dropdown
        item.
        """
        profile = resolve_pixel_calibration_profile(
            {"configName": "128x128_1ms_6.5Hz"},
            {"rig_name": "OdorRig"},
            mappings=(),
        )

        self.assertEqual(profile.rig, "night")

    def test_direct_scanner_conflict_rejects_config_mapping(self) -> None:
        """Confirm direct scanner metadata prevents stale mapping use.

        Inputs: a mapped galvo config name plus direct resonant scanner metadata.
        Outputs: only the directly observed resonant scanner is retained.
        """
        profile = resolve_pixel_calibration_profile(
            {
                "configName": "128x128_1ms_6.5Hz",
                "SI.hScan2D.scannerType": "Resonant",
            },
            {},
            mappings=load_pixel_calibration_profile_mappings(
                DEFAULT_PIXEL_CALIBRATION_PROFILE_PATH,
            ),
        )

        self.assertIsNone(profile.mode)
        self.assertEqual(profile.scanner, "res")
        self.assertEqual(profile.source, "metadata")

    def test_selects_unique_calibration_group(self) -> None:
        """Confirm complete profile fields select one measured group.

        Inputs: day and night rows sharing mode/scanner plus a complete day
        profile.
        Outputs: the unique day group.
        """
        group = select_pixel_calibration_group(
            PixelCalibrationProfile(
                rig="day",
                mode=2,
                scanner="galvo",
                config_name=None,
                source="metadata",
                evidence=("run.rig_name", "acq.mode", "scanimage.imagingSystem"),
            ),
            _calibrations(),
        )

        self.assertIsNotNone(group)
        if group is not None:
            self.assertEqual(group.rig, "day")
            self.assertEqual(group.mode, 2)
            self.assertEqual(group.scanner, "galvo")

    def test_selects_unique_group_from_partial_profile(self) -> None:
        """Confirm measured rows can disambiguate missing rig metadata.

        Inputs: a profile with mode and scanner but no rig plus a calibration
        registry where only one group has that mode/scanner.
        Outputs: the unique measured group is selected.
        """
        group = select_pixel_calibration_group(
            PixelCalibrationProfile(
                rig=None,
                mode=3,
                scanner="res",
                config_name=None,
                source="metadata",
                evidence=("acq.mode", "SI.hScan2D.scannerType"),
            ),
            (
                *_calibrations(),
                PixelCalibrationRow(
                    "day",
                    3,
                    "res",
                    7.0,
                    0.3,
                    date(2023, 12, 14),
                ),
            ),
        )

        self.assertIsNotNone(group)
        if group is not None:
            self.assertEqual(group.rig, "day")
            self.assertEqual(group.mode, 3)
            self.assertEqual(group.scanner, "res")

    def test_incomplete_profile_does_not_select_ambiguous_group(self) -> None:
        """Confirm mode/scanner alone do not choose between rigs.

        Inputs: measured day and night rows plus a profile without rig.
        Outputs: no automatic group selection.
        """
        group = select_pixel_calibration_group(
            PixelCalibrationProfile(
                rig=None,
                mode=2,
                scanner="galvo",
                config_name="128x128_1ms_6_5hz",
                source="metadata+mapping",
                evidence=("profile_mapping.config_name",),
            ),
            _calibrations(),
        )

        self.assertIsNone(group)


def _calibrations() -> tuple[PixelCalibrationRow, ...]:
    """Return small day/night calibration rows for profile selection tests.

    Returns:
        Two measured rows with the same mode and scanner on different rigs.
    """
    measured_on = date(2023, 12, 14)
    return (
        PixelCalibrationRow("day", 2, "galvo", 10.0, 0.2, measured_on),
        PixelCalibrationRow("night", 2, "galvo", 10.0, 0.25, measured_on),
    )


if __name__ == "__main__":
    unittest.main()
