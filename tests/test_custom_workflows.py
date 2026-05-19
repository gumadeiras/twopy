"""Tests for custom workflow discovery, validation, and saved metadata."""

import unittest
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import yaml
from tests.converted_files import write_converted_recording_files
from tests.tempdir import temporary_directory

from twopy.analysis.dff_options import DeltaFOverFOptions
from twopy.analysis.response_processing import ResponseProcessingOptions
from twopy.converted import load_converted_recording
from twopy.custom import (
    CustomLinePlot,
    CustomResult,
    CustomRunContext,
    CustomWorkflowProvenance,
    discover_custom_workflows,
    native_custom_workflow_paths,
    parameter_specs,
    provenance_sidecar_path,
    validate_custom_result,
    write_result_provenance,
)
from twopy.custom.native_workflows.direction_selectivity import (
    DirectionSelectivityParams,
)
from twopy.custom.native_workflows.direction_selectivity import (
    run as run_direction_selectivity,
)
from twopy.custom.native_workflows.response_kernels import (
    ResponseKernelParams,
    _kernel_output_stem,
    _lag_column_labels,
)
from twopy.napari.plotting.data import EpochResponsePlotData, ResponsePlotData
from twopy.roi import RoiSet, make_roi_set


class CustomWorkflowDiscoveryTest(unittest.TestCase):
    """Tests strict custom workflow discovery."""

    def test_discovers_valid_versioned_workflow(self) -> None:
        """Confirm discovery loads a complete versioned workflow."""
        with temporary_directory() as temp_dir:
            workflow_path = Path(temp_dir) / "dsi.py"
            workflow_path.write_text(
                "from dataclasses import dataclass\n"
                "from twopy.custom import CustomResult, CustomRunContext, workflow\n"
                "\n"
                "@dataclass(frozen=True)\n"
                "class Params:\n"
                "    threshold: float = 0.5\n"
                "\n"
                "@workflow(\n"
                "    id='direction-selectivity',\n"
                "    name='Direction selectivity',\n"
                "    version='1.0',\n"
                "    description='Computes DSI for current ROIs.',\n"
                "    params=Params,\n"
                ")\n"
                "def run(ctx: CustomRunContext, params: Params) -> CustomResult:\n"
                "    return CustomResult(message='ok')\n",
                encoding="utf-8",
            )

            result = discover_custom_workflows((workflow_path,))

            self.assertEqual(result.errors, ())
            self.assertEqual(len(result.workflows), 1)
            workflow = result.workflows[0]
            self.assertEqual(workflow.id, "direction-selectivity")
            self.assertEqual(workflow.version, "1.0")
            self.assertEqual(len(workflow.source_hash), 64)

    def test_rejects_workflow_without_version(self) -> None:
        """Confirm missing required metadata keeps a workflow out of the GUI."""
        with temporary_directory() as temp_dir:
            workflow_path = Path(temp_dir) / "bad.py"
            workflow_path.write_text(
                "from twopy.custom import CustomResult, CustomRunContext, workflow\n"
                "\n"
                "@workflow(\n"
                "    id='bad',\n"
                "    name='Bad',\n"
                "    description='Missing version.',\n"
                ")\n"
                "def run(ctx: CustomRunContext) -> CustomResult:\n"
                "    return CustomResult(message='bad')\n",
                encoding="utf-8",
            )

            result = discover_custom_workflows((workflow_path,))

            self.assertEqual(result.workflows, ())
            self.assertEqual(len(result.errors), 1)
            self.assertIn("import failed", result.errors[0].message)

    def test_rejects_three_part_workflow_version(self) -> None:
        """Confirm workflow versions use the simple ``X.Y`` format."""
        with temporary_directory() as temp_dir:
            workflow_path = Path(temp_dir) / "bad_version.py"
            workflow_path.write_text(
                "from twopy.custom import CustomResult, CustomRunContext, workflow\n"
                "\n"
                "@workflow(\n"
                "    id='bad-version',\n"
                "    name='Bad version',\n"
                "    version='1.0.0',\n"
                "    description='Has package-style version.',\n"
                ")\n"
                "def run(ctx: CustomRunContext) -> CustomResult:\n"
                "    return CustomResult(message='bad')\n",
                encoding="utf-8",
            )

            result = discover_custom_workflows((workflow_path,))

            self.assertEqual(result.workflows, ())
            self.assertEqual(len(result.errors), 1)
            self.assertIn("MAJOR.MINOR", result.errors[0].message)

    def test_rejects_unsupported_parameter_type(self) -> None:
        """Confirm GUI-unsupported parameters reject the workflow."""
        with temporary_directory() as temp_dir:
            workflow_path = Path(temp_dir) / "bad_params.py"
            workflow_path.write_text(
                "from dataclasses import dataclass\n"
                "from twopy.custom import CustomResult, CustomRunContext, workflow\n"
                "\n"
                "@dataclass(frozen=True)\n"
                "class Params:\n"
                "    values: list[float] = None\n"
                "\n"
                "@workflow(\n"
                "    id='bad-params',\n"
                "    name='Bad params',\n"
                "    version='1.0',\n"
                "    description='Has unsupported params.',\n"
                "    params=Params,\n"
                ")\n"
                "def run(ctx: CustomRunContext, params: Params) -> CustomResult:\n"
                "    return CustomResult(message='bad')\n",
                encoding="utf-8",
            )

            result = discover_custom_workflows((workflow_path,))

            self.assertEqual(result.workflows, ())
            self.assertEqual(len(result.errors), 1)
            self.assertIn("invalid workflow parameters", result.errors[0].message)

    def test_rejects_duplicate_id_version(self) -> None:
        """Confirm duplicate workflow identity is rejected."""
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            import_line = (
                "from twopy.custom import CustomResult, CustomRunContext, workflow\n"
            )
            for filename in ("first.py", "second.py"):
                (root / filename).write_text(
                    import_line + "\n"
                    "@workflow(\n"
                    "    id='duplicate',\n"
                    "    name='Duplicate',\n"
                    "    version='1.0',\n"
                    "    description='Duplicate identity.',\n"
                    ")\n"
                    "def run(ctx: CustomRunContext) -> CustomResult:\n"
                    "    return CustomResult(message='ok')\n",
                    encoding="utf-8",
                )

            result = discover_custom_workflows((root,))

            self.assertEqual(result.workflows, ())
            self.assertEqual(len(result.errors), 1)
            self.assertIn("duplicate workflow id/version", result.errors[0].message)

    def test_discovers_workflow_with_sibling_import(self) -> None:
        """Confirm workflow files can import helpers from their own folder."""
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            (root / "_helper.py").write_text(
                "WORKFLOW_NAME = 'Sibling import workflow'\n",
                encoding="utf-8",
            )
            (root / "workflow_file.py").write_text(
                "from _helper import WORKFLOW_NAME\n"
                "from twopy.custom import CustomResult, CustomRunContext, workflow\n"
                "\n"
                "@workflow(\n"
                "    id='sibling-import',\n"
                "    name=WORKFLOW_NAME,\n"
                "    version='1.0',\n"
                "    description='Uses a sibling helper.',\n"
                ")\n"
                "def run(ctx: CustomRunContext) -> CustomResult:\n"
                "    return CustomResult(message='ok')\n",
                encoding="utf-8",
            )

            result = discover_custom_workflows((root,))

            self.assertEqual(result.errors, ())
            self.assertEqual(len(result.workflows), 1)
            self.assertEqual(result.workflows[0].name, "Sibling import workflow")

    def test_discovers_native_direction_selectivity_workflow(self) -> None:
        """Confirm twopy ships DSI without requiring user config."""
        result = discover_custom_workflows(native_custom_workflow_paths())

        self.assertEqual(result.errors, ())
        workflows_by_id = {workflow.id: workflow for workflow in result.workflows}
        self.assertIn("direction-selectivity", workflows_by_id)
        self.assertIn("response-kernels", workflows_by_id)
        workflow = workflows_by_id["direction-selectivity"]
        self.assertEqual(workflow.name, "Direction selectivity")
        self.assertEqual(workflow.version, "1.0")
        self.assertIsNotNone(workflow.params_type)
        if workflow.params_type is None:
            self.fail("native DSI should define parameter controls")
        specs = {spec.name: spec for spec in parameter_specs(workflow.params_type)}
        self.assertEqual(specs["preferred_epoch"].kind, "str")
        self.assertEqual(specs["preferred_epoch"].role, "epoch")
        self.assertEqual(specs["null_epoch"].kind, "str")
        self.assertEqual(specs["null_epoch"].role, "epoch")
        self.assertEqual(specs["metric"].kind, "str")
        self.assertEqual(specs["metric"].role, "response_metric")
        self.assertEqual(specs["roi_selector"].kind, "str")
        self.assertEqual(specs["roi_selector"].role, "roi_selector")
        self.assertEqual(specs["window_start_seconds"].kind, "float")
        self.assertEqual(specs["window_start_seconds"].role, "epoch_window_start")
        self.assertIsNone(specs["window_start_seconds"].minimum)
        self.assertIsNone(specs["window_start_seconds"].maximum)
        self.assertIsNone(specs["window_start_seconds"].step)
        self.assertEqual(specs["window_stop_seconds"].kind, "float")
        self.assertEqual(specs["window_stop_seconds"].role, "epoch_window_stop")
        self.assertIsNone(specs["window_stop_seconds"].minimum)
        self.assertIsNone(specs["window_stop_seconds"].maximum)
        self.assertIsNone(specs["window_stop_seconds"].step)
        self.assertEqual(specs["rectify_responses"].kind, "bool")
        self.assertEqual(specs["dsi_threshold"].kind, "float")
        self.assertEqual(specs["dsi_threshold"].role, "table_highlight_threshold")
        self.assertEqual(specs["dsi_threshold"].maximum, 1.0)
        self.assertEqual(specs["output_name"].kind, "str")
        self.assertEqual(specs["output_name"].role, "output_name")

        kernel_workflow = workflows_by_id["response-kernels"]
        self.assertEqual(kernel_workflow.name, "Response kernels")
        self.assertEqual(kernel_workflow.version, "1.0")
        self.assertIsNotNone(kernel_workflow.params_type)
        if kernel_workflow.params_type is None:
            self.fail("native response kernels should define parameter controls")
        self.assertEqual(
            kernel_workflow.params_type.__name__,
            ResponseKernelParams.__name__,
        )
        kernel_specs = {
            spec.name: spec for spec in parameter_specs(kernel_workflow.params_type)
        }
        self.assertEqual(kernel_specs["epoch_selector"].kind, "str")
        self.assertEqual(kernel_specs["epoch_selector"].role, "epoch_selector")
        self.assertEqual(kernel_specs["stimulus_modality"].kind, "choice")
        self.assertEqual(
            kernel_specs["stimulus_modality"].choices,
            ("olfaction", "vision"),
        )

    def test_custom_line_plot_validates_y_label(self) -> None:
        """Confirm custom line plots require a readable y-axis label."""
        plot = CustomLinePlot(
            "Plot",
            np.array([0.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            y_label="",
        )

        with (
            temporary_directory() as temp_dir,
            self.assertRaisesRegex(ValueError, "y_label"),
        ):
            validate_custom_result(
                CustomResult(message="ok", plots=(plot,)),
                output_dir=Path(temp_dir) / "custom_outputs",
                expected_roi_shape=(1, 1),
            )

    def test_reference_showcase_uses_all_parameter_kinds(self) -> None:
        """Confirm the reference example covers every parameter role."""
        workflow_path = Path("examples/custom_workflows/reference_showcase.py")

        result = discover_custom_workflows((workflow_path,))

        self.assertEqual(result.errors, ())
        self.assertEqual(len(result.workflows), 1)
        params_type = result.workflows[0].params_type
        if params_type is None:
            self.fail("reference showcase should define parameter controls")
        kinds = {spec.kind for spec in parameter_specs(params_type)}
        self.assertEqual(kinds, {"bool", "int", "float", "str", "path", "choice"})
        roles = {spec.role for spec in parameter_specs(params_type)}
        self.assertEqual(
            {
                "baseline_epoch",
                "comparison_epoch",
                "epoch",
                "epoch_window_start",
                "epoch_window_stop",
                "output_name",
                "response_metric",
                "response_window_start",
                "response_window_stop",
                "roi_limit",
                "roi_selector",
                "stimulus_column",
                "table_highlight_threshold",
                None,
            },
            roles,
        )

    def test_rejects_standard_role_on_wrong_parameter_type(self) -> None:
        """Confirm standard roles validate field types during discovery."""
        with temporary_directory() as temp_dir:
            workflow_path = Path(temp_dir) / "bad_role.py"
            workflow_path.write_text(
                "from dataclasses import dataclass, field\n"
                "from twopy.custom import CustomResult, CustomRunContext, workflow\n"
                "\n"
                "@dataclass(frozen=True)\n"
                "class Params:\n"
                "    epoch: float = field(\n"
                "        default=1.0,\n"
                "        metadata={'twopy_role': 'epoch'},\n"
                "    )\n"
                "\n"
                "@workflow(\n"
                "    id='bad-role',\n"
                "    name='Bad role',\n"
                "    version='1.0',\n"
                "    description='Has an invalid role/type pairing.',\n"
                "    params=Params,\n"
                ")\n"
                "def run(ctx: CustomRunContext, params: Params) -> CustomResult:\n"
                "    return CustomResult(message='bad')\n",
                encoding="utf-8",
            )

            result = discover_custom_workflows((workflow_path,))

            self.assertEqual(result.workflows, ())
            self.assertEqual(len(result.errors), 1)
            self.assertIn("invalid workflow parameters", result.errors[0].message)

    def test_example_workflows_only_import_custom_api(self) -> None:
        """Confirm example workflows do not import private twopy internals."""
        forbidden_fragments = (
            "from twopy.analysis",
            "from twopy.converted",
            "from twopy.napari",
            "from twopy.roi",
            "from twopy.stimulus",
            "import twopy.analysis",
            "import twopy.converted",
            "import twopy.napari",
            "import twopy.roi",
            "import twopy.stimulus",
        )
        for path in Path("examples/custom_workflows").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                for fragment in forbidden_fragments:
                    self.assertNotIn(fragment, source)


class CustomRunContextApiTest(unittest.TestCase):
    """Tests the public context API exposed to workflows."""

    def test_exposes_epoch_names_and_durations(self) -> None:
        """Confirm workflows can read epoch metadata without internal imports."""
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = write_converted_recording_files(
                root,
                movie_values=np.arange(48, dtype=np.float64).reshape(12, 2, 2),
                stimulus_data=np.array(
                    [
                        [0.0, 1.0],
                        [0.3, 1.0],
                        [0.4, 2.0],
                        [0.8, 2.0],
                    ],
                    dtype=np.float64,
                ),
                high_res_pd=_timeline_photodiode(),
                stimulus_parameters_json=(
                    '[{"epochName": "Gray"}, {"epochName": "Odor"}]'
                ),
            )
            ctx = _custom_context(recording_path)

            choices = ctx.epoch_choices()

            self.assertEqual(ctx.epoch_names(), {1: "Gray", 2: "Odor"})
            self.assertEqual(
                tuple(choice.label for choice in choices),
                ("1: Gray", "2: Odor"),
            )
            self.assertEqual(tuple(choice.selector for choice in choices), (1, 2))
            self.assertEqual(ctx.epoch_durations_seconds(), {1: 0.4, 2: 0.4})
            self.assertEqual(ctx.min_epoch_duration_seconds(), 0.4)
            self.assertEqual(ctx.epoch_selector("2: Odor"), 2)
            self.assertEqual(ctx.epoch_window(0.1, 0.4), (0.1, 0.4))
            self.assertEqual(ctx.response_window(-1.0, 0.4), (-1.0, 0.4))
            with self.assertRaisesRegex(ValueError, "stop"):
                ctx.epoch_window(1.0, 1.0)
            with self.assertRaisesRegex(ValueError, "stop"):
                ctx.response_window(1.0, 1.0)

    def test_roi_selector_returns_visible_subset(self) -> None:
        """Confirm workflows can request all or visible ROIs through the API."""
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = write_converted_recording_files(
                root,
                movie_values=np.ones((3, 2, 2), dtype=np.float64),
            )
            roi_set = make_roi_set(
                np.array(
                    [
                        [[True, False], [False, False]],
                        [[False, True], [False, False]],
                        [[False, False], [True, False]],
                    ],
                    dtype=np.bool_,
                ),
                labels=("roi_0001", "roi_0002", "roi_0003"),
            )
            ctx = _custom_context(
                recording_path,
                roi_set=roi_set,
                visible_roi_indices=(0, 2),
            )

            all_rois = ctx.rois_for_selector("all_rois")
            visible_rois = ctx.rois_for_selector("visible_rois")

            self.assertEqual(all_rois.labels, roi_set.labels)
            self.assertEqual(visible_rois.labels, ("roi_0001", "roi_0003"))

    def test_matrix_csv_accepts_explicit_column_labels(self) -> None:
        """Confirm workflow matrix CSVs can carry scientific axis labels."""
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = write_converted_recording_files(root)
            ctx = _custom_context(recording_path)
            path = ctx.output_path("matrix.csv")

            ctx.write_matrix_csv(
                path,
                np.array([[1.0, 2.0]], dtype=np.float64),
                row_labels=("roi_0001",),
                column_labels=("lag_s_-0.100000", "lag_s_0.000000"),
            )

            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines()[0],
                "label,lag_s_-0.100000,lag_s_0.000000",
            )

    def test_response_kernel_output_names_and_lag_labels_are_stable(self) -> None:
        """Confirm kernel CSV outputs encode epochs and lag seconds."""
        self.assertEqual(
            _kernel_output_stem(
                "response_kernels",
                epoch_index=0,
                epoch_name="contrast/0.2",
                epoch_numbers=(2, 4),
            ),
            "response_kernels_group_01_epochs_2_4_contrast_0.2",
        )
        self.assertEqual(
            _kernel_output_stem(
                "response_kernels",
                epoch_index=1,
                epoch_name="contrast 0.2",
                epoch_numbers=(3,),
            ),
            "response_kernels_group_02_epochs_3_contrast_0.2",
        )
        self.assertEqual(
            _lag_column_labels(np.array([-0.02, -0.0, 0.02], dtype=np.float64)),
            ("lag_s_-0.020000", "lag_s_0.000000", "lag_s_0.020000"),
        )


class NativeDirectionSelectivityWorkflowTest(unittest.TestCase):
    """Tests for the packaged DSI custom workflow."""

    def test_absolute_threshold_filters_and_highlights_rois(self) -> None:
        """Confirm absolute DSI threshold controls plotting and row highlights."""
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            ctx = _DsiFakeContext(root)
            params = DirectionSelectivityParams(
                preferred_epoch="preferred",
                null_epoch="null",
                dsi_threshold=0.5,
            )

            result = run_direction_selectivity(cast(CustomRunContext, ctx), params)

            table_text = (root / "direction_selectivity.csv").read_text(
                encoding="utf-8",
            )
            self.assertIn("roi_0001,0.600", table_text)
            self.assertIn("roi_0002,-0.200", table_text)
            self.assertIn("roi_0003,-1.000", table_text)
            self.assertEqual(ctx.plotted_roi_indices, (0, 2))
            self.assertEqual(result.tables[0].highlighted_rows, (0, 2))
            self.assertIsNotNone(result.response_plot_data)

    def test_unchecked_rectification_uses_signed_response_sum(self) -> None:
        """Confirm unchecked rectification uses signed response sums."""
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            ctx = _DsiFakeContext(root)
            params = DirectionSelectivityParams(
                preferred_epoch="preferred",
                null_epoch="null",
                dsi_threshold=0.5,
                rectify_responses=False,
            )

            result = run_direction_selectivity(cast(CustomRunContext, ctx), params)

            table_text = (root / "direction_selectivity.csv").read_text(
                encoding="utf-8",
            )
            self.assertIn("roi_0001,0.600", table_text)
            self.assertIn("roi_0002,-0.200", table_text)
            self.assertIn("roi_0003,1.667", table_text)
            self.assertEqual(ctx.plotted_roi_indices, (0, 2))
            self.assertEqual(result.tables[0].highlighted_rows, (0, 2))

    def test_zero_sum_dsi_is_nan_and_does_not_pass_threshold(self) -> None:
        """Confirm zero-sum DSI is NaN and does not pass threshold."""
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            ctx = _DsiFakeContext(root, zero_sum_first_roi=True)
            params = DirectionSelectivityParams(
                preferred_epoch="preferred",
                null_epoch="null",
                dsi_threshold=0.5,
                rectify_responses=False,
            )

            result = run_direction_selectivity(cast(CustomRunContext, ctx), params)

            table_text = (root / "direction_selectivity.csv").read_text(
                encoding="utf-8",
            )
            self.assertIn("roi_0001,nan", table_text)
            self.assertEqual(ctx.plotted_roi_indices, (2,))
            self.assertEqual(result.tables[0].highlighted_rows, (2,))


class CustomWorkflowProvenanceTest(unittest.TestCase):
    """Tests workflow metadata saved beside outputs."""

    def test_writes_workflow_version_sidecar_for_files(self) -> None:
        """Confirm exported files receive workflow metadata sidecars."""
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            output_path = root / "direction_selectivity.csv"
            output_path.write_text("roi_label,dsi\nroi_0001,1.0\n", encoding="utf-8")
            provenance = CustomWorkflowProvenance(
                workflow_id="direction-selectivity",
                workflow_name="Direction selectivity",
                workflow_version="1.2",
                workflow_source_path=root / "dsi.py",
                workflow_source_hash="a" * 64,
                twopy_version="0.1.9",
                run_started_at="2026-05-18T00:00:00+00:00",
                parameters={"preferred_epoch": "right"},
                recording_path=root / "recording_data.h5",
            )

            written_paths = write_result_provenance(
                CustomResult(message="ok", files=(output_path,)),
                provenance,
            )

            sidecar = provenance_sidecar_path(output_path)
            self.assertEqual(written_paths, (sidecar,))
            loaded = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(loaded["workflow_version"], "1.2")
            self.assertEqual(loaded["workflow_source_hash"], "a" * 64)
            self.assertEqual(loaded["parameters"]["preferred_epoch"], "right")

    def test_rejects_result_files_outside_workflow_output_dir(self) -> None:
        """Confirm workflows cannot publish arbitrary filesystem paths."""
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "custom_outputs"
            outside_path = root / "outside.csv"
            outside_path.write_text("value\n1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "below ctx.output_dir"):
                validate_custom_result(
                    CustomResult(message="ok", files=(outside_path,)),
                    output_dir=output_dir,
                    expected_roi_shape=(2, 2),
                )

    def test_rejects_invalid_response_plot_data(self) -> None:
        """Confirm response plot data is checked before GUI rendering."""
        with temporary_directory() as temp_dir:
            plot_data = ResponsePlotData(
                source_path=Path(temp_dir) / "custom.h5",
                epochs=(
                    EpochResponsePlotData(
                        epoch_name="right",
                        epoch_number=1,
                        roi_labels=("roi_0001",),
                        time_seconds=np.array([0.0, 1.0]),
                        mean_values=np.array([[1.0]]),
                        sem_values=np.array([[0.1, 0.2]]),
                    ),
                ),
            )

            with self.assertRaisesRegex(ValueError, "mean_values"):
                validate_custom_result(
                    CustomResult(message="ok", response_plot_data=plot_data),
                    output_dir=Path(temp_dir) / "custom_outputs",
                    expected_roi_shape=(2, 2),
                )

    def test_rejects_fake_response_plot_data(self) -> None:
        """Confirm response plot output must use twopy's plot-data object."""
        with temporary_directory() as temp_dir:
            fake_plot_data = _FakeResponsePlotData(
                epochs=(
                    EpochResponsePlotData(
                        epoch_name="right",
                        epoch_number=1,
                        roi_labels=("roi_0001",),
                        time_seconds=np.array([0.0, 1.0]),
                        mean_values=np.array([[1.0, 2.0]]),
                        sem_values=np.array([[0.1, 0.2]]),
                    ),
                ),
            )

            with self.assertRaisesRegex(ValueError, "ResponsePlotData"):
                validate_custom_result(
                    CustomResult(
                        message="ok",
                        response_plot_data=cast(ResponsePlotData, fake_plot_data),
                    ),
                    output_dir=Path(temp_dir) / "custom_outputs",
                    expected_roi_shape=(2, 2),
                )


@dataclass(frozen=True)
class _FakeResponsePlotData:
    """Fake plot-data object with the right shape but wrong type."""

    epochs: tuple[EpochResponsePlotData, ...]


@dataclass(frozen=True)
class _DsiFakeComputation:
    """Tiny computation object carrying fake grouped responses."""

    grouped_responses: object


class _DsiFakeContext:
    """Minimal DSI context that records which ROIs were sent to plotting."""

    def __init__(self, root: Path, *, zero_sum_first_roi: bool = False) -> None:
        """Create fake ROI and output state under ``root``."""
        self._root = root
        self._zero_sum_first_roi = zero_sum_first_roi
        self._rois = make_roi_set(
            np.ones((3, 1, 1), dtype=np.bool_),
            labels=("roi_0001", "roi_0002", "roi_0003"),
        )
        self.plotted_roi_indices: tuple[int, ...] = ()

    def current_rois(self) -> RoiSet:
        """Return three fake ROIs."""
        return self._rois

    def rois_for_selector(self, selector: str) -> RoiSet:
        """Return fake ROIs for the standard DSI ROI selector."""
        if selector != "all_rois":
            raise ValueError(selector)
        return self._rois

    def compute_standard_responses(
        self,
        roi_set: RoiSet | None = None,
    ) -> _DsiFakeComputation:
        """Return fake grouped responses for DSI calls."""
        return _DsiFakeComputation(grouped_responses=object())

    def epoch_window(
        self,
        start_seconds: float,
        stop_seconds: float,
    ) -> tuple[float, float]:
        """Return the selected DSI metric window."""
        return float(start_seconds), float(stop_seconds)

    def epoch_metric(
        self,
        grouped: object,
        epoch: object,
        metric: object,
        *,
        window_seconds: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Return deterministic preferred and null responses."""
        if epoch == "preferred":
            return np.array([0.8, 0.2, -0.4], dtype=np.float64)
        if self._zero_sum_first_roi:
            return np.array([-0.8, 0.3, 0.1], dtype=np.float64)
        return np.array([0.2, 0.3, 0.1], dtype=np.float64)

    def output_path(self, filename: str | Path) -> Path:
        """Return an output path inside the temporary directory."""
        return self._root / filename

    def write_roi_table(
        self,
        path: Path,
        columns: Mapping[str, Sequence[object]],
        *,
        roi_labels: tuple[str, ...] | None = None,
    ) -> None:
        """Write the small ROI table needed by the DSI test."""
        labels = roi_labels if roi_labels is not None else self._rois.labels
        column_names = tuple(columns)
        rows = [",".join(("roi_label", *column_names))]
        for index, label in enumerate(labels):
            rows.append(
                ",".join(
                    (label, *(str(columns[name][index]) for name in column_names)),
                ),
            )
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def response_plot_data(
        self,
        grouped: object,
        *,
        source_path: Path | None = None,
        max_rois: int | None = None,
        roi_indices: Sequence[int] | None = None,
    ) -> ResponsePlotData:
        """Record ROI indices selected for plotting."""
        self.plotted_roi_indices = tuple(roi_indices or ())
        return ResponsePlotData(source_path=source_path, epochs=())


def _custom_context(
    recording_path: Path,
    *,
    roi_set: RoiSet | None = None,
    visible_roi_indices: tuple[int, ...] = (),
) -> CustomRunContext:
    """Return a minimal custom workflow context for one converted recording."""
    recording = load_converted_recording(recording_path)
    return CustomRunContext(
        recording=recording,
        roi_set=roi_set,
        output_dir=recording_path.parent / "custom_outputs",
        delta_f_over_f_options=DeltaFOverFOptions(),
        response_processing_options=ResponseProcessingOptions(),
        response_pre_window_seconds=2.0,
        response_post_window_seconds=0.0,
        provenance=CustomWorkflowProvenance(
            workflow_id="test",
            workflow_name="Test",
            workflow_version="1.0",
            workflow_source_path=recording_path.parent / "workflow.py",
            workflow_source_hash="a" * 64,
            twopy_version="0.1.9",
            run_started_at="2026-05-18T00:00:00+00:00",
            parameters={},
            recording_path=recording_path,
        ),
        visible_roi_indices=visible_roi_indices,
    )


def _timeline_photodiode() -> np.ndarray:
    """Return synthetic high-rate photodiode bounds for context tests."""
    values = np.zeros(120, dtype=np.float64)
    values[0:2] = 1.0
    values[80:116] = 1.0
    return values


if __name__ == "__main__":
    unittest.main()
