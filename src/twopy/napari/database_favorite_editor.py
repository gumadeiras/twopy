"""Qt dialog for editing one database-search favorite.

Inputs: one saved favorite from the Search database window.
Outputs: edited favorite name and filter values for the caller to validate and
persist.

The dialog only owns widgets. Favorite validation, duplicate handling, and YAML
persistence stay in ``twopy.napari.database_favorites`` and the search dialog
workflow.
"""

from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from twopy.database.search import ExperimentSearchFilters
from twopy.napari.database_favorites import (
    ExperimentSearchFavorite,
    normalized_database_search_favorite,
)
from twopy.napari.theme import apply_twopy_theme

__all__ = ["ExperimentFavoriteEditDialog"]


class ExperimentFavoriteEditDialog(QDialog):
    """Dialog that lets users edit a favorite name and filters.

    Args:
        favorite: Favorite to show in editable fields.
        parent: Optional Qt parent widget.

    Outputs:
        Modal dialog with text fields for the favorite display name and every
        saved database-search filter.
    """

    def __init__(
        self,
        favorite: ExperimentSearchFavorite,
        *,
        parent: QWidget | None = None,
    ) -> None:
        """Create a favorite editor dialog."""
        super().__init__(parent)
        normalized = normalized_database_search_favorite(favorite)
        filters = normalized.filters

        self._name = QLineEdit(normalized.name)
        self._user = QLineEdit(filters.user or "")
        self._cell_type = QLineEdit(filters.cell_type or "")
        self._sensor = QLineEdit(filters.sensor or "")
        self._stimulus = QLineEdit(filters.stimulus or "")
        self._date = QLineEdit(filters.date or "")

        self.setWindowTitle("Edit Favorite")
        self.resize(520, self.sizeHint().height())

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow("Favorite name", self._name)
        form.addRow("User", self._user)
        form.addRow("Cell type", self._cell_type)
        form.addRow("Sensor", self._sensor)
        form.addRow("Stimulus", self._stimulus)
        form.addRow("Date", self._date)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)
        apply_twopy_theme(self, name="twopy_database_favorite_editor")

    def favorite(self) -> ExperimentSearchFavorite:
        """Return the favorite currently shown in the editor fields.

        Args:
            None.

        Returns:
            Favorite values for the caller to validate and persist.

        The dialog does not normalize or reject values here, so the caller can
        report validation errors through the same favorite-error path used by
        saving and removing favorites.
        """
        return ExperimentSearchFavorite(
            name=self._name.text(),
            filters=ExperimentSearchFilters(
                user=self._user.text(),
                cell_type=self._cell_type.text(),
                sensor=self._sensor.text(),
                stimulus=self._stimulus.text(),
                date=self._date.text(),
            ),
        )
