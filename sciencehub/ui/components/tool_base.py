from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt


class ScienceHubTool(QWidget):
    """
    Base class for all ScienceHub tools.
    Handles spacing, margins, and future shared behavior.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(20, 20, 20, 20)
        self.root_layout.setSpacing(20)

        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
