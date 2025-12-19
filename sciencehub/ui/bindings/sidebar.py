from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QComboBox
)
from sciencehub.main.tool_registry import get_categories, get_tools_for_category
from PyQt6.QtCore import Qt

class Sidebar(QWidget):
    def __init__(self, on_open_tool):
        super().__init__()
        self.setObjectName("sidebar")
        self.on_open_tool = on_open_tool
        self.category_buttons = {}
        self._build_ui()
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def _build_ui(self):
        self.setObjectName("sidebar")
        self.setFixedWidth(200)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(5)

        self.setLayout(sidebar_layout)

        self.tools_combo = QComboBox()
        self.tools_combo.addItem("Select a category above")

        categories = get_categories()
        for category in categories:
            button = QPushButton(category)
            button.setCheckable(True)
            button.clicked.connect(
                lambda _, c=category: self._on_category_selected(c)
            )
            sidebar_layout.addWidget(button)
            self.category_buttons[category] = button

        sidebar_layout.addStretch()
        sidebar_layout.addWidget(self.tools_combo)
        open_button = QPushButton("Open")
        open_button.setObjectName("openButton")
        open_button.clicked.connect(self._on_open_clicked)
        sidebar_layout.addWidget(open_button)

        if categories:
            self.category_buttons[categories[0]].click()

    def _on_category_selected(self, category):
        for cat, button in self.category_buttons.items():
            button.setChecked(cat == category)

        self.tools_combo.clear()
        self.tools_combo.addItems(get_tools_for_category(category))

    def _on_open_clicked(self):
        tool = self.tools_combo.currentText()
        if tool and tool != "Select a category above":
            self.on_open_tool(tool)
