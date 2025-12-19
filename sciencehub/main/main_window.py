from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QMessageBox

from sciencehub.ui.bindings.styles import apply_stylesheet
from sciencehub.ui.bindings.menu_bar import create_menu_bar
from sciencehub.ui.bindings.sidebar import Sidebar
from sciencehub.ui.bindings.main_area import MainArea


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Science Hub")
        self.setGeometry(100, 100, 800, 600)

        apply_stylesheet("sciencehub/ui/bindings/styles.css")
        create_menu_bar(self)

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        layout = QHBoxLayout(central)

        sidebar = Sidebar(self._open_tool)
        main_area = MainArea()

        layout.addWidget(sidebar)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        layout.addWidget(main_area)

        self.setCentralWidget(central)

    def _open_tool(self, tool_name):
        QMessageBox.information(self, "Tool Selected", f"Opening: {tool_name}")
