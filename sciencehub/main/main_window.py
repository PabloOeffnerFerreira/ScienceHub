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

        self.sidebar = Sidebar(self._open_tool)
        self.main_area = MainArea()

        layout.addWidget(self.sidebar)
        layout.addWidget(self.main_area)

        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        self.setCentralWidget(central)


    def _open_tool(self, tool_name: str):
        try:
            category = next(
                cat for cat, btn in self.sidebar.category_buttons.items() if btn.isChecked()
            )

            module_name = tool_name
            category_folder = category.lower().replace(" ", "_")

            module_path = f"sciencehub.domains.{category_folder}.{module_name}"

            module = __import__(module_path, fromlist=["create_tool"])

            tool_widget = module.create_tool(self.main_area)
            self.main_area.set_tool(tool_widget)

        except Exception as e:
            QMessageBox.warning(
                self,
                "Failed to open tool",
                f"Could not load tool '{tool_name}'.\n\n{e}"
            )
