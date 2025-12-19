import sys
from PyQt6.QtWidgets import QApplication
from sciencehub.main.main_window import MainWindow
from sciencehub.ui.bindings.styles import apply_stylesheet

def main():
    app = QApplication(sys.argv)

    apply_stylesheet("sciencehub/ui/bindings/styles.css")  # 👈 HERE

    window = MainWindow()
    window.showFullScreen()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
