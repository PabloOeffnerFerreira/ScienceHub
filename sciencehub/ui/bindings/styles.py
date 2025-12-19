from PyQt6.QtWidgets import QApplication

def apply_stylesheet(path: str):
    with open(path, "r") as file:
        QApplication.instance().setStyleSheet(file.read())
