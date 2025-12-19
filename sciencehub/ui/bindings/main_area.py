from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt


class MainArea(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        label = QLabel("Welcome to Science Hub!")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
