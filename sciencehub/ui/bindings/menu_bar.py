from PyQt6.QtGui import QAction

def create_menu_bar(window):
    menu_bar = window.menuBar()

    file_menu = menu_bar.addMenu("&File")

    open_action = QAction("&Open", window)
    save_action = QAction("&Save", window)
    exit_action = QAction("&Exit", window)
    exit_action.triggered.connect(window.close)

    file_menu.addAction(open_action)
    file_menu.addAction(save_action)
    file_menu.addSeparator()
    file_menu.addAction(exit_action)

    help_menu = menu_bar.addMenu("&Help")
    help_menu.addAction(QAction("&About", window))
