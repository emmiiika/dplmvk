import sys
from PySide6 import QtWidgets
from UI import Window, GREEN, ENDC

if __name__ == "__main__":
    # Initialize and run the Qt application
    app = QtWidgets.QApplication([])

    widget = Window()
    widget.resize(1000, 600)
    widget.show()

    print(f"{GREEN}✓{ENDC} Application initialized successfully.")

    sys.exit(app.exec())
