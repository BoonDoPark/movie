from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QCheckBox


class QCheckBoxUtils:
    @staticmethod
    def check_box_aligned_center():
        widget = QWidget()
        layout = QHBoxLayout()
        check_box = QCheckBox()
        layout.setAlignment(Qt.AlignCenter)
        layout.addWidget(check_box)
        widget.setLayout(layout)
        return widget, check_box

    @staticmethod
    def get_check_box_from_widget(widget: QWidget):
        components = widget.children()
        check_box = None
        for component in components:
            if type(component) == QCheckBox:
                check_box = component
        return check_box
