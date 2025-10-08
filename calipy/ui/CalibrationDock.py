# (c) 2019 MPI for Neurobiology of Behavior, Florian Franzen, Abhilash Cheekoti
# SPDX-License-Identifier: LGPL-2.1

from PyQt5.Qt import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QComboBox, QLabel
from PyQt5.QtWidgets import QWidget, QDockWidget, QVBoxLayout


class CalibrationDock(QDockWidget):
    display_calib_changed = pyqtSignal()

    def __init__(self, context):
        self.context = context

        # Setup widget
        super().__init__("Calibration")
        self.setFeatures(self.NoDockWidgetFeatures)
        self.widget = QWidget()

        # Display selection
        self.text_display_calib = QLabel(self)
        self.text_display_calib.setText("Display")

        self.combo_display_calib = QComboBox(self)
        self.combo_display_calib.addItems(["Single Calibration",
                                           "System Calibration"])
        self.combo_display_calib.currentIndexChanged.connect(self.on_display_calib_change)

        # Result stats
        self.table_calibrations = QTableWidget(0, 5, self)
        # Camera: Camera name/id
        # Inputs: Number of frames used in single calibrations and system calibration
        # Overall single err: Single camera calibration reprojection errors
        # Overall sys err: System camera calibration overall errors
        # Frame sys err: System camera calibration frame errors
        # Mean, median, max
        self.table_calibrations.setHorizontalHeaderLabels(["Camera", "Inputs", "Overall single err", "Overall sys err",
                                                           "Frame sys err"])

        # Setup layout
        main_layout = QVBoxLayout()

        main_layout.addWidget(self.text_display_calib)
        main_layout.addWidget(self.combo_display_calib)
        main_layout.addWidget(self.table_calibrations)

        self.widget.setLayout(main_layout)
        self.setWidget(self.widget)

    def set_calibration_table(self, row, column, value):
        item = QTableWidgetItem(value)
        item.setFlags(Qt.ItemIsEnabled)
        self.table_calibrations.setItem(row, column, item)

    def update_result(self, reset=False):
        stats = self.context.get_calibration_stats(reset=reset)
        self.table_calibrations.clearContents()
        self.table_calibrations.setRowCount(len(stats))

        for index, (id, result) in enumerate(stats.items()):
            self.set_calibration_table(index, 0, id)
            self.set_calibration_table(index, 1, f"{result['single_estimations']} / {result['multicam_estimations']}")
            self.set_calibration_table(index, 2, f"_ / _ / _")
            if 'system_errors' in result:
                self.set_calibration_table(index, 3, "{:.2f} / {:.2f} / {:.2f}".format(*result['system_errors']))
            if 'system_frame_errors' in result:
                self.set_calibration_table(index, 4, "{:.2f} / {:.2f} / {:.2f}".format(*result['system_frame_errors']))

    # Button Callbacks

    def on_display_calib_change(self):
        self.context.select_display_calib(self.combo_display_calib.currentIndex())
        self.display_calib_changed.emit()
