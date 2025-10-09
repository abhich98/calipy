# (c) 2019 MPI for Neurobiology of Behavior, Florian Franzen, Abhilash Cheekoti
# SPDX-License-Identifier: LGPL-2.1

import datetime
from math import isinf
import logging

from PyQt5.Qt import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDockWidget, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QSlider, QComboBox, QSpinBox, QLabel


logger = logging.getLogger(__name__)


class TimelineDock(QDockWidget):
    time_index_changed = pyqtSignal()
    subset_changed = pyqtSignal()

    def __init__(self, context):
        self.context = context
        self.current_subset = None
        self.subsets = context.get_available_subsets()
        self._updating_dock = False

        # Set up widget
        super().__init__("Timeline")
        self.setFeatures(self.NoDockWidgetFeatures)

        # Init time slider
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setRange(0, 0)
        self.slider.setTracking(False)
        self.slider.valueChanged.connect(self.on_index_change)

        # Init label
        self.label_left = QLabel("0")
        self.label_right = QLabel("0")

        # Init center combo box
        self.box_subset = QComboBox()
        self.box_subset.currentIndexChanged.connect(self.on_subset_change)

        # Init center spin box
        self.box_current = QSpinBox()
        self.box_current.setRange(0, 0)
        self.box_current.setKeyboardTracking(False)
        self.box_current.valueChanged.connect(self.on_index_change)

        self.label_current = QLabel("")

        # Layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.slider)

        label_layout = QHBoxLayout()
        label_layout.addWidget(self.label_left)
        label_layout.addStretch()
        label_layout.addWidget(self.box_subset)
        label_layout.addWidget(self.box_current)
        label_layout.addWidget(self.label_current)
        label_layout.addStretch()
        label_layout.addWidget(self.label_right)

        main_layout.addLayout(label_layout)

        widget = QWidget(self)
        widget.setLayout(main_layout)
        self.setWidget(widget)

    def update_slider(self):
        """Update slider based on current document state"""
        if self.current_subset is None:
            maximum = self.context.get_length() - 1
        else:
            maximum = len(self.current_subset) - 1

        if isinf(maximum):
            maximum = -1

        self._updating_dock = True
        self.slider.setRange(0, maximum)
        self.box_current.setRange(0, maximum)
        self.label_right.setText(f"{maximum}")
        self._updating_dock = False

        index = self.context.get_current_frame()

        if self.current_subset is not None:
            index = self.current_subset.index(index)

        self.update_index(index)

    def update_index(self, value: int):
        """ Update current frame index in center """

        # To avoid the trigger of on_index_change again
        self._updating_dock = True
        self.slider.setValue(value)
        self.box_current.setValue(value)
        self._updating_dock = False

        if self.current_subset is None:
            current_time = value / self.context.get_fps() if self.context.get_fps() else 0
            current_time = str(datetime.timedelta(seconds=current_time)).ljust(11, '0')
            self.label_current.setText(f"{current_time[:11]}")
        else:
            current_frame = self.current_subset[value]
            current_time = str(datetime.timedelta(seconds=current_frame / self.context.get_fps())).ljust(11, '0')
            self.label_current.setText(f"{current_time[:11]} ({current_frame:d})")

    def update_subsets(self):
        """Update available frame subsets"""
        self.subsets = self.context.get_available_subsets()
        self.box_subset.clear()
        self.box_subset.addItems(list(self.subsets.keys()))

    # UI callbacks

    def on_index_change(self, value: int):
        if not self._updating_dock:
            # Remap index if subset is set
            if self.current_subset is not None:
                if value < len(self.current_subset):
                    frame_idx = self.current_subset[value]
                else:
                    logger.log(logging.WARNING, f"{value} is out of range!")
                    return
            else:
                frame_idx = value

            self.update_index(value)   # Sync center ui
            self.context.set_current_frame(frame_idx)

            # Update frame views
            self.time_index_changed.emit()

    def on_subset_change(self, value):
        # Ignore empty selection
        if value < 0:
            return

        if not self._updating_dock:
            sub_id = self.box_subset.currentText()
            self.current_subset = self.subsets[sub_id]

            # Map indices between subsets
            if self.current_subset is not None:
                frm_idx = self.context.get_current_frame()
                if frm_idx in self.current_subset:
                    index = self.current_subset.index(frm_idx)
                else:
                    self.context.set_current_frame(self.current_subset[0])
                    index = 0

                self.update_index(index)

            self.update_slider()
            self.subset_changed.emit()
