# (c) 2019 MPI for Neurobiology of Behavior, Florian Franzen, Abhilash Cheekoti
# SPDX-License-Identifier: LGPL-2.1

import logging
from pathlib import Path

import numpy as np
from PyQt5.Qt import Qt, QIcon
from PyQt5.QtWidgets import QMainWindow, QMdiArea, QFileDialog, QMessageBox

from calipy import ui

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):

    def __init__(self, context):
        self.context = context

        # Setup widget
        super().__init__()
        self.mdi = QMdiArea()
        self.setCentralWidget(self.mdi)
        self.subwindows = {}
        self.error_table = None
        self.table_thread = None

        # Setup menu bar
        session_menu = self.menuBar().addMenu("&File")
        session_menu.addAction(QIcon.fromTheme("document-open"), "&Open...", self.on_system_open)
        session_menu.addAction("&Save...", self.on_system_save)
        session_menu.addSeparator()
        session_menu.addAction("&Clear", self.on_system_clear)
        session_menu.addSeparator()
        session_menu.addAction("&Quit", self.on_quit)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction("&Tile", self.mdi.tileSubWindows)
        view_menu.addAction("&Cascade", self.mdi.cascadeSubWindows)

        result_menu = self.menuBar().addMenu("&Result")
        result_menu.addAction("&Load Calib", self.on_load_calib)
        result_menu.addSeparator()
        result_menu.addAction("&Show multicam errors table", self.on_show_error_table)
        result_menu.addAction("&Plot multicam errors", self.context.plot_system_calibration_errors)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("&About", self.on_about)

        # Setup docks
        self.dock_time = ui.TimelineDock(context)
        self.dock_time.time_index_changed.connect(self.on_timeline_change)
        self.dock_time.subset_changed.connect(self.on_timeline_change)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_time)

        self.dock_sessions = ui.SourcesDock(context)
        self.dock_sessions.sources_modified.connect(self.sync_subwindows_cameras)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_sessions)

        self.dock_detection = ui.DetectionDock(context)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_detection)

        self.dock_calibration = ui.CalibrationDock(context)
        self.dock_calibration.display_calib_changed.connect(self.update_subwindows)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_calibration)

    def open(self, file):
        """Open specified system file in UI"""

        if file.endswith(".system.yml"):
            self.context.load(file)
        else:
            logger.log(logging.WARNING, f"{file}: unrecognised file!")

        self.setWindowTitle(file)
        self.dock_sessions.update_sources()
        self.dock_time.update_subsets()

        self.sync_subwindows_cameras()

    def open_videos(self, videos, pipelines=None):
        self.context.add_session()
        for i, rec in enumerate(videos):
            pipeline = None
            if pipelines is not None:
                pipeline = pipelines[i] if len(pipelines[i]) else None
            self.context.add_recording(rec, pipeline=pipeline)

        self.dock_sessions.update_sources()
        self.dock_time.update_subsets()

        self.sync_subwindows_cameras()

    def on_cameras_change(self):
        """ Helper to update UI on camera changes """
        self.dock_sessions.update_sources()

    def on_timeline_change(self):
        self.update_subwindows()
        self.dock_calibration.update_result()

    def on_calib_model_change(self):
        self.update_timeline_dock()
        self.update_subwindows()

    def sync_subwindows_cameras(self):
        """ Create or destroy windows based on available cameras """
        win_ids = list(self.subwindows.keys())
        cam_ids = self.context.get_current_cam_ids()

        # Close obsolete views
        for id in win_ids:
            if id not in cam_ids:
                self.subwindows[id].subwindow.close()
                del self.subwindows[id]

        # Open missing windows
        for id in cam_ids:
            if id not in win_ids:
                window = ui.FrameWindow(self.context, id)
                self.mdi.addSubWindow(window.subwindow)
                self.subwindows[id] = window
                window.show()

        # Update time control
        self.update_timeline_dock()

        # Reload current frame
        self.update_subwindows()
        self.mdi.tileSubWindows()

        # Update list of detections and calibrations (e.g., on session select)
        self.dock_detection.update_result()
        self.dock_calibration.update_result(reset=True)

    def update_subwindows(self):
        """ Update current frame on all subwindows """
        if self.context.session is None:
            return

        for sub in self.subwindows.values():
            sub.update_frame()

    def update_subwindow(self, id):
        """ Update current frame on specific subwindow """
        if self.context.session is None:
            return

        if id in self.subwindows:
            self.subwindows[id].update_frame()

    def update_timeline_dock(self):
        """ Update the timeline dock """
        if self.context.session is None:
            return

        self.dock_time.update_slider()
        self.dock_time.update_subsets()

    # File Menu Callbacks

    def on_system_open(self):
        """ MenuBar > Camera System > Open ..."""
        file = QFileDialog.getOpenFileName(self, "Open Camera System Config", "", "Session File (*.system.yml)")[0]

        if file:
            self.open(file)

    def on_system_save(self):
        """ MenuBar > Camera System > Save ..."""
        file = QFileDialog.getSaveFileName(self, "Save Camera System Config", "", "Session File (*.system.yml)")[0]

        if file:
            if not file.endswith(".system.yml"):
                file += ".system.yml"
            self.context.save(file)

    def on_system_clear(self):
        """ MenuBar > Camera System > Clear """
        if QMessageBox.question(self, "Clear Session?", "All unsaved changes will be lost!") == QMessageBox.Yes:
            self.context.clear()
            self.dock_sessions.update_sources()
            self.dock_time.update_subsets()
            self.sync_subwindows_cameras()

    def on_quit(self):
        """ MenuBar > Camera System > Quit """
        self.close()

    # Result Menu Callbacks

    def on_load_calib(self, file: str = None, load_recordings=False, load_detections=True, load_calibrations_single=True):
        """ MenuBar > Result > Load Calib.
        Loads multicamera calibration from calibcam. Also supports other calibration results."""

        if file is None:
            file = QFileDialog.getOpenFileName(self, "Load Calibcam Result", "", "Result File ( *.yml, *.npy)")[0]

        if file:
            file = Path(file)
            if file.is_dir():
                result_dir = Path(str(file))
                file = file / 'multicam_calibration.yml'
            else:
                result_dir = file.parent

            logger.log(logging.INFO, f"Loading Multicamera Calibration: {file}")
            if file.suffix == '.npy':
                calib_dict = np.load(file, allow_pickle=True)[()]
            elif file.suffix == '.yml':
                calib_dict = self.context.read_yml_files([str(file)])[0]
            else:
                logger.warning(f"Unknown file type: {file}")
                return
            assert calib_dict.get('version', 0) >= 4, "Calibration file version mismatch!"

            if load_recordings:
                self.open_videos(videos=calib_dict['info']['rec_file_names'],
                                 pipelines=calib_dict.get('rec_pipelines', None))
                calibcam_cam_indexes = range(len(calib_dict['info']['rec_file_names']))
            else:
                calibcam_cam_indexes = self.context.get_calipy_calibcam_indexes(calib_dict['info']['rec_file_names'])

            if load_detections:
                if isinstance(calib_dict['info']['opts']['detection'], bool):
                    det_files = [result_dir / f"detection_{i_cam:03d}.yml"
                                 for i_cam, _ in enumerate(self.context.get_current_cam_ids())]
                else:
                    det_files = [result_dir / det for det in calib_dict['info']['opts']['detection']]

                logger.log(logging.INFO, f"LOADING DETECTIONS")
                for i in calibcam_cam_indexes:
                    logger.log(logging.INFO, f"Loading file: {det_files[i]}")
                self.context.load_detections([det_files[i] for i in calibcam_cam_indexes]) # Detection object can read data from files

            if load_calibrations_single:
                if isinstance(calib_dict['info']['opts']['calibration_single'], bool):
                    sin_calib_files = [result_dir / f"calibration_single_{i_cam:03d}.yml"
                                       for i_cam, _ in enumerate(self.context.get_current_cam_ids())]
                else:
                    sin_calib_files = [str(result_dir / sc) for sc in calib_dict['info']['opts']['calibration_single']]

                logger.log(logging.INFO, f"LOADING SINGLE CALIBRATIONS")
                for i in calibcam_cam_indexes:
                    logger.log(logging.INFO, f"Loading file: {sin_calib_files[i]}")
                self.context.load_calibrations_single(self.context.read_yml_files(
                    [sin_calib_files[i] for i in calibcam_cam_indexes])
                )

            boards_file = str(result_dir / 'multicam_calibration_board_positions.yml')
            logger.log(logging.INFO, f"Loading Board Poses: {boards_file}")
            self.context.load_calibration_multicam(calibcam_dict=calib_dict,
                                                   boards_dict=self.context.read_yml_files([boards_file])[0],
                                                   calibcam_cam_indexes=calibcam_cam_indexes)

            self.dock_detection.update_result()
            self.dock_calibration.update_result(reset=True)
            self.dock_time.update_subsets()

            self.update_subwindows()

    def on_show_error_table(self):
        """ MenuBar > Result > Show multicam errors table """
        error_data = self.context.get_errors_as_lods()

        self.error_table = ui.ErrorTableWindow()
        self.error_table.update_table(error_data)
        self.error_table.frame_select_signal.connect(self.dock_time.on_index_change)
        self.error_table.show()

    # Help menu

    def on_about(self):
        QMessageBox.about(self, "About",
                          "(c) 2019 Florian Franzen, Abhilash Cheekoti, MPI for Neurobiology of Behavior, Bonn\nLicensed under LGPL 2.1")
