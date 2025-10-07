# (c) 2019 MPI for Neurobiology of Behavior, Florian Franzen, Abhilash Cheekoti
# SPDX-License-Identifier: LGPL-2.1

import logging
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R  # noqa

from calibcamlib.yaml_helper import collection_to_array
from calibcamlib import Camerasystem, Board, Detections

from calipy import VERSION
from .BaseContext import BaseContext

logger = logging.getLogger(__name__)


class CalibrationContext(BaseContext):
    """ Controller-style class to handle camera systems calibration """

    def __init__(self):
        super().__init__()

        # Current selection
        self.detector_index = 0
        self.model_index = 0
        self.display_calib_index = 0

        # Initialize results
        self.detections = {}  # session_id > Detections
        self.boards = {}  # session_id > cam_id > Board

        self.calibrations_single = {}  # session_id > 'cs': calibcamlib.CameraSystem
        self.estimations_single = {}  # session_id > cam_id > 'poses' > frm_idx > { rvec: vec3, tvec: vec3 }

        self.calibrations_multi = {}  # session_id > 'cs': calibcamlib.CameraSystem
                                    #  session_id > cam_id > 'errors' > {max, med, mean}

        # Assumed single source for each camera
        self.estimations_boards = {}  # session_id > cam_id > 'poses' > frm_idx > { rvec: vec3, tvec: vec3 }
                                    # session_id > cam_id > 'errors' > frm_idx > {max, med, mean}

        self.other = {}

    def get_available_subsets(self):
        """ Override available subsets to add calibration based subsets"""
        subsets = super().get_available_subsets()

        if self.session is None:
            return subsets

        # Add detections as subsets
        detections = self.get_current_detections()
        if not detections.is_empty():
            det_idxs = detections.to_array()['frame_idxs'].flatten()
            det_idxs = sorted(set(det_idxs))
            det_idxs.remove(-1) # -1 is placed in the array instead of nan!
            subsets['Detections'] = det_idxs

        # Add estimations as subsets
        estimations = self.get_current_estimations_single()
        est_idx = set()

        for cam_id in self.get_current_cam_ids():
            estimations_cam = estimations.get(cam_id, {})
            est_idx.update(estimations_cam.get('poses', {}).keys())

        if len(est_idx):
            subsets['Estimations'] = sorted(est_idx)

        return subsets

    def get_frame(self, cam_id):
        """ Override frame retrieval to draw calibration result """
        frame = super().get_frame(cam_id)
        if frame is None:
            return frame

        frame = frame.copy()
        sensor_offset = self.get_sensor_offset(cam_id)
        sensor_offset = np.asarray(sensor_offset)[np.newaxis, :]
        cam_idx = self.get_current_cam_ids().index(cam_id)

        # Make sure we draw in color by converting the frame to color first if necessary
        if frame.ndim < 3 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Draw detections
        detections = self.get_current_detections()
        if not detections.is_empty():
            detection = detections.get_frame_detections(self.frame_index, [cam_idx])
            cv2.aruco.drawDetectedCornersCharuco(frame,
                                                 np.asarray(detection, dtype=np.float32)
                                                 - np.asarray(sensor_offset))

        # Draw calibration result
        board = self.get_current_boards().get(cam_id, None)
        if board is None:
            return frame
        board_points = board.get_board_points()

        if self.display_calib_index == 0:
            calibration_cs = self.get_current_calibrations_single().get('cs', None)
            estimation = self.get_current_estimations_single().get(cam_id, {}).get('poses', {})
            estimation = estimation.get(self.frame_index, None)
        else:
            # TODO: check the order of transformations!
            calibration_cs = self.get_current_calibrations_multi().get('cs', None)
            estimation = self.get_current_estimations_boards().get(cam_id, {}).get('poses', {})
            estimation = estimation.get(self.frame_index, None)

        if calibration_cs is None or estimation is None:
            return frame

        coords_cam = R.from_rotvec(estimation['rvec']).apply(board_points) + estimation['tvec']
        img_points =  calibration_cs.project(coords_cam, offsets=sensor_offset, cam_idx=cam_idx)
        for point in img_points:
            if not np.all(np.isnan(point)):
                cv2.drawMarker(frame, (int(point[0]), int(point[1])), (0, 0, 255))

        return frame

    # Detector and detection management

    def get_current_boards(self):
        return self.boards.get(self.session.id, {})

    def get_current_detections(self):
        return self.detections.get(self.session.id, Detections())

    # Model and calibration management

    def select_display_calib(self, index):
        self.display_calib_index = index

    def get_current_calibrations_single(self):
        return self.calibrations_single.get(self.session.id, {})

    def get_current_estimations_single(self):
        return self.estimations_single.get(self.session.id, {})

    def get_current_calibrations_multi(self):
        return self.calibrations_multi.get(self.session.id, {})

    def get_current_estimations_boards(self):
        return self.estimations_boards.get(self.session.id, {})

    # Overall result management

    def get_calipy_calibcam_indexes(self, calibcam_rec_files: list[str]):
        """
        Matches the recordings loaded into the calipy gui with recordings in the calibcam file.
        :param calibcam_rec_files:
        :return: indic that match with the recordings loaded into the calipy gui
        """
        if not self.session:
            return []

        def get_path(path:str):
            if "\\" in path:
                return PureWindowsPath(path)
            else:
                return Path(path)

        # Generally, all the videos in a session have different names.
        # In case the videos have the same names, identification is set to be based on the dir containing the video.
        # Identifying the unique parts in the path to the videos, which will be used to match with the available videos.
        calibcam_rec_file_parts = [list(get_path(file).parts) for file in calibcam_rec_files]
        unique_idx = -1  # from left to right, the first part that is not unique is the unique part
        for unique_idx in range(1, len(calibcam_rec_file_parts[0])):
            part_list = [parts[-unique_idx] for parts in calibcam_rec_file_parts]
            if len(part_list) == len(set(part_list)):
                logger.log(logging.INFO, f"Unique parts in file names: {part_list}")
                unique_idx *= -1
                break
        calibcam_rec_unique_names = [parts[unique_idx] for parts in calibcam_rec_file_parts]

        calipy_rec_unique_names = []
        for cam_id, rec in self.session.recordings.items():
            calipy_rec_unique_names.append(get_path(rec.url).parts[unique_idx])

        calibcam_indexes = [calibcam_rec_unique_names.index(name) for name in calipy_rec_unique_names
                                   if name in calibcam_rec_unique_names]

        return calibcam_indexes

    def load_detections(self, detection_files: list[str]):
        if self.session is None:
            return
        # It is assumed that files are provided in the order of existing/loaded recordings
        assert len(detection_files) == len(self.session.recordings), (f"Total number of cameras in "
                                                                      f"detections: {len(detection_files)} "
                                                                      f"does not match available number of recordings!")
        detections = Detections.from_file(detection_files)
        self.detections[self.session.id] = detections

    def load_calibrations_single(self, calibrations_single: list[dict]):
        if self.session is None:
            return
        # It is assumed that calibs are provided in the order of existing/loaded recordings
        assert len(calibrations_single) == len(self.session.recordings), (f"Total number of single camera "
                                                                               f"calibrations: {len(calibrations_single)} "
                                                                               "does not match available number of recordings!")
        sess_id = self.session.id
        sin_calib_cs = Camerasystem.from_calibs(collection_to_array(calibrations_single))
        self.calibrations_single[sess_id] = {
            'cs': sin_calib_cs,
        }

        self.estimations_single[sess_id] = {}
        for cam_id, calib_dict in zip(self.get_current_cam_ids(), calibrations_single):
            poses = {}
            for index, frame_idx in enumerate(calib_dict['frame_idxs']):
                poses[frame_idx] = {
                    'rvec': calib_dict['rvecs'][index],
                    'tvec': calib_dict['tvecs'][index],
                }
            self.estimations_single[sess_id][cam_id] = {
                'poses': poses,
            }

    def load_calibration_multicam(self, calibcam_dict: dict, boards_dict: dict, calibcam_cam_indexes: list | None = None):
        """ Read calibration info from calibcam dict, only if the corresponding recordings are already loaded """
        if not self.session:
            return

        logger.log(logging.INFO,
                   f"Current software version: {VERSION}, Calibcam file version: {calibcam_dict.get('version', None)}.")

        if calibcam_cam_indexes is None:
            calibcam_cam_indexes = self.get_calipy_calibcam_indexes(calibcam_dict['info']['rec_file_names'])

        calibcam_board_params = calibcam_dict['info']['board_params']
        calibcam_calibs = calibcam_dict['calibs']

        boards_dict = collection_to_array(boards_dict)
        board_rvecs = boards_dict['rvecs']
        board_tvecs = boards_dict['tvecs']
        frame_idxs_cams = boards_dict['frame_idxs']

        # Error from final calibration
        final_err_shape = (*frame_idxs_cams.shape,
                           len(Board(calibcam_board_params[0]).get_corner_ids()), 2)
        if 'fun_final' in boards_dict['info']:
            final_err = np.asarray(boards_dict['info']['fun_final']).reshape(final_err_shape)
            final_err = np.abs(final_err)
        else:
            final_err = np.empty(final_err_shape)
            final_err[:] = np.nan

        sess_id = self.session.id
        calibs = []
        self.boards[sess_id] = {}
        self.calibrations_multi[sess_id] = {
            "cs": None,
        }
        self.estimations_boards[sess_id] = {}

        # Set data

        for cam_id, calibcam_cam_idx in zip(self.session.recordings.keys(), calibcam_cam_indexes):

            calibs.append(calibcam_calibs[calibcam_cam_idx])

            self.boards[sess_id][cam_id] = Board(calibcam_board_params[calibcam_cam_idx])

            self.calibrations_multi[sess_id][cam_id] = {}
            self.calibrations_multi[sess_id][cam_id]['errors'] = {
                'max': np.nanmax(final_err[calibcam_cam_idx]),
                'med': np.nanmedian(final_err[calibcam_cam_idx]),
                'mean': np.nanmean(final_err[calibcam_cam_idx]),
            }

            frame_idxs_cam = frame_idxs_cams[calibcam_cam_idx]
            self.estimations_boards[sess_id][cam_id] = {'errors': {}, 'poses': {}}
            for index, frm_idx in enumerate(frame_idxs_cam):
                if frm_idx != -1:
                    self.estimations_boards[sess_id][cam_id]['poses'][frm_idx] = {
                        'rvec': board_rvecs[index],
                        'tvec': board_tvecs[index],
                    }
                    self.estimations_boards[sess_id][cam_id]['errors'][frm_idx] = {
                        'max': np.nanmax(
                              final_err[calibcam_cam_idx, index]),
                        'med': np.nanmedian(
                              final_err[calibcam_cam_idx, index]),
                        'mean': np.nanmean(
                              final_err[calibcam_cam_idx, index])
                    }

        # Multi camera calibration
        multicam_cs = Camerasystem.from_calibs(collection_to_array(calibs))
        self.calibrations_multi[sess_id]["cs"] = multicam_cs

    def clear_result(self):
        self.detections.clear()

        self.calibrations_single.clear()
        self.estimations_single.clear()

        self.calibrations_multi.clear()
        self.estimations_boards.clear()

    # Results statistics

    def get_detection_stats(self):
        stats = {}

        detections = self.get_current_detections()
        if not detections.is_empty():
            num_detected_markers = detections.get_n_detections_markers()

            for cam_id, ndr in zip(self.get_current_cam_ids(), num_detected_markers):

                # Count detections and markers
                detected_frames = np.sum(ndr > 0)
                markers = np.sum(ndr)

                stats[cam_id] = (detected_frames, markers)

        return stats

    def get_calibration_stats(self):
        stats = {}

        det_stats = self.get_detection_stats()

        # calibrations = self.get_current_calibrations_single()
        estimations = self.get_current_estimations_single()

        calibrations_multi = self.get_current_calibrations_multi()
        estimations_board = self.get_current_estimations_boards()

        for cam_id in self.get_current_cam_ids():

            count_det = det_stats.get(cam_id, (0, 0))[0]
            estimations_cam = estimations.get(cam_id, {})
            count_est = len(estimations_cam.get('poses', {}))

            stats[cam_id] = {
                'detections': count_det,
                'single_estimations': count_est,
            }
            if 'errors' in calibrations_multi.get(cam_id, {}):
                stats[cam_id].update({'system_errors': (calibrations_multi[cam_id]['errors']['mean'],
                                                        calibrations_multi[cam_id]['errors']['med'],
                                                        calibrations_multi[cam_id]['errors']['max'])
                                      })

            estimations_board_cam = estimations_board.get(cam_id, {})
            if self.frame_index in estimations_board_cam.get('errors', {}):
                stats[cam_id].update({'system_frame_errors': (estimations_board_cam['errors'][self.frame_index]['mean'],
                                                              estimations_board_cam['errors'][self.frame_index]['med'],
                                                              estimations_board_cam['errors'][self.frame_index]['max'])
                                      })
        return stats

    def plot_system_calibration_errors(self):

        estimations_board = self.get_current_estimations_boards()

        fig, axs = plt.subplots(len(estimations_board.keys()), sharex=True)
        if not isinstance(axs, np.ndarray):
            axs = [axs]
        for i, (cam_id, estimations_cam) in enumerate(estimations_board.items()):
            estimations_cam = estimations_cam['errors']
            frames_cam = []
            errors_cam = [[] for _ in range(3)]
            for frame_idx, estimation in estimations_cam.items():
                if 'med' in estimation:
                    frames_cam.append(frame_idx)
                    errors_cam[0].append(estimation['mean'])
                    errors_cam[1].append(estimation['med'])
                    errors_cam[2].append(estimation['max'])

            axs[i].plot(frames_cam, errors_cam[0], '*-', label='mean')
            axs[i].plot(frames_cam, errors_cam[1], '*-', label='median')
            axs[i].plot(frames_cam, errors_cam[2], '*-', label='max')
            axs[i].set_title(cam_id)
            axs[i].legend()

        plt.show()
