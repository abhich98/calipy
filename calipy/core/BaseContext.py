# (c) 2019 MPI for Neurobiology of Behavior, Florian Franzen, Abhilash Cheekoti
# SPDX-License-Identifier: LGPL-2.1
import logging
from typing import Dict
import yaml
import multiprocessing
from functools import partial

from calipy import metaio


logger = logging.getLogger(__name__)

try:
    from yaml import CSafeLoader as yaml_loader
except ImportError:
    logger.warning("Using pure Python YAML loader. This might be slow!")
    from yaml import SafeLoader as yaml_loader


def read_yml(file, loader=yaml_loader):
    with open(file, 'r') as f:
        return yaml.load(f, Loader=loader)


class BaseContext:
    """ Controller-style class to handle camera systems management """
    vid_readers: Dict

    def __init__(self):
        self.system = metaio.CalipySystem()
        self.session = None
        self.frame_index = 0

        self.vid_readers = {}

        self.subset = None

    # Camera System

    def load(self, path):
        """ Load current camera system from file """
        self.system = metaio.CalipySystem.load(path)

        if self.system.sessions:
            self.select_session(0)

    def save(self, path):
        """ Save current camera system to file """
        self.system.save(path)

    def clear(self):
        """ Clear current state """
        self.__init__()

    def close(self):
        """ Close all open files """
        for reader in self.vid_readers.values():
            reader.close()

    # Sessions

    def get_session(self, index):
        """ Get session of specific index """
        return self.system.sessions[index]

    def get_sessions(self):
        """ Get all available sessions """
        return self.system.sessions

    def add_session(self):
        """ Add a new session """
        self.vid_readers.clear()

        self.session = self.system.add_session()

    def select_session(self, index):
        """ Select session by index """
        self.vid_readers.clear()

        self.session = self.system.sessions[index]

        for id, rec in self.session.recordings.items():
            self.vid_readers[id] = rec.init_reader()

    def remove_session(self, index):
        """ Remove session by index """
        if self.session == self.system.sessions[index]:
            self.vid_readers.clear()
            self.session = None

        self.system.remove_session(index)

    # Recordings

    def add_recording(self, path, pipeline=None, id_str=None):
        """ Add recording to current session """
        if not self.session:
            return

        if id_str is None:
            id_str = f"cam_{len(self.vid_readers)}"

        if id_str in self.vid_readers:
            del self.vid_readers[id_str]

        rec = self.session.add_recording(id_str, path, None, pipeline=pipeline)
        self.vid_readers[id_str] = rec.init_reader()

    def get_current_cam_ids(self):
        """ Get all available camera ids """
        if not self.session:
            return []

        return list(self.session.recordings.keys())

    def remove_recording(self, id):
        """ Remove recording from current session """
        if not self.session:
            return

        if id in self.vid_readers:
            del self.vid_readers[id]

        self.session.remove_recording(id)

    # Frames

    def get_length(self):
        """ Get frame count of current session or subset """
        if not self.session or not self.vid_readers:
            return 0

        # Find minimum length
        return min([reader.n_frames for reader in self.vid_readers.values()])

    def get_fps(self):
        """ Get frames per second for the session """
        if not self.session or not self.vid_readers:
            return 0

        if self.session.fps is None:
            def most_common(lst):
                return max(set(lst), key=lst.count)

            fps_list = []
            for id, reader in self.vid_readers.items():
                fps = reader.get_meta_data().get('fps', None)
                if fps is None:
                    logger.log(logging.WARNING, f"Could not find 'fps' for the video: "
                                                f"{self.session.recordings[id].url}, returning 60")
                    fps = 60
                fps_list.append(fps)

            # Find most common fps
            self.session.fps = most_common(fps_list)

        return self.session.fps

    def set_current_frame(self, index):
        """ Set the current frame index """
        self.frame_index = index

    def get_current_frame(self):
        """ Get current frame index """
        return self.frame_index

    def get_frame(self, cam_id):
        """ Get current frame by camera id"""
        # Abort if there is no recording for camera
        if cam_id not in self.vid_readers:
            return None

        # Return frame at current index
        return self.vid_readers[cam_id].get_data(self.frame_index)

    def get_sensor_offset(self, cam_id):
        """ Get current offset """
        if cam_id not in self.session.recordings:
            return None

        return self.session.recordings[cam_id].get_sensor_offset()

    # Index subsets

    def get_available_subsets(self):
        """ Return available subsets of frames """
        return {"All": None}

    @staticmethod
    def read_yml_files(files: list[str]):
        """ Read yml files """

        with multiprocessing.Pool() as pool:
            data = pool.map(read_yml, files)
            return data
