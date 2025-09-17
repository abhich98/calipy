# (c) 2019 MPI for Neurobiology of Behavior, Florian Franzen, Abhilash Cheekoti
# SPDX-License-Identifier: LGPL-2.1

import yaml

from .RecordingSession import Session


class CalipySystem(yaml.YAMLObject):

    @staticmethod
    def load(url):
        with open(url, 'r') as file:
            return yaml.load(file, Loader=yaml.Loader)

    def save(self, url):
        with open(url, 'w') as file:
            yaml.dump(self, file)

    def __init__(self):
        # List of all recording sessions
        self.sessions = []

    def add_session(self, session_id=None):
        if session_id is None:
            session_id = f"sess_{len(self.sessions)}"

        self.sessions.append(Session(session_id))
        return self.sessions[-1]

    def remove_session(self, index):
        del self.sessions[index]
