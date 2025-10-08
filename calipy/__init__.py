# (c) 2019 MPI for Neurobiology of Behavior, Florian Franzen, Abhilash Cheekoti
# SPDX-License-Identifier: LGPL-2.1

SOFTWARE = "calipy"
VERSION = "1.4.1"

# Always at least provide VERSION
try:
    from . import core, ui, metaio, detect
    from .main import main
except ImportError:
    pass
