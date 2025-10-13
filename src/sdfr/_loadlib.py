#!/usr/bin/env python
# coding: utf-8

# Copyright 2022 University of Warwick, University of York
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import ctypes as ct
from itertools import product
from pathlib import Path
from site import getsitepackages


def _loadlib():
    """Finds and load the library ``libsdfc_shared.so``, or an alternative.

    Searches first in the current file's directory, then iterates through site
    packages. This enables the library to be found even on an editable install.
    Raises a ``RuntimeError`` if it is missing.
    """

    # Find library in site-packages or local to this module
    local_dir = Path(__file__).resolve().parent
    site_dirs = [Path(x) / "sdfr" for x in getsitepackages()]
    path_dirs = [Path(x) for x in sys.path]
    libs = ("libsdfc_shared.so", "libsdfc_shared.dylib", "sdfc_shared.dll")
    for lib_dir, libname in product([local_dir, *site_dirs, *path_dirs], libs):
        if (lib := lib_dir / libname).exists():
            return ct.cdll.LoadLibrary(str(lib))
    raise RuntimeError("Could not find library 'libsdfc_shared'")


sdf_lib = _loadlib()


def _get_lib_commit_id():
    global sdf_lib
    sdf_lib.sdf_get_library_commit_id.restype = ct.c_char_p
    return sdf_lib.sdf_get_library_commit_id().decode()


def _get_lib_commit_date():
    global sdf_lib
    sdf_lib.sdf_get_library_commit_date.restype = ct.c_char_p
    return sdf_lib.sdf_get_library_commit_date().decode()


__library_commit_id__ = _get_lib_commit_id()
__library_commit_date__ = _get_lib_commit_date()
