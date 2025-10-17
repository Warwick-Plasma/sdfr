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

from .SDF import _new as new
from .SDF import _read as read
from .SDF import _get_md5 as get_md5
from .SDF import _get_sha as get_sha
from .SDF import _add_cpu_split as add_cpu_split
from .SDF import _add_stitched as add_stitched
from .SDF import _add_stitched_vector as add_stitched_vector
from .SDF import _add_stitched_material as add_stitched_material
from .SDF import _add_stitched_matvar as add_stitched_matvar
from .SDF import _add_stitched_species as add_stitched_species
from .SDF import _add_runinfo as add_runinfo
from .SDF import _add_block as add_block
from .SDF import _write as write
from . import sdf_helper
from ._commit_info import (
    __commit_date__,
    __commit_id__,
)
from ._loadlib import (
    __library_commit_date__,
    __library_commit_id__,
)

from importlib.metadata import version

_module_name = "sdfr"

try:
    __version__ = version(_module_name)
except Exception:
    __version__ = "0.0.0"

__all__ = [
    "SDF",
    "new",
    "read",
    "sdf_helper",
    "get_md5",
    "get_sha",
    "add_cpu_split",
    "add_stitched",
    "add_stitched_vector",
    "add_stitched_material",
    "add_stitched_matvar",
    "add_stitched_species",
    "add_runinfo",
    "add_block",
    "write",
    "__library_commit_date__",
    "__library_commit_id__",
    "__version__",
    "__commit_date__",
    "__commit_id__",
]

del version
