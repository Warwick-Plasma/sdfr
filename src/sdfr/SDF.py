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

"""Constants and types for representing the blocks contained in an SDF file."""

import ctypes as _c
import numpy as _np
import struct as _struct
import re as _re
import io as _io
import hashlib as _hashlib
import tarfile as _tarfile
import gzip as _gzip
import os as _os
from enum import IntEnum as _IntEnum
from ._loadlib import sdf_lib as _sdf_lib
from typing import Dict as _Dict
from typing import Any as _Any

# try:
#    import xarray as xr
#
#    got_xarray = True
# except ImportError:
#    print("WARNING: xarray not installed. Generating plain numpy arrays.")
#    got_xarray = False


# Enum representation using _c
class SdfBlockType(_IntEnum):
    SCRUBBED = -1
    NULL = 0
    PLAIN_MESH = 1
    POINT_MESH = 2
    PLAIN_VARIABLE = 3
    POINT_VARIABLE = 4
    CONSTANT = 5
    ARRAY = 6
    RUN_INFO = 7
    SOURCE = 8
    STITCHED_TENSOR = 9
    STITCHED_MATERIAL = 10
    STITCHED_MATVAR = 11
    STITCHED_SPECIES = 12
    SPECIES = 13
    PLAIN_DERIVED = 14
    POINT_DERIVED = 15
    CONTIGUOUS_TENSOR = 16
    CONTIGUOUS_MATERIAL = 17
    CONTIGUOUS_MATVAR = 18
    CONTIGUOUS_SPECIES = 19
    CPU_SPLIT = 20
    STITCHED_OBSTACLE_GROUP = 21
    UNSTRUCTURED_MESH = 22
    STITCHED = 23
    CONTIGUOUS = 24
    LAGRANGIAN_MESH = 25
    STATION = 26
    STATION_DERIVED = 27
    DATABLOCK = 28
    NAMEVALUE = 29


class SdfGeometry(_IntEnum):
    NULL = 0
    CARTESIAN = 1
    CYLINDRICAL = 2
    SPHERICAL = 3


class SdfStagger(_IntEnum):
    CELL_CENTRE = 0
    FACE_X = 1
    FACE_Y = 2
    FACE_Z = 3
    EDGE_X = 4
    EDGE_Y = 5
    EDGE_Z = 6
    VERTEX = 7
    HIDDEN0 = 10
    HIDDEN1 = 11
    HIDDEN2 = 12


class SdfDataType(_IntEnum):
    NULL = 0
    INTEGER4 = 1
    INTEGER8 = 2
    REAL4 = 3
    REAL8 = 4
    REAL16 = 5
    CHARACTER = 6
    LOGICAL = 7
    OTHER = 8


class SdfMode(_IntEnum):
    READ = 1
    WRITE = 2


_np_datatypes = [
    0,
    _np.int32,
    _np.int64,
    _np.float32,
    _np.float64,
    _np.longdouble,
    _np.byte,
    bool,
    0,
]
_ct_datatypes = [
    0,
    _c.c_int32,
    _c.c_int64,
    _c.c_float,
    _c.c_double,
    _c.c_longdouble,
    _c.c_char,
    _c.c_bool,
    0,
]
_st_datatypes = [
    0,
    "i",
    "q",
    "f",
    "d",
    "d",
    "c",
    "?",
    0,
]

# Constants
_SDF_MAXDIMS = 4


class SdfBlock(_c.Structure):
    pass  # Forward declaration for self-referencing structure


class SdfFile(_c.Structure):
    pass  # Forward declaration for function pointer compatibility


SdfBlock._fields_ = [
    ("extents", _c.POINTER(_c.c_double)),
    ("dim_mults", _c.POINTER(_c.c_double)),
    ("station_x", _c.POINTER(_c.c_double)),
    ("station_y", _c.POINTER(_c.c_double)),
    ("station_z", _c.POINTER(_c.c_double)),
    ("mult", _c.c_double),
    ("time", _c.c_double),
    ("time_increment", _c.c_double),
    ("dims", _c.c_int64 * _SDF_MAXDIMS),
    ("local_dims", _c.c_int64 * _SDF_MAXDIMS),
    ("block_start", _c.c_int64),
    ("next_block_location", _c.c_int64),
    ("data_location", _c.c_int64),
    ("inline_block_start", _c.c_int64),
    ("inline_next_block_location", _c.c_int64),
    ("summary_block_start", _c.c_int64),
    ("summary_next_block_location", _c.c_int64),
    ("nelements", _c.c_int64),
    ("nelements_local", _c.c_int64),
    ("data_length", _c.c_int64),
    ("nelements_blocks", _c.POINTER(_c.c_int64)),
    ("data_length_blocks", _c.POINTER(_c.c_int64)),
    ("array_starts", _c.POINTER(_c.c_int64)),
    ("array_ends", _c.POINTER(_c.c_int64)),
    ("array_strides", _c.POINTER(_c.c_int64)),
    ("global_array_starts", _c.POINTER(_c.c_int64)),
    ("global_array_ends", _c.POINTER(_c.c_int64)),
    ("global_array_strides", _c.POINTER(_c.c_int64)),
    ("ndims", _c.c_int32),
    ("geometry", _c.c_int32),
    ("datatype", _c.c_int32),
    ("blocktype", _c.c_int32),
    ("info_length", _c.c_int32),
    ("type_size", _c.c_int32),
    ("stagger", _c.c_int32),
    ("datatype_out", _c.c_int32),
    ("type_size_out", _c.c_int32),
    ("nstations", _c.c_int32),
    ("nvariables", _c.c_int32),
    ("step", _c.c_int32),
    ("step_increment", _c.c_int32),
    ("dims_in", _c.POINTER(_c.c_int32)),
    ("station_nvars", _c.POINTER(_c.c_int32)),
    ("variable_types", _c.POINTER(_c.c_int32)),
    ("station_index", _c.POINTER(_c.c_int32)),
    ("station_move", _c.POINTER(_c.c_int32)),
    ("nm", _c.c_int),
    ("n_ids", _c.c_int),
    ("opt", _c.c_int),
    ("ng", _c.c_int),
    ("nfaces", _c.c_int),
    ("ngrids", _c.c_int),
    ("offset", _c.c_int),
    ("ngb", _c.c_int * 6),
    ("const_value", _c.c_byte * 16),
    ("id", _c.c_char_p),
    ("units", _c.c_char_p),
    ("mesh_id", _c.c_char_p),
    ("material_id", _c.c_char_p),
    ("vfm_id", _c.c_char_p),
    ("obstacle_id", _c.c_char_p),
    ("station_id", _c.c_char_p),
    ("name", _c.c_char_p),
    ("material_name", _c.c_char_p),
    ("must_read", _c.c_char_p),
    ("dim_labels", _c.POINTER(_c.c_char_p)),
    ("dim_units", _c.POINTER(_c.c_char_p)),
    ("station_ids", _c.POINTER(_c.c_char_p)),
    ("variable_ids", _c.POINTER(_c.c_char_p)),
    ("station_names", _c.POINTER(_c.c_char_p)),
    ("material_names", _c.POINTER(_c.c_char_p)),
    ("node_list", _c.POINTER(_c.c_int)),
    ("boundary_cells", _c.POINTER(_c.c_int)),
    ("grids", _c.POINTER(_c.c_void_p)),
    ("data", _c.c_void_p),
    ("done_header", _c.c_bool),
    ("done_info", _c.c_bool),
    ("done_data", _c.c_bool),
    ("dont_allocate", _c.c_bool),
    ("dont_display", _c.c_bool),
    ("dont_own_data", _c.c_bool),
    ("use_mult", _c.c_bool),
    ("next_block_modified", _c.c_bool),
    ("rewrite_metadata", _c.c_bool),
    ("in_file", _c.c_bool),
    ("ng_any", _c.c_bool),
    ("no_internal_ghost", _c.c_bool),
    ("next", _c.POINTER(SdfBlock)),
    ("prev", _c.POINTER(SdfBlock)),
    ("subblock", _c.POINTER(SdfBlock)),
    ("subblock2", _c.POINTER(SdfBlock)),
    (
        "populate_data",
        _c.CFUNCTYPE(
            _c.POINTER(SdfBlock), _c.POINTER(SdfFile), _c.POINTER(SdfBlock)
        ),
    ),
    ("cpu_split", _c.c_int * _SDF_MAXDIMS),
    ("starts", _c.c_int * _SDF_MAXDIMS),
    ("proc_min", _c.c_int * 3),
    ("proc_max", _c.c_int * 3),
    ("ndim_labels", _c.c_int),
    ("ndim_units", _c.c_int),
    ("nstation_ids", _c.c_int),
    ("nvariable_ids", _c.c_int),
    ("nstation_names", _c.c_int),
    ("nmaterial_names", _c.c_int),
    ("option", _c.c_int),
    ("mimetype", _c.c_char_p),
    ("checksum_type", _c.c_char_p),
    ("checksum", _c.c_char_p),
    ("mmap", _c.c_char_p),
    ("mmap_len", _c.c_int64),
    ("derived", _c.c_bool),
    ("id_orig", _c.c_char_p),
    ("name_orig", _c.c_char_p),
]

SdfFile._fields_ = [
    ("dbg_count", _c.c_int64),
    ("sdf_lib_version", _c.c_int32),
    ("sdf_lib_revision", _c.c_int32),
    ("sdf_extension_version", _c.c_int32),
    ("sdf_extension_revision", _c.c_int32),
    ("file_version", _c.c_int32),
    ("file_revision", _c.c_int32),
    ("dbg", _c.c_char_p),
    ("dbg_buf", _c.c_char_p),
    ("extension_names", _c.POINTER(_c.c_char_p)),
    ("time", _c.c_double),
    ("first_block_location", _c.c_int64),
    ("summary_location", _c.c_int64),
    ("start_location", _c.c_int64),
    ("soi", _c.c_int64),
    ("sof", _c.c_int64),
    ("current_location", _c.c_int64),
    ("jobid1", _c.c_int32),
    ("jobid2", _c.c_int32),
    ("endianness", _c.c_int32),
    ("summary_size", _c.c_int32),
    ("block_header_length", _c.c_int32),
    ("string_length", _c.c_int32),
    ("id_length", _c.c_int32),
    ("code_io_version", _c.c_int32),
    ("step", _c.c_int32),
    ("nblocks", _c.c_int32),
    ("nblocks_file", _c.c_int32),
    ("error_code", _c.c_int32),
    ("rank", _c.c_int),
    ("ncpus", _c.c_int),
    ("ndomains", _c.c_int),
    ("rank_master", _c.c_int),
    ("indent", _c.c_int),
    ("print", _c.c_int),
    ("buffer", _c.c_char_p),
    ("filename", _c.c_char_p),
    ("done_header", _c.c_bool),
    ("restart_flag", _c.c_bool),
    ("other_domains", _c.c_bool),
    ("use_float", _c.c_bool),
    ("use_summary", _c.c_bool),
    ("use_random", _c.c_bool),
    ("station_file", _c.c_bool),
    ("swap", _c.c_bool),
    ("inline_metadata_read", _c.c_bool),
    ("summary_metadata_read", _c.c_bool),
    ("inline_metadata_invalid", _c.c_bool),
    ("summary_metadata_invalid", _c.c_bool),
    ("tmp_flag", _c.c_bool),
    ("metadata_modified", _c.c_bool),
    ("can_truncate", _c.c_bool),
    ("first_block_modified", _c.c_bool),
    ("code_name", _c.c_char_p),
    ("error_message", _c.c_char_p),
    ("blocklist", _c.POINTER(SdfBlock)),
    ("tail", _c.POINTER(SdfBlock)),
    ("current_block", _c.POINTER(SdfBlock)),
    ("last_block_in_file", _c.POINTER(SdfBlock)),
    ("mmap", _c.c_char_p),
    ("ext_data", _c.c_void_p),
    ("stack_handle", _c.c_void_p),
    ("array_count", _c.c_int),
    ("fd", _c.c_int),
    ("purge_duplicated_ids", _c.c_int),
    ("internal_ghost_cells", _c.c_int),
    ("ignore_nblocks", _c.c_int),
]


class RunInfo(_c.Structure):
    _fields_ = [
        ("defines", _c.c_int64),
        ("version", _c.c_int32),
        ("revision", _c.c_int32),
        ("compile_date", _c.c_int32),
        ("run_date", _c.c_int32),
        ("io_date", _c.c_int32),
        ("minor_rev", _c.c_int32),
        ("commit_id", _c.c_char_p),
        ("sha1sum", _c.c_char_p),
        ("compile_machine", _c.c_char_p),
        ("compile_flags", _c.c_char_p),
    ]


class BlockDict(dict):
    def __init__(self, _dict, block):
        self.handle = block._handle
        super().__init__(_dict)

    def __setitem__(self, key, value):
        h = self.handle
        if key in (
            "step",
            "time",
            "code_io_version",
            "string_length",
            "jobid1",
            "jobid2",
        ):
            setattr(h.contents, key, value)
        elif key == "code_name":
            h._clib.sdf_set_code_name(h, value.encode("utf-8"))
        else:
            print(f'WARNING: unable to set header key "{key}"')
            return
        super().__setitem__(key, value)


class BlockList:
    """Contains all the blocks"""

    def __init__(
        self,
        filename=None,
        convert=False,
        derived=True,
        mode=SdfMode.READ,
        code_name="sdfr",
        restart=False,
    ):
        self._handle = None
        clib = _sdf_lib
        self._clib = clib
        clib.sdf_open.restype = _c.POINTER(SdfFile)
        clib.sdf_open.argtypes = [_c.c_char_p, _c.c_int, _c.c_int, _c.c_int]
        clib.sdf_new.restype = _c.POINTER(SdfFile)
        clib.sdf_new.argtypes = [_c.c_int, _c.c_int]
        clib.sdf_stack_init.argtypes = [_c.c_void_p]
        clib.sdf_read_blocklist.argtypes = [_c.c_void_p]
        clib.sdf_read_blocklist_all.argtypes = [_c.c_void_p]
        clib.sdf_helper_read_data.argtypes = [_c.c_void_p, _c.POINTER(SdfBlock)]
        clib.sdf_free_block_data.argtypes = [_c.c_void_p, _c.POINTER(SdfBlock)]
        clib.sdf_stack_destroy.argtypes = [_c.c_void_p]
        clib.sdf_close.argtypes = [_c.c_void_p]
        clib.sdf_write.argtypes = [_c.c_void_p, _c.c_char_p]
        clib.sdf_get_next_block.argtypes = [_c.c_void_p]
        clib.sdf_set_namevalue.argtypes = [
            _c.POINTER(SdfBlock),
            _c.POINTER(_c.c_char_p),
            _c.POINTER(_c.c_void_p),
        ]
        clib.sdf_set_code_name.argtypes = [_c.c_void_p, _c.c_char_p]
        clib.sdf_set_block_name.argtypes = [
            _c.c_void_p,
            _c.c_char_p,
            _c.c_char_p,
        ]
        clib.sdf_set_defaults.argtypes = [
            _c.c_void_p,
            _c.POINTER(SdfBlock),
        ]
        clib.sdf_create_id.argtypes = [
            _c.c_void_p,
            _c.c_char_p,
        ]
        clib.sdf_create_id.restype = _c.POINTER(_c.c_char_p)
        clib.sdf_create_id_array.argtypes = [
            _c.c_void_p,
            _c.c_int,
            _c.POINTER(_c.c_char_p),
        ]
        clib.sdf_create_id_array.restype = _c.POINTER(_c.c_char_p)
        clib.sdf_create_string.argtypes = [
            _c.c_void_p,
            _c.c_char_p,
        ]
        clib.sdf_create_string.restype = _c.POINTER(_c.c_char_p)
        clib.sdf_create_string_array.argtypes = [
            _c.c_void_p,
            _c.c_int,
            _c.POINTER(_c.c_char_p),
        ]
        clib.sdf_create_string_array.restype = _c.POINTER(_c.c_char_p)

        comm = 0
        use_mmap = 0
        if filename is None:
            h = clib.sdf_new(comm, use_mmap)
        else:
            h = clib.sdf_open(filename.encode("utf-8"), comm, mode, use_mmap)
        if h is None or not bool(h):
            raise Exception(f"Failed to open file: '{filename}'")

        if convert:
            h.contents.use_float = True

        h._clib = clib
        self._handle = h
        clib.sdf_stack_init(h)
        if mode == SdfMode.READ:
            if derived:
                clib.sdf_read_blocklist_all(h)
            else:
                clib.sdf_read_blocklist(h)
        else:
            clib.sdf_set_code_name(h, code_name.encode("utf-8"))

        block = h.contents.blocklist
        h.contents.restart_flag = restart
        self._header = self._get_header(h.contents)
        mesh_id_map = {}
        mesh_vars = []
        self._block_ids = {"Header": self.Header}
        self._block_names = {"Header": self.Header}
        for n in range(h.contents.nblocks):
            block = block.contents
            block._handle = h
            block._blocklist = self
            blocktype = block.blocktype
            newblock = None
            newblock_mid = None
            if block.name_orig:
                name = _get_member_name(block.name_orig)
            else:
                name = _get_member_name(block.name)
            if blocktype == SdfBlockType.ARRAY:
                newblock = BlockArray(block)
            elif blocktype == SdfBlockType.CONSTANT:
                newblock = BlockConstant(block)
            elif (
                blocktype == SdfBlockType.CONTIGUOUS
                or blocktype == SdfBlockType.STITCHED
            ):
                if block.stagger in (SdfStagger.HIDDEN0, SdfStagger.HIDDEN2):
                    newblock = BlockStitchedPath(block)
                else:
                    newblock = BlockStitched(block)
                mesh_vars.append(newblock)
            elif (
                blocktype == SdfBlockType.CONTIGUOUS_MATERIAL
                or blocktype == SdfBlockType.STITCHED_MATERIAL
            ):
                newblock = BlockStitchedMaterial(block)
                mesh_vars.append(newblock)
            elif (
                blocktype == SdfBlockType.CONTIGUOUS_MATVAR
                or blocktype == SdfBlockType.STITCHED_MATVAR
            ):
                newblock = BlockStitchedMatvar(block)
                mesh_vars.append(newblock)
            elif (
                blocktype == SdfBlockType.CONTIGUOUS_SPECIES
                or blocktype == SdfBlockType.STITCHED_SPECIES
            ):
                newblock = BlockStitchedSpecies(block)
                mesh_vars.append(newblock)
            elif (
                blocktype == SdfBlockType.CONTIGUOUS_TENSOR
                or blocktype == SdfBlockType.STITCHED_TENSOR
            ):
                newblock = BlockStitchedTensor(block)
                mesh_vars.append(newblock)
            elif blocktype == SdfBlockType.DATABLOCK:
                newblock = BlockData(block)
            elif blocktype == SdfBlockType.LAGRANGIAN_MESH:
                if block.datatype_out != 0:
                    newblock = BlockLagrangianMesh(block)
                    newblock_mid = block
                    newblock_mid._grid_block = newblock
                    mesh_id_map[newblock.id] = newblock
            elif blocktype == SdfBlockType.NAMEVALUE:
                newblock = BlockNameValue(block)
            elif (
                blocktype == SdfBlockType.PLAIN_DERIVED
                or blocktype == SdfBlockType.PLAIN_VARIABLE
            ):
                newblock = BlockPlainVariable(block)
                mesh_vars.append(newblock)
            elif blocktype == SdfBlockType.PLAIN_MESH:
                if block.datatype_out != 0:
                    newblock = BlockPlainMesh(block)
                    newblock_mid = block
                    newblock_mid._grid_block = newblock
                    mesh_id_map[newblock.id] = newblock
            elif (
                blocktype == SdfBlockType.POINT_DERIVED
                or blocktype == SdfBlockType.POINT_VARIABLE
            ):
                newblock = BlockPointVariable(block)
                mesh_vars.append(newblock)
            elif blocktype == SdfBlockType.POINT_MESH:
                newblock = BlockPointMesh(block)
                mesh_id_map[newblock.id] = newblock
            elif blocktype == SdfBlockType.RUN_INFO:
                newblock = BlockRunInfo(block)
            elif blocktype == SdfBlockType.STATION:
                sdict = _BlockStation(block, name)
                self.__dict__.update({"StationBlocks": sdict})
                self._block_ids.update({block.id.decode(): sdict})
                self._block_names.update({block.name.decode(): sdict})
            elif blocktype == SdfBlockType.CPU_SPLIT:
                newblock = BlockCpuSplit(block)
                name = "_" + name
            else:
                # Block not supported
                # print(name,SdfBlockType(blocktype).name)
                pass
            if newblock is not None:
                if not block.dont_display:
                    self.__dict__[name] = newblock
                self._block_ids.update({block.id.decode(): newblock})
                self._block_names.update({block.name.decode(): newblock})
            block = block.next

            if newblock_mid is not None:
                block_mid = newblock_mid
                block_mid._handle = h
                block_mid._blocklist = self
                blocktype = block_mid.blocktype
                name = _get_member_name(block_mid.name) + "_mid"
                if blocktype == SdfBlockType.LAGRANGIAN_MESH:
                    newblock = BlockLagrangianMesh(block_mid, mid=True)
                elif blocktype == SdfBlockType.PLAIN_MESH:
                    newblock = BlockPlainMesh(block_mid, mid=True)
                if not newblock_mid.dont_display:
                    self.__dict__[name] = newblock
                nm = block_mid.id.decode() + "_mid"
                self._block_ids.update({nm: newblock})
                nm = block_mid.name.decode() + "_mid"
                self._block_names.update({nm: newblock})
                newblock_mid._grid_block._grid_mid = newblock

        for var in mesh_vars:
            gid = var.grid_id
            if gid in mesh_id_map:
                var._grid = mesh_id_map[gid]

    def __del__(self):
        if self._handle:
            self._clib.sdf_stack_destroy(self._handle)
            self._clib.sdf_close(self._handle)
            self._handle = None

    @property
    def Header(self) -> _Dict[str, _Any]:
        """SDF file header"""
        return self._header

    @Header.setter
    def Header(self, value):
        try:
            for k, v in value.items():
                self.Header[k] = v
        except Exception:
            print("failed")

    def _get_header(self, h):
        d = {}
        for k in [
            "filename",
            "file_version",
            "file_revision",
            "code_name",
            "step",
            "time",
            "jobid1",
            "jobid2",
            "code_io_version",
            "restart_flag",
            "other_domains",
            "station_file",
        ]:
            attr = getattr(h, k)
            if isinstance(attr, bytes):
                d[k] = attr.decode()
            else:
                d[k] = attr
        return BlockDict(d, self)

    def _set_block_name(self, id, name):
        self._clib.sdf_set_block_name(
            self._handle, id.encode("utf-8"), name.encode("utf-8")
        )

    def _create_id(self, values):
        tmp = self._clib.sdf_create_id(self._handle, values.encode("utf-8"))
        return _c.cast(tmp, _c.c_char_p)

    def _create_string(self, values):
        tmp = self._clib.sdf_create_string(self._handle, values.encode("utf-8"))
        return _c.cast(tmp, _c.c_char_p)

    def _string_array_ctype(self, values):
        strings = [s.encode("utf-8") for s in values]
        strings = [_c.create_string_buffer(s) for s in strings]
        strings = [_c.cast(s, _c.c_char_p) for s in strings]
        strings = (_c.c_char_p * len(values))(*strings)
        return strings

    def _create_id_array(self, values):
        values = self._string_array_ctype(values)
        res = self._clib.sdf_create_id_array(self._handle, len(values), values)
        return res

    def _create_string_array(self, values):
        values = self._string_array_ctype(values)
        res = self._clib.sdf_create_string_array(
            self._handle, len(values), values
        )
        return res

    @property
    def name_dict(self):
        """Dictionary of blocks using name field as key"""
        return self._block_names

    @property
    def id_dict(self):
        """Dictionary of blocks using id field as key"""
        return self._block_ids


class Block:
    """SDF block type
    Contains the data and metadata for a single
    block from an SDF file.
    """

    def __init__(self, block):
        self._handle = block._handle
        if block.id_orig:
            self._id = block.id_orig.decode()
            self._name = block.name_orig.decode()
        else:
            self._id = block.id.decode()
            self._name = block.name.decode()
        self._datatype = _np_datatypes[block.datatype_out]
        self._data_length = block.data_length
        self._dims = tuple(block.dims[: block.ndims])
        self._contents = block
        self._blocklist = block._blocklist
        self._in_file = block.in_file
        self._data = None
        self._grid = None

    def _numpy_from_buffer(self, data, blen):
        buffer_from_memory = _c.pythonapi.PyMemoryView_FromMemory
        buffer_from_memory.restype = _c.py_object
        dtype = self._datatype
        if dtype == _np.byte:
            dtype = _np.dtype("|S1")
        totype = _ct_datatypes[self._contents.datatype_out]
        cast = _c.cast(data, _c.POINTER(totype))
        buf = buffer_from_memory(cast, blen)
        return _np.frombuffer(buf, dtype)

    @property
    def blocklist(self):
        """Blocklist"""
        return self._blocklist

    @property
    def data(self):
        """Block data contents"""
        return self._data

    @property
    def datatype(self):
        """Data type"""
        return self._datatype

    @property
    def data_length(self):
        """Data size"""
        return self._data_length

    @property
    def dims(self):
        """Data dimensions"""
        return self._dims

    @property
    def id(self):
        """Block id"""
        return self._id

    @property
    def name(self):
        """Block name"""
        return self._name


class BlockRunInfo(Block, dict):
    """Run info block"""

    def __init__(self, block, info=None):
        import datetime
        from datetime import datetime as dtm

        if isinstance(block, Block):
            block = block._contents

        Block.__init__(self, block)

        if info is not None:
            self._run_info = self._build_info(info)
            block.data = _c.cast(_c.byref(self._run_info), _c.c_void_p)

        utc = datetime.timezone.utc

        h = _c.cast(block.data, _c.POINTER(RunInfo)).contents

        self._dict = {
            "version": f"{h.version}.{h.revision}.{h.minor_rev}",
            "commit_id": h.commit_id.decode(),
            "sha1sum": h.sha1sum.decode(),
            "compile_machine": h.compile_machine.decode(),
            "compile_flags": h.compile_flags.decode(),
            "compile_date": dtm.fromtimestamp(h.compile_date, utc).strftime(
                "%c"
            ),
            "run_date": dtm.fromtimestamp(h.run_date, utc).strftime("%c"),
            "io_date": dtm.fromtimestamp(h.io_date, utc).strftime("%c"),
        }

        dict.__init__(self, self._dict)

    def _build_info(self, info):
        import datetime as dtm
        import dateutil.parser as dtp

        run_info = RunInfo()
        fields = [f[0] for f in run_info._fields_]
        for f in run_info._fields_:
            if f[1] == _c.c_char_p:
                setattr(run_info, f[0], "".encode())
            else:
                setattr(run_info, f[0], 0)
        k = "version"
        if k in info and isinstance(info[k], str):
            ver = [int(s) for s in info[k].split(".")]
            info[k] = ver[0]
            if len(ver) > 1:
                info["revision"] = ver[1]
            if len(ver) > 2:
                info["minor_rev"] = ver[2]
        for k, v in info.items():
            if k.endswith("_date"):
                if isinstance(v, str):
                    date = dtp.parse(v)
                    date = date.replace(tzinfo=dtm.timezone.utc)
                    v = int(date.timestamp())
                elif isinstance(v, dtm.datetime):
                    v = int(v.timestamp())
            if k in fields:
                setattr(run_info, k, v)
        return run_info


class BlockConstant(Block):
    """Constant block"""

    def __init__(self, block):
        super().__init__(block)
        offset = getattr(SdfBlock, "const_value").offset
        self._datatype = _np_datatypes[block.datatype]
        totype = _ct_datatypes[block.datatype]
        self._data = totype.from_buffer(block, offset).value


class BlockPlainVariable(Block):
    """Plain variable block"""

    @property
    def data(self):
        """Block data contents"""
        if self._data is None:
            clib = self._handle._clib
            clib.sdf_helper_read_data(self._handle, self._contents)
            blen = _np.dtype(self._datatype).itemsize
            for d in self.dims:
                blen *= d
            array = self._numpy_from_buffer(self._contents.data, blen)
            self._data = array.reshape(self.dims, order="F")
        return self._data

    @property
    def grid(self):
        """Associated mesh"""
        return self._grid

    @property
    def grid_mid(self):
        """Associated median mesh"""
        return self._grid._grid_mid

    @property
    def grid_id(self):
        """Associated mesh id"""
        return self._contents.mesh_id.decode()

    @property
    def mult(self):
        """Multiplication factor"""
        return self._contents.mult

    @property
    def stagger(self):
        """Grid stagger"""
        return SdfStagger(self._contents.stagger)

    @property
    def units(self):
        """Units of variable"""
        return self._contents.units.decode()


class BlockPlainMesh(Block):
    """Plain mesh block"""

    def __init__(self, block, mid=False):
        super().__init__(block)
        self._mid = mid
        self._data = None
        self._units = tuple(
            [block.dim_units[i].decode() for i in range(block.ndims)]
        )
        self._labels = tuple(
            [block.dim_labels[i].decode() for i in range(block.ndims)]
        )
        self._mult = None
        self._bdims = self._dims
        if mid:
            self._id += "_mid"
            self._name += "_mid"
            self._dims = tuple([i - 1 for i in self._dims])
            self._in_file = False
        if bool(block.dim_mults):
            self._mult = tuple(block.dim_mults[: block.ndims])
        if bool(block.extents):
            self._extents = tuple(block.extents[: 2 * block.ndims])

    @property
    def data(self):
        """Block data contents"""
        if self._data is None:
            clib = self._handle._clib
            clib.sdf_helper_read_data(self._handle, self._contents)
            grids = []
            for i, d in enumerate(self._bdims):
                blen = _np.dtype(self._datatype).itemsize * d
                array = self._numpy_from_buffer(self._contents.grids[i], blen)
                if self._mid:
                    array = 0.5 * (array[1:] + array[:-1])
                grids.append(array)
            self._data = tuple(grids)
        return self._data

    @property
    def extents(self):
        """Axis extents"""
        return self._extents

    @property
    def geometry(self):
        """Domain geometry"""
        return SdfGeometry(self._contents.geometry)

    @property
    def labels(self):
        """Axis labels"""
        return self._labels

    @property
    def mult(self):
        """Multiplication factor"""
        return self._mult

    @property
    def units(self):
        """Units of variable"""
        return self._units


class BlockLagrangianMesh(BlockPlainMesh):
    """Lagrangian mesh block"""

    @property
    def data(self):
        """Block data contents"""
        if self._data is None:
            clib = self._handle._clib
            clib.sdf_helper_read_data(self._handle, self._contents)
            blen = _np.dtype(self._datatype).itemsize
            for d in self._bdims:
                blen *= d
            grids = []
            for i, d in enumerate(self._bdims):
                array = self._numpy_from_buffer(self._contents.grids[i], blen)
                array = array.reshape(self._bdims, order="F")
                if self._mid:
                    nn = len(self._bdims)
                    for j in range(nn):
                        s1 = nn * [slice(None)]
                        s2 = nn * [slice(None)]
                        s1[j] = slice(1, None)
                        s2[j] = slice(None, -1)
                        array = 0.5 * (array[tuple(s1)] + array[tuple(s2)])
                grids.append(array)
            self._data = tuple(grids)
        return self._data


class BlockPointMesh(BlockPlainMesh):
    """Point mesh block"""

    @property
    def species_id(self):
        """Species ID"""
        return self._contents.material_id.decode()


class BlockPointVariable(BlockPlainVariable):
    """Point variable block"""

    @property
    def species_id(self):
        """Species ID"""
        return self._contents.material_id.decode()


class BlockNameValue(Block):
    """Name/value block"""

    def __init__(self, block):
        super().__init__(block)
        self._dims = (block.ndims,)
        vals = {}
        for n in range(block.ndims):
            val = None
            if block.datatype == SdfDataType.CHARACTER:
                p = _c.cast(block.data, _c.POINTER(_c.c_char_p))
                val = p[n].decode()
            else:
                dt = _ct_datatypes[block.datatype]
                val = _c.cast(block.data, _c.POINTER(dt))[n]
            nid = _get_member_name(block.material_names[n])
            vals[nid] = val
            self.__dict__[nid] = val
        self._data = vals


class BlockArray(Block):
    """Array block"""

    @property
    def data(self):
        """Block data contents"""
        if self._data is None:
            clib = self._handle._clib
            clib.sdf_helper_read_data(self._handle, self._contents)
            blen = _np.dtype(self._datatype).itemsize
            for d in self.dims:
                blen *= d
            array = self._numpy_from_buffer(self._contents.data, blen)
            self._data = array.reshape(self.dims, order="F")
        return self._data


class BlockData(Block):
    """Data block"""

    def __init__(self, block):
        super().__init__(block)
        self._checksum = block.checksum.decode()
        self._checksum_type = block.checksum_type.decode()
        self._mimetype = block.mimetype.decode()

    @property
    def data(self):
        """Block data contents"""
        if self._data is None:
            clib = self._handle._clib
            clib.sdf_helper_read_data(self._handle, self._contents)
            blen = self._contents.data_length
            _data = _c.cast(self._contents.data, _c.POINTER(_c.c_char * blen))
            self._data = _data.contents[:]
        return self._data

    @property
    def checksum(self):
        """Block data checksum"""
        return self._checksum

    @property
    def checksum_type(self):
        """Block data checksum type"""
        return self._checksum_type

    @property
    def mimetype(self):
        """mimetype for Block data contents"""
        return self._mimetype


def _BlockStation(block, name):
    """Station block"""
    sdict = dict(
        stations=None,
        step=block.step,
        step_increment=block.step_increment,
        time=block.time,
        time_increment=block.time_increment,
    )

    tdict = {}
    for i in range(block.nstations):
        varnames = []
        for j in range(block.station_nvars[i]):
            varnames.append(block.material_names[i + j + 1].decode())
        stat = dict(variables=varnames)
        stat.update({"station_move": bool(block.station_move[i])})
        if block.ndims > 0:
            stat.update({"station_x": block.station_x[i]})
        if block.ndims > 1:
            stat.update({"station_y": block.station_y[i]})
        if block.ndims > 2:
            stat.update({"station_z": block.station_z[i]})
        tdict.update({block.station_names[i].decode(): stat})
    sdict.update({"stations": tdict})

    return {name: sdict}


class BlockStitched(Block):
    """Stitched block"""

    @property
    def data(self):
        """Block data contents"""
        if self._data is None:
            self._data = []
            for i in range(self._contents.ndims):
                vid = self._contents.variable_ids[i]
                if len(vid) > 0:
                    vid = vid.decode()
                    if vid in self._blocklist._block_ids:
                        self._data.append(self._blocklist._block_ids[vid])
                    else:
                        self._data.append(None)
                else:
                    self._data.append(None)
        return self._data

    @property
    def grid(self):
        """Associated mesh"""
        return self._grid

    @property
    def grid_id(self):
        """Associated mesh id"""
        return self._contents.mesh_id.decode()


class BlockStitchedPath(BlockStitched):
    """Stitched path block"""

    pass


class BlockStitchedMaterial(BlockStitched):
    """Stitched material block"""

    @property
    def material_names(self):
        """Material names"""
        b = self._contents
        return [b.material_names[i].decode() for i in range(b.ndims)]


class BlockStitchedMatvar(BlockStitched):
    """Stitched material variable block"""

    @property
    def material_id(self):
        """Material ID"""
        return self._contents.material_id.decode()


class BlockStitchedSpecies(BlockStitched):
    """Stitched species block"""

    @property
    def material_id(self):
        """Material ID"""
        return self._contents.material_id.decode()

    @property
    def material_name(self):
        """Material name"""
        return self._contents.material_name.decode()

    @property
    def material_names(self):
        """Species names"""
        b = self._contents
        return [b.material_names[i].decode() for i in range(b.ndims)]


class BlockStitchedTensor(BlockStitched):
    """Stitched tensor block"""

    pass


class BlockCpuSplit(Block):
    """CPU split block"""

    def __init__(self, block):
        super().__init__(block)
        nelements = 0
        if self._contents.geometry in (1, 4):
            nelements = sum(self.dims)
        elif self._contents.geometry == 2:
            nelements, adim = 0, []
            for dim in self.dims:
                adim.append(dim)
                nelements += _np.prod(adim)
        elif self._contents.geometry == 3:
            nelements = _np.prod(self.dims)
        else:
            raise Exception("CPU split geometry not supported")
        self._contents.nelements = nelements

    @property
    def data(self):
        """Block data contents"""
        if self._data is None:
            clib = self._handle._clib
            clib.sdf_helper_read_data(self._handle, self._contents)
            nelements = self._contents.nelements
            blen = _np.dtype(self._datatype).itemsize * nelements
            array = self._numpy_from_buffer(self._contents.data, blen)
            if self._contents.geometry in (1, 4):
                d0, data = 0, []
                for dim in self.dims:
                    d1 = d0 + dim
                    data.append(array[d0:d1])
                    d0 = d1
                self._data = tuple(data)
            elif self._contents.geometry == 2:
                d0, data, adim = 0, [], []
                for dim in self.dims:
                    adim.append(dim)
                    d1 = d0 + _np.prod(adim)
                    data.append(array[d0:d1].reshape(adim, order="F"))
                    d0 = d1
                self._data = tuple(data)
            elif self._contents.geometry == 3:
                self._data = array.reshape(self.dims, order="F")
        return self._data


_re_pattern = _re.compile(r"[^a-zA-Z0-9]")


def _get_member_name(name):
    sname = name.decode()
    return _re_pattern.sub("_", sname)


def _read(file=None, convert=False, mmap=0, dict=False, derived=True):
    """Reads the SDF data and returns a dictionary of NumPy arrays.

    Parameters
    ----------
    file : string
        The name of the SDF file to open.
    convert : bool, optional
        Convert double precision data to single when reading file.
    dict : bool, optional
        Return file contents as a dictionary rather than member names.
    derived : bool, optional
        Include derived variables in the data structure.
    """

    import warnings

    if file is None:
        raise TypeError("Missing file parameter")

    if mmap != 0:
        warnings.warn("mmap flag ignored")

    blocklist = BlockList(file, convert, derived)

    if isinstance(dict, str):
        if dict == "id" or dict == "ids":
            return blocklist._block_ids
    elif isinstance(dict, bool) and dict:
        return blocklist._block_names

    return blocklist


def _new(dict=False, code_name="sdfr", restart=False):
    """Creates a new SDF blocklist and returns a dictionary of NumPy arrays.

    Parameters
    ----------
    dict : bool, optional
        Return file contents as a dictionary rather than member names.
    """

    blocklist = BlockList(
        mode=SdfMode.WRITE, code_name=code_name, restart=restart
    )

    if isinstance(dict, str):
        if dict == "id" or dict == "ids":
            return blocklist._block_ids
    elif isinstance(dict, bool) and dict:
        return blocklist._block_names

    return blocklist


def _get_md5(data):
    return _hashlib.md5(data).hexdigest()


def _get_sha(data):
    sha = _hashlib.sha256()
    try:
        with _io.BytesIO(data) as buf:
            with _tarfile.open(mode="r:*", fileobj=buf) as file:
                for name in sorted(file.getnames()):
                    fd = file.extractfile(name)
                    if fd:
                        sha.update(fd.read())
                        fd.close()
    except Exception:
        try:
            sha.update(_gzip.decompress(data))
        except Exception:
            sha.update(data)
    return sha.hexdigest()


def _get_tarfile_data(source=None):
    if not source or not _os.path.exists(source):
        return source
    if _os.path.isdir(source):
        buf = _io.BytesIO()
        with _tarfile.open("out", "w:gz", fileobj=buf) as tar:
            tar.add(source)
        buf.seek(0)
        return buf.read()
    else:
        with open(source, "rb") as fd:
            return fd.read()


def _get_mimetype(data):
    try:
        _ = _gzip.decompress(data)
        return "application/gzip"
    except Exception:
        pass
    try:
        buf = _io.BytesIO(data)
        _ = _tarfile.open(mode="r:*", fileobj=buf)
        return "application/tar"
    except Exception:
        return "text/plain"


def _get_checksum_info(value=None, checksum_type=None):
    if not value:
        return None, None, None, value
    data = _get_tarfile_data(value)
    mimetype = _get_mimetype(data)
    if checksum_type is None:
        if mimetype == "text/plain":
            checksum_type = "md5"
        else:
            checksum_type = "sha256"
    if checksum_type == "md5":
        checksum = _get_md5(data)
    else:
        checksum = _get_sha(data)

    return checksum, checksum_type, mimetype, data


def _add_preamble(blocklist, id, name, datatype):
    blocklist._clib.sdf_get_next_block(blocklist._handle)
    h = blocklist._handle.contents
    h.nblocks += 1
    h.nblocks_file += 1
    block = h.current_block.contents
    block._handle = blocklist._handle
    block._blocklist = h.blocklist
    block._data = None
    block.datatype = datatype
    block.in_file = 1
    block.AddBlock = None
    blocklist._set_block_name(id, name)
    return h, block


def _add_post(blocklist, block, extra=None):
    if block.AddBlock:
        if extra is None:
            newblock = block.AddBlock(block)
        else:
            newblock = block.AddBlock(block, extra)
    else:
        return

    id = block.id.decode()
    name = block.name.decode()
    if not block.dont_display:
        blocklist.__dict__[name] = newblock
    if block._data is not None:
        newblock._data = block._data
    blocklist._block_ids.update({id: newblock})
    blocklist._block_names.update({name: newblock})


def _add_constant(blocklist, name, value=0, datatype=None, id=None):
    h, block = _add_preamble(blocklist, id, name, datatype)
    block.blocktype = SdfBlockType.CONSTANT
    block.AddBlock = BlockConstant

    const_value = _struct.pack(_st_datatypes[block.datatype], value)
    _c.memmove(block.const_value, const_value, 16)

    _add_post(blocklist, block)


def _add_namevalue(blocklist, name, value={}, datatype=None, id=None):
    h, block = _add_preamble(blocklist, id, name, datatype)
    block.blocktype = SdfBlockType.NAMEVALUE
    block.AddBlock = BlockNameValue

    nvalue = len(value)
    block.ndims = nvalue
    ctype = _ct_datatypes[block.datatype]
    if block.datatype == SdfDataType.CHARACTER:
        vals = blocklist._string_array_ctype(value.values())
    else:
        vals = (ctype * nvalue)(*value.values())
    names = blocklist._string_array_ctype(value.keys())
    vals = _c.cast(vals, _c.POINTER(_c.c_void_p))
    blocklist._clib.sdf_set_namevalue(block, names, vals)

    _add_post(blocklist, block)


def _add_array(blocklist, name, value=(), datatype=None, id=None):
    h, block = _add_preamble(blocklist, id, name, datatype)
    block.blocktype = SdfBlockType.ARRAY
    block.AddBlock = BlockArray

    block._data = _np.array(value)
    block.ndims = block._data.ndim
    for i in range(block.ndims):
        block.dims[i] = block._data.shape[i]
    block.data = block._data.ctypes.data_as(_c.c_void_p)

    _add_post(blocklist, block)


def _add_cpu_split(
    blocklist, name, value=(), id=None, datatype=None, geometry=1
):
    from itertools import chain

    if not isinstance(blocklist, BlockList):
        print("ERROR: first argument must be of type BlockList")
        return
    if id is None:
        id = name
    if datatype is None:
        if geometry == 4:
            datatype = SdfDataType.INTEGER8
        else:
            datatype = SdfDataType.INTEGER4
    h, block = _add_preamble(blocklist, id, name, datatype)
    block.blocktype = SdfBlockType.CPU_SPLIT
    block.AddBlock = BlockCpuSplit

    dtype = _np_datatypes[datatype]
    block._data = _np.asarray(list(chain.from_iterable(value)), dtype=dtype)
    block.ndims = len(value)
    block.geometry = geometry
    dims = []
    for i in range(block.ndims):
        dims.append(len(value[i]))
        block.dims[i] = len(value[i])

    block.data = block._data.ctypes.data_as(_c.c_void_p)

    _add_post(blocklist, block)


def _add_datablock(
    blocklist,
    name,
    value=(),
    id=None,
    checksum=None,
    checksum_type=None,
    mimetype=None,
    datatype=None,
):
    datatype = SdfDataType.CHARACTER
    h, block = _add_preamble(blocklist, id, name, datatype)
    block.blocktype = SdfBlockType.DATABLOCK
    block.AddBlock = BlockData

    if not checksum:
        checksum, checksum_type, mimetype, value = _get_checksum_info(
            value, checksum_type
        )

    if isinstance(checksum, str):
        block.checksum = blocklist._create_string(checksum)
    if isinstance(checksum_type, str):
        block.checksum_type = blocklist._create_id(checksum_type)
    if isinstance(mimetype, str):
        block.mimetype = blocklist._create_id(mimetype)

    block._data = _np.array(value)
    block.ndims = 0
    block.nelements = len(value)
    block.data = block._data.ctypes.data_as(_c.c_void_p)

    _add_post(blocklist, block)


def _add_plainvar(
    blocklist,
    name,
    value=(),
    datatype=None,
    id=None,
    mult=None,
    units=None,
    mesh_id=None,
    stagger=None,
    species=None,
):
    try:
        mult = float(mult)
    except Exception:
        if mult is not None:
            print(f"ERROR: unable to use mult parameter, {mult}")
            return
    try:
        stagger = SdfStagger(stagger)
    except Exception:
        if stagger is not None:
            print(f"ERROR: unable to use stagger parameter, {stagger}")
            return
    if units is not None and not isinstance(units, str):
        print(f"ERROR: unable to use units parameter, {units}")
        return
    if mesh_id is not None and not isinstance(mesh_id, str):
        print(f"ERROR: unable to use mesh_id parameter, {mesh_id}")
        return

    h, block = _add_preamble(blocklist, id, name, datatype)

    block._data = _np.array(value, order="F")
    block.ndims = block._data.ndim

    if block.ndims == 1 and isinstance(species, str):
        block.blocktype = SdfBlockType.POINT_VARIABLE
        block.AddBlock = BlockPointVariable
        block.material_id = blocklist._create_id(species)
    else:
        block.blocktype = SdfBlockType.PLAIN_VARIABLE
        block.AddBlock = BlockPlainVariable

    for i in range(block.ndims):
        block.dims[i] = block._data.shape[i]
    block.data = block._data.ctypes.data_as(_c.c_void_p)
    if mult is not None:
        block.mult = mult
    if isinstance(units, str):
        block.units = blocklist._create_id(units)
    if isinstance(mesh_id, str):
        block.mesh_id = blocklist._create_id(mesh_id)
    if stagger:
        block.stagger = stagger

    blocklist._clib.sdf_set_defaults(blocklist._handle, block)
    _add_post(blocklist, block)


def _add_mesh(
    blocklist,
    name,
    value=None,
    datatype=None,
    id=None,
    units=None,
    labels=None,
    geometry=None,
    species=None,
    **kwargs,
):
    h, block = _add_preamble(blocklist, id, name, datatype)

    keys = ["x", "y", "z"]
    keys = [k for k in keys if k in kwargs and kwargs[k] is not None]
    val = _np.concatenate([kwargs[k] for k in keys]).flatten()[0]

    block._data = [_np.array(kwargs[k], dtype=val.dtype) for k in keys]
    block._data = [_np.array(row, order="F") for row in block._data]
    block._data = tuple(block._data)
    block.ndims = len(block._data)
    block.ngrids = block.ndims
    grids = [row.ctypes.data_as(_c.c_void_p) for row in block._data]
    block.grids = (_c.c_void_p * block.ngrids)(*grids)
    if block._data[0].ndim == 1:
        block.blocktype = SdfBlockType.PLAIN_MESH
        block.AddBlock = BlockPlainMesh
        for i in range(block.ndims):
            block.dims[i] = block._data[i].shape[0]
        if isinstance(species, str):
            block.blocktype = SdfBlockType.POINT_MESH
            block.AddBlock = BlockPointMesh
            block.material_id = blocklist._create_id(species)
    else:
        block.blocktype = SdfBlockType.LAGRANGIAN_MESH
        block.AddBlock = BlockLagrangianMesh
        for i in range(block.ndims):
            block.dims[i] = block._data[0].shape[i]
    if isinstance(units, str):
        units = (units,)
    if isinstance(units, (list, tuple)):
        block.dim_units = blocklist._create_id_array(units)
    if isinstance(labels, str):
        labels = (labels,)
    if isinstance(labels, (list, tuple)):
        block.dim_labels = blocklist._create_id_array(labels)
    if isinstance(geometry, str):
        if geometry == "rz":
            geometry = SdfGeometry.CYLINDRICAL
    if isinstance(geometry, int):
        block.geometry = geometry

    blocklist._clib.sdf_set_defaults(blocklist._handle, block)
    _add_post(blocklist, block)


def _add_stitched(
    blocklist,
    name,
    value={},
    id=None,
    mesh_id=None,
    btype=None,
    datatype=None,
    stagger=SdfStagger.HIDDEN0,
    material_id=None,
    material_name=None,
    material_names=None,
):
    if not isinstance(blocklist, BlockList):
        print("ERROR: first argument must be of type BlockList")
        return
    if not isinstance(value, (list, tuple)):
        print("ERROR: invalid value supplied for stitched block")
        return
    if not isinstance(mesh_id, str):
        found_mesh = None
        warn = True
        for val in value:
            if val in blocklist._block_ids:
                tmp = blocklist._block_ids[val]._contents.mesh_id
                if isinstance(tmp, bytes):
                    tmp = tmp.decode()
                    if warn and found_mesh is not None and found_mesh != tmp:
                        print("WARNING: stitched blocks on different meshes")
                        warn = False
                    found_mesh = tmp
            else:
                print(f'WARNING: stitched id "{val}" not found in blocklist')
        if found_mesh is not None:
            mesh_id = found_mesh
        else:
            print("ERROR: no mesh_id supplied for stitched block")
            return
    if id is None:
        id = name
    if datatype is None:
        datatype = SdfDataType.NULL
    h, block = _add_preamble(blocklist, id, name, datatype)

    if btype is None:
        btype = SdfBlockType.STITCHED
    block.blocktype = btype

    if btype in (
        SdfBlockType.CONTIGUOUS_TENSOR,
        SdfBlockType.STITCHED_TENSOR,
    ):
        block.AddBlock = BlockStitchedTensor
    elif btype in (
        SdfBlockType.CONTIGUOUS_MATERIAL,
        SdfBlockType.STITCHED_MATERIAL,
    ):
        block.AddBlock = BlockStitchedMaterial
        block.material_names = blocklist._create_string_array(material_names)
    elif btype in (
        SdfBlockType.CONTIGUOUS_MATVAR,
        SdfBlockType.STITCHED_MATVAR,
    ):
        block.AddBlock = BlockStitchedMatvar
        block.material_id = blocklist._create_id(material_id)
    elif btype in (
        SdfBlockType.CONTIGUOUS_SPECIES,
        SdfBlockType.STITCHED_SPECIES,
    ):
        block.AddBlock = BlockStitchedSpecies
        block.material_id = blocklist._create_id(material_id)
        block.material_name = blocklist._create_string(material_name)
        block.material_names = blocklist._create_id_array(material_names)
    else:
        if stagger in (SdfStagger.HIDDEN0, SdfStagger.HIDDEN2):
            block.AddBlock = BlockStitchedPath
        else:
            block.AddBlock = BlockStitched

    block.stagger = stagger
    nvalue = len(value)
    block.ndims = nvalue
    block.mesh_id = blocklist._create_id(mesh_id)
    block.variable_ids = blocklist._create_id_array(value)
    block._blocklist._block_ids = blocklist._block_ids

    _add_post(blocklist, block)


def _add_stitched_vector(blocklist, name, value={}, id=None, mesh_id=None):
    return _add_stitched(
        blocklist,
        name,
        value,
        id,
        mesh_id,
        btype=SdfBlockType.STITCHED_TENSOR,
    )


def _add_stitched_material(
    blocklist, name, value={}, id=None, mesh_id=None, material_names=None
):
    return _add_stitched(
        blocklist,
        name,
        value,
        id,
        mesh_id,
        material_names=material_names,
        btype=SdfBlockType.STITCHED_MATERIAL,
    )


def _add_stitched_matvar(
    blocklist, name, value={}, id=None, mesh_id=None, material_id=None
):
    return _add_stitched(
        blocklist,
        name,
        value,
        id,
        mesh_id,
        material_id=material_id,
        btype=SdfBlockType.STITCHED_MATVAR,
    )


def _add_stitched_species(
    blocklist,
    name,
    value={},
    id=None,
    mesh_id=None,
    material_id=None,
    material_name=None,
    material_names=None,
):
    return _add_stitched(
        blocklist,
        name,
        value,
        id,
        mesh_id,
        material_id=material_id,
        material_name=material_name,
        material_names=material_names,
        btype=SdfBlockType.STITCHED_SPECIES,
    )


def _add_runinfo(blocklist, name, value=None, **kwargs):
    if not isinstance(blocklist, BlockList):
        print("ERROR: first argument must be of type BlockList")
        return
    id = None
    args = None
    data = None
    if isinstance(value, Block):
        id = value.id
        name = value.name
        data = value._contents.data
    elif isinstance(name, dict) and value is None:
        args = name
        name = None
    else:
        args = value

    if "id" in kwargs:
        id = kwargs["id"]

    if name is None:
        name = "Run_info"
    if id is None:
        id = name.lower()

    datatype = SdfDataType.CHARACTER
    h, block = _add_preamble(blocklist, id, name, datatype)
    block.blocktype = SdfBlockType.RUN_INFO
    block.AddBlock = BlockRunInfo

    block.data = data

    _add_post(blocklist, block, args)


def _copy_block(blocklist, block=None, **kwargs):
    if not block._in_file:
        return

    _ = block.data
    kwargs["value"] = block.data
    kwargs["id"] = block.id
    kwargs["name"] = block.name
    kwargs["datatype"] = SdfDataType(block._contents.datatype)

    if isinstance(block, BlockConstant):
        _add_constant(blocklist, **kwargs)
    elif isinstance(block, BlockNameValue):
        _add_namevalue(blocklist, **kwargs)
    elif isinstance(block, BlockArray):
        _add_array(blocklist, **kwargs)
    elif isinstance(block, BlockRunInfo):
        kwargs["value"] = block
        _add_runinfo(blocklist, **kwargs)
    elif isinstance(block, BlockData):
        kwargs["checksum"] = block.checksum
        kwargs["checksum_type"] = block.checksum_type
        kwargs["mimetype"] = block.mimetype
        del kwargs["datatype"]
        _add_datablock(blocklist, **kwargs)
    elif isinstance(block, BlockPlainVariable):
        kwargs["mult"] = block.mult
        kwargs["units"] = block.units
        kwargs["mesh_id"] = block.grid_id
        kwargs["stagger"] = block.stagger
        if hasattr(block, "species_id"):
            kwargs["species"] = block.species_id
        _add_plainvar(blocklist, **kwargs)
    elif isinstance(block, BlockPlainMesh):
        if len(block.data) > 0:
            kwargs["x"] = block.data[0]
        if len(block.data) > 1:
            kwargs["y"] = block.data[1]
        if len(block.data) > 2:
            kwargs["z"] = block.data[2]
        if hasattr(block, "species_id"):
            kwargs["species"] = block.species_id
        kwargs["units"] = block.units
        kwargs["labels"] = block.labels
        kwargs["geometry"] = block.geometry
        _add_mesh(blocklist, **kwargs)
    elif isinstance(block, BlockStitched):
        b = block._contents
        btype = b.blocktype
        kwargs["mesh_id"] = b.mesh_id.decode()
        kwargs["value"] = [d.id if d else "" for d in block.data]
        kwargs["stagger"] = b.stagger
        kwargs["btype"] = btype

        if btype in (
            SdfBlockType.CONTIGUOUS_MATERIAL,
            SdfBlockType.STITCHED_MATERIAL,
        ):
            kwargs["material_names"] = [
                b.material_names[i].decode() for i in range(b.ndims)
            ]
        elif btype in (
            SdfBlockType.CONTIGUOUS_MATVAR,
            SdfBlockType.STITCHED_MATVAR,
        ):
            kwargs["material_id"] = b.material_id.decode()
        elif btype in (
            SdfBlockType.CONTIGUOUS_SPECIES,
            SdfBlockType.STITCHED_SPECIES,
        ):
            kwargs["material_id"] = b.material_id.decode()
            kwargs["material_name"] = b.material_name.decode()
            kwargs["material_names"] = [
                b.material_names[i].decode() for i in range(b.ndims)
            ]

        _add_stitched(blocklist, **kwargs)
    elif isinstance(block, BlockCpuSplit):
        kwargs["geometry"] = block._contents.geometry
        _add_cpu_split(blocklist, **kwargs)
    else:
        print(
            f'WARNING: block id "{block.id}" of type '
            f'"{type(block).__name__}" not supported'
        )
        return


def _add_block(blocklist, name=None, value=None, id=None, **kwargs):
    if not isinstance(blocklist, BlockList):
        print("ERROR: first argument must be of type BlockList")
        return
    if isinstance(name, Block):
        return _copy_block(blocklist, block=name, value=value, id=id, **kwargs)

    add_func = None
    if isinstance(value, dict):
        val = next(iter(value.values()), None)
        add_func = _add_namevalue
    elif isinstance(value, (tuple, list, _np.ndarray)):
        arr = _np.array(value)
        if arr.ndim == 1:
            val = value[0]
            if isinstance(arr[0], str):
                add_func = _add_stitched
            elif "species" in kwargs or "mesh_id" in kwargs:
                add_func = _add_plainvar
            else:
                add_func = _add_array
        else:
            val = arr.flatten()[0]
            add_func = _add_plainvar
    elif isinstance(value, (str, bytes)):
        val = value
        add_func = _add_datablock
    elif value is not None:
        val = value
        add_func = _add_constant
    else:
        keys = ["x", "y", "z"]
        keys = [k for k in keys if k in kwargs and kwargs[k] is not None]
        if len(keys) > 0:
            val = _np.concatenate([kwargs[k] for k in keys]).flatten()[0]
            add_func = _add_mesh
            if id is None:
                k = "species"
                if k in kwargs:
                    id = f"grid/{kwargs[k]}"
                else:
                    id = "grid"

    if id is None:
        id = name
    if id in blocklist._block_ids:
        print(f'Unable to create block. ID duplicated: "{id}"')
        return

    datatype = None
    if isinstance(val, (bool, _np.bool)):
        datatype = SdfDataType.LOGICAL
    elif isinstance(val, _np.int32):
        datatype = SdfDataType.INTEGER4
    elif isinstance(val, (int, _np.int64)):
        datatype = SdfDataType.INTEGER8
    elif isinstance(val, _np.float32):
        datatype = SdfDataType.REAL4
    elif isinstance(val, float):
        datatype = SdfDataType.REAL8
    elif isinstance(val, str) or isinstance(val, bytes):
        datatype = SdfDataType.CHARACTER
        if add_func not in (
            _add_namevalue,
            _add_datablock,
            _add_stitched,
        ):
            add_func = None
    else:
        add_func = None

    if add_func:
        add_func(
            blocklist, name, value=value, id=id, datatype=datatype, **kwargs
        )
    else:
        print(f'Block "{id}", unsupported datatype: {type(value)}')
        return


def _write(blocklist, filename):
    if not isinstance(blocklist, BlockList):
        print("ERROR: first argument must be of type BlockList")
        return
    if not blocklist._handle:
        return
    for k, b in blocklist._block_ids.items():
        if isinstance(b, Block) and b._contents.in_file:
            _ = b.data
    blocklist._clib.sdf_write(blocklist._handle, filename.encode())
