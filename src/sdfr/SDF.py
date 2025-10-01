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

import ctypes as ct
import numpy as np
import struct
from enum import IntEnum
from .loadlib import sdf_lib

# try:
#    import xarray as xr
#
#    got_xarray = True
# except ImportError:
#    print("WARNING: xarray not installed. Generating plain numpy arrays.")
#    got_xarray = False


# Enum representation using ct
class SdfBlockType(IntEnum):
    SDF_BLOCKTYPE_SCRUBBED = -1
    SDF_BLOCKTYPE_NULL = 0
    SDF_BLOCKTYPE_PLAIN_MESH = 1
    SDF_BLOCKTYPE_POINT_MESH = 2
    SDF_BLOCKTYPE_PLAIN_VARIABLE = 3
    SDF_BLOCKTYPE_POINT_VARIABLE = 4
    SDF_BLOCKTYPE_CONSTANT = 5
    SDF_BLOCKTYPE_ARRAY = 6
    SDF_BLOCKTYPE_RUN_INFO = 7
    SDF_BLOCKTYPE_SOURCE = 8
    SDF_BLOCKTYPE_STITCHED_TENSOR = 9
    SDF_BLOCKTYPE_STITCHED_MATERIAL = 10
    SDF_BLOCKTYPE_STITCHED_MATVAR = 11
    SDF_BLOCKTYPE_STITCHED_SPECIES = 12
    SDF_BLOCKTYPE_SPECIES = 13
    SDF_BLOCKTYPE_PLAIN_DERIVED = 14
    SDF_BLOCKTYPE_POINT_DERIVED = 15
    SDF_BLOCKTYPE_CONTIGUOUS_TENSOR = 16
    SDF_BLOCKTYPE_CONTIGUOUS_MATERIAL = 17
    SDF_BLOCKTYPE_CONTIGUOUS_MATVAR = 18
    SDF_BLOCKTYPE_CONTIGUOUS_SPECIES = 19
    SDF_BLOCKTYPE_CPU_SPLIT = 20
    SDF_BLOCKTYPE_STITCHED_OBSTACLE_GROUP = 21
    SDF_BLOCKTYPE_UNSTRUCTURED_MESH = 22
    SDF_BLOCKTYPE_STITCHED = 23
    SDF_BLOCKTYPE_CONTIGUOUS = 24
    SDF_BLOCKTYPE_LAGRANGIAN_MESH = 25
    SDF_BLOCKTYPE_STATION = 26
    SDF_BLOCKTYPE_STATION_DERIVED = 27
    SDF_BLOCKTYPE_DATABLOCK = 28
    SDF_BLOCKTYPE_NAMEVALUE = 29


class SdfGeometry(IntEnum):
    SDF_GEOMETRY_NULL = 0
    SDF_GEOMETRY_CARTESIAN = 1
    SDF_GEOMETRY_CYLINDRICAL = 2
    SDF_GEOMETRY_SPHERICAL = 3


class SdfStagger(IntEnum):
    SDF_STAGGER_CELL_CENTRE = 0
    SDF_STAGGER_FACE_X = 1
    SDF_STAGGER_FACE_Y = 2
    SDF_STAGGER_FACE_Z = 3
    SDF_STAGGER_EDGE_X = 4
    SDF_STAGGER_EDGE_Y = 5
    SDF_STAGGER_EDGE_Z = 6
    SDF_STAGGER_VERTEX = 7


class SdfDataType(IntEnum):
    SDF_DATATYPE_NULL = 0
    SDF_DATATYPE_INTEGER4 = 1
    SDF_DATATYPE_INTEGER8 = 2
    SDF_DATATYPE_REAL4 = 3
    SDF_DATATYPE_REAL8 = 4
    SDF_DATATYPE_REAL16 = 5
    SDF_DATATYPE_CHARACTER = 6
    SDF_DATATYPE_LOGICAL = 7
    SDF_DATATYPE_OTHER = 8


_np_datatypes = [
    0,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    np.longdouble,
    np.byte,
    np.int32,
    bool,
    0,
]
_ct_datatypes = [
    0,
    ct.c_int32,
    ct.c_int64,
    ct.c_float,
    ct.c_double,
    ct.c_longdouble,
    ct.c_char,
    ct.c_bool,
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
SDF_READ = 1
SDF_WRITE = 2
SDF_MAXDIMS = 4


class SdfBlock(ct.Structure):
    pass  # Forward declaration for self-referencing structure


class SdfFile(ct.Structure):
    pass  # Forward declaration for function pointer compatibility


SdfBlock._fields_ = [
    ("extents", ct.POINTER(ct.c_double)),
    ("dim_mults", ct.POINTER(ct.c_double)),
    ("station_x", ct.POINTER(ct.c_double)),
    ("station_y", ct.POINTER(ct.c_double)),
    ("station_z", ct.POINTER(ct.c_double)),
    ("mult", ct.c_double),
    ("time", ct.c_double),
    ("time_increment", ct.c_double),
    ("dims", ct.c_int64 * SDF_MAXDIMS),
    ("local_dims", ct.c_int64 * SDF_MAXDIMS),
    ("block_start", ct.c_int64),
    ("next_block_location", ct.c_int64),
    ("data_location", ct.c_int64),
    ("inline_block_start", ct.c_int64),
    ("inline_next_block_location", ct.c_int64),
    ("summary_block_start", ct.c_int64),
    ("summary_next_block_location", ct.c_int64),
    ("nelements", ct.c_int64),
    ("nelements_local", ct.c_int64),
    ("data_length", ct.c_int64),
    ("nelements_blocks", ct.POINTER(ct.c_int64)),
    ("data_length_blocks", ct.POINTER(ct.c_int64)),
    ("array_starts", ct.POINTER(ct.c_int64)),
    ("array_ends", ct.POINTER(ct.c_int64)),
    ("array_strides", ct.POINTER(ct.c_int64)),
    ("global_array_starts", ct.POINTER(ct.c_int64)),
    ("global_array_ends", ct.POINTER(ct.c_int64)),
    ("global_array_strides", ct.POINTER(ct.c_int64)),
    ("ndims", ct.c_int32),
    ("geometry", ct.c_int32),
    ("datatype", ct.c_int32),
    ("blocktype", ct.c_int32),
    ("info_length", ct.c_int32),
    ("type_size", ct.c_int32),
    ("stagger", ct.c_int32),
    ("datatype_out", ct.c_int32),
    ("type_size_out", ct.c_int32),
    ("nstations", ct.c_int32),
    ("nvariables", ct.c_int32),
    ("step", ct.c_int32),
    ("step_increment", ct.c_int32),
    ("dims_in", ct.POINTER(ct.c_int32)),
    ("station_nvars", ct.POINTER(ct.c_int32)),
    ("variable_types", ct.POINTER(ct.c_int32)),
    ("station_index", ct.POINTER(ct.c_int32)),
    ("station_move", ct.POINTER(ct.c_int32)),
    ("nm", ct.c_int),
    ("n_ids", ct.c_int),
    ("opt", ct.c_int),
    ("ng", ct.c_int),
    ("nfaces", ct.c_int),
    ("ngrids", ct.c_int),
    ("offset", ct.c_int),
    ("ngb", ct.c_int * 6),
    ("const_value", ct.c_byte * 16),
    ("id", ct.c_char_p),
    ("units", ct.c_char_p),
    ("mesh_id", ct.c_char_p),
    ("material_id", ct.c_char_p),
    ("vfm_id", ct.c_char_p),
    ("obstacle_id", ct.c_char_p),
    ("station_id", ct.c_char_p),
    ("name", ct.c_char_p),
    ("material_name", ct.c_char_p),
    ("must_read", ct.c_char_p),
    ("dim_labels", ct.POINTER(ct.c_char_p)),
    ("dim_units", ct.POINTER(ct.c_char_p)),
    ("station_ids", ct.POINTER(ct.c_char_p)),
    ("variable_ids", ct.POINTER(ct.c_char_p)),
    ("station_names", ct.POINTER(ct.c_char_p)),
    ("material_names", ct.POINTER(ct.c_char_p)),
    ("node_list", ct.POINTER(ct.c_int)),
    ("boundary_cells", ct.POINTER(ct.c_int)),
    ("grids", ct.POINTER(ct.c_void_p)),
    ("data", ct.c_void_p),
    ("done_header", ct.c_bool),
    ("done_info", ct.c_bool),
    ("done_data", ct.c_bool),
    ("dont_allocate", ct.c_bool),
    ("dont_display", ct.c_bool),
    ("dont_own_data", ct.c_bool),
    ("use_mult", ct.c_bool),
    ("next_block_modified", ct.c_bool),
    ("rewrite_metadata", ct.c_bool),
    ("in_file", ct.c_bool),
    ("ng_any", ct.c_bool),
    ("no_internal_ghost", ct.c_bool),
    ("next", ct.POINTER(SdfBlock)),
    ("prev", ct.POINTER(SdfBlock)),
    ("subblock", ct.POINTER(SdfBlock)),
    ("subblock2", ct.POINTER(SdfBlock)),
    (
        "populate_data",
        ct.CFUNCTYPE(
            ct.POINTER(SdfBlock), ct.POINTER(SdfFile), ct.POINTER(SdfBlock)
        ),
    ),
    ("cpu_split", ct.c_int * SDF_MAXDIMS),
    ("starts", ct.c_int * SDF_MAXDIMS),
    ("proc_min", ct.c_int * 3),
    ("proc_max", ct.c_int * 3),
    ("ndim_labels", ct.c_int),
    ("ndim_units", ct.c_int),
    ("nstation_ids", ct.c_int),
    ("nvariable_ids", ct.c_int),
    ("nstation_names", ct.c_int),
    ("nmaterial_names", ct.c_int),
    ("option", ct.c_int),
    ("mimetype", ct.c_char_p),
    ("checksum_type", ct.c_char_p),
    ("checksum", ct.c_char_p),
    ("mmap", ct.c_char_p),
    ("mmap_len", ct.c_int64),
    ("derived", ct.c_bool),
]

SdfFile._fields_ = [
    ("dbg_count", ct.c_int64),
    ("sdf_lib_version", ct.c_int32),
    ("sdf_lib_revision", ct.c_int32),
    ("sdf_extension_version", ct.c_int32),
    ("sdf_extension_revision", ct.c_int32),
    ("file_version", ct.c_int32),
    ("file_revision", ct.c_int32),
    ("dbg", ct.c_char_p),
    ("dbg_buf", ct.c_char_p),
    ("extension_names", ct.POINTER(ct.c_char_p)),
    ("time", ct.c_double),
    ("first_block_location", ct.c_int64),
    ("summary_location", ct.c_int64),
    ("start_location", ct.c_int64),
    ("soi", ct.c_int64),
    ("sof", ct.c_int64),
    ("current_location", ct.c_int64),
    ("jobid1", ct.c_int32),
    ("jobid2", ct.c_int32),
    ("endianness", ct.c_int32),
    ("summary_size", ct.c_int32),
    ("block_header_length", ct.c_int32),
    ("string_length", ct.c_int32),
    ("id_length", ct.c_int32),
    ("code_io_version", ct.c_int32),
    ("step", ct.c_int32),
    ("nblocks", ct.c_int32),
    ("nblocks_file", ct.c_int32),
    ("error_code", ct.c_int32),
    ("rank", ct.c_int),
    ("ncpus", ct.c_int),
    ("ndomains", ct.c_int),
    ("rank_master", ct.c_int),
    ("indent", ct.c_int),
    ("print", ct.c_int),
    ("buffer", ct.c_char_p),
    ("filename", ct.c_char_p),
    ("done_header", ct.c_bool),
    ("restart_flag", ct.c_bool),
    ("other_domains", ct.c_bool),
    ("use_float", ct.c_bool),
    ("use_summary", ct.c_bool),
    ("use_random", ct.c_bool),
    ("station_file", ct.c_bool),
    ("swap", ct.c_bool),
    ("inline_metadata_read", ct.c_bool),
    ("summary_metadata_read", ct.c_bool),
    ("inline_metadata_invalid", ct.c_bool),
    ("summary_metadata_invalid", ct.c_bool),
    ("tmp_flag", ct.c_bool),
    ("metadata_modified", ct.c_bool),
    ("can_truncate", ct.c_bool),
    ("first_block_modified", ct.c_bool),
    ("code_name", ct.c_char_p),
    ("error_message", ct.c_char_p),
    ("blocklist", ct.POINTER(SdfBlock)),
    ("tail", ct.POINTER(SdfBlock)),
    ("current_block", ct.POINTER(SdfBlock)),
    ("last_block_in_file", ct.POINTER(SdfBlock)),
    ("mmap", ct.c_char_p),
    ("ext_data", ct.c_void_p),
    ("stack_handle", ct.c_void_p),
    ("array_count", ct.c_int),
    ("fd", ct.c_int),
    ("purge_duplicated_ids", ct.c_int),
    ("internal_ghost_cells", ct.c_int),
    ("ignore_nblocks", ct.c_int),
]


class RunInfo(ct.Structure):
    _fields_ = [
        ("defines", ct.c_int64),
        ("version", ct.c_int32),
        ("revision", ct.c_int32),
        ("compile_date", ct.c_int32),
        ("run_date", ct.c_int32),
        ("io_date", ct.c_int32),
        ("minor_rev", ct.c_int32),
        ("commit_id", ct.c_char_p),
        ("sha1sum", ct.c_char_p),
        ("compile_machine", ct.c_char_p),
        ("compile_flags", ct.c_char_p),
    ]


class BlockList:
    """Contains all the blocks"""

    def __init__(
        self,
        filename=None,
        convert=False,
        derived=True,
        mode=SDF_READ,
        code_name="sdfr",
        restart=False,
    ):
        self._handle = None
        clib = sdf_lib
        self._clib = clib
        clib.sdf_open.restype = ct.POINTER(SdfFile)
        clib.sdf_open.argtypes = [ct.c_char_p, ct.c_int, ct.c_int, ct.c_int]
        clib.sdf_new.restype = ct.POINTER(SdfFile)
        clib.sdf_new.argtypes = [ct.c_int, ct.c_int]
        clib.sdf_stack_init.argtypes = [ct.c_void_p]
        clib.sdf_read_blocklist.argtypes = [ct.c_void_p]
        clib.sdf_read_blocklist_all.argtypes = [ct.c_void_p]
        clib.sdf_helper_read_data.argtypes = [ct.c_void_p, ct.POINTER(SdfBlock)]
        clib.sdf_free_block_data.argtypes = [ct.c_void_p, ct.POINTER(SdfBlock)]
        clib.sdf_stack_destroy.argtypes = [ct.c_void_p]
        clib.sdf_close.argtypes = [ct.c_void_p]
        clib.sdf_write.argtypes = [ct.c_void_p, ct.c_char_p]
        clib.sdf_get_next_block.argtypes = [ct.c_void_p]
        clib.sdf_set_namevalue.argtypes = [
            ct.POINTER(SdfBlock),
            ct.POINTER(ct.c_char_p),
            ct.POINTER(ct.c_void_p),
        ]
        clib.sdf_set_code_name.argtypes = [ct.c_void_p, ct.c_char_p]
        clib.sdf_set_block_name.argtypes = [
            ct.c_void_p,
            ct.c_char_p,
            ct.c_char_p,
        ]
        clib.sdf_set_defaults.argtypes = [
            ct.c_void_p,
            ct.POINTER(SdfBlock),
        ]
        clib.sdf_create_id.argtypes = [
            ct.c_void_p,
            ct.c_char_p,
        ]
        clib.sdf_create_id.restype = ct.POINTER(ct.c_char_p)
        clib.sdf_create_id_array.argtypes = [
            ct.c_void_p,
            ct.c_int,
            ct.POINTER(ct.c_char_p),
        ]
        clib.sdf_create_id_array.restype = ct.POINTER(ct.c_char_p)

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
        if mode == SDF_READ:
            if derived:
                clib.sdf_read_blocklist_all(h)
            else:
                clib.sdf_read_blocklist(h)
        else:
            clib.sdf_set_code_name(h, code_name.encode("utf-8"))

        block = h.contents.blocklist
        h.contents.restart_flag = restart
        self.Header = get_header(h.contents)
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
            name = get_member_name(block.name)
            if blocktype == SdfBlockType.SDF_BLOCKTYPE_ARRAY:
                newblock = BlockArray(block)
            elif blocktype == SdfBlockType.SDF_BLOCKTYPE_CONSTANT:
                newblock = BlockConstant(block)
            elif (
                blocktype == SdfBlockType.SDF_BLOCKTYPE_CONTIGUOUS
                or blocktype == SdfBlockType.SDF_BLOCKTYPE_STITCHED
            ):
                if block.stagger == 10 or block.stagger == 12:
                    newblock = BlockStitchedPath(block)
                else:
                    newblock = BlockStitched(block)
            elif (
                blocktype == SdfBlockType.SDF_BLOCKTYPE_CONTIGUOUS_MATERIAL
                or blocktype == SdfBlockType.SDF_BLOCKTYPE_STITCHED_MATERIAL
            ):
                newblock = BlockStitchedMaterial(block)
            elif (
                blocktype == SdfBlockType.SDF_BLOCKTYPE_CONTIGUOUS_MATVAR
                or blocktype == SdfBlockType.SDF_BLOCKTYPE_STITCHED_MATVAR
            ):
                newblock = BlockStitchedMatvar(block)
            elif (
                blocktype == SdfBlockType.SDF_BLOCKTYPE_CONTIGUOUS_SPECIES
                or blocktype == SdfBlockType.SDF_BLOCKTYPE_STITCHED_SPECIES
            ):
                newblock = BlockStitchedSpecies(block)
            elif (
                blocktype == SdfBlockType.SDF_BLOCKTYPE_CONTIGUOUS_TENSOR
                or blocktype == SdfBlockType.SDF_BLOCKTYPE_STITCHED_TENSOR
            ):
                newblock = BlockStitchedTensor(block)
            elif blocktype == SdfBlockType.SDF_BLOCKTYPE_DATABLOCK:
                newblock = BlockData(block)
            elif blocktype == SdfBlockType.SDF_BLOCKTYPE_LAGRANGIAN_MESH:
                if block.datatype_out != 0:
                    newblock = BlockLagrangianMesh(block)
                    newblock_mid = block
                    newblock_mid._grid_block = newblock
                    mesh_id_map[newblock.id] = newblock
            elif blocktype == SdfBlockType.SDF_BLOCKTYPE_NAMEVALUE:
                newblock = BlockNameValue(block)
            elif (
                blocktype == SdfBlockType.SDF_BLOCKTYPE_PLAIN_DERIVED
                or blocktype == SdfBlockType.SDF_BLOCKTYPE_PLAIN_VARIABLE
            ):
                newblock = BlockPlainVariable(block)
                mesh_vars.append(newblock)
            elif blocktype == SdfBlockType.SDF_BLOCKTYPE_PLAIN_MESH:
                if block.datatype_out != 0:
                    newblock = BlockPlainMesh(block)
                    newblock_mid = block
                    newblock_mid._grid_block = newblock
                    mesh_id_map[newblock.id] = newblock
            elif (
                blocktype == SdfBlockType.SDF_BLOCKTYPE_POINT_DERIVED
                or blocktype == SdfBlockType.SDF_BLOCKTYPE_POINT_VARIABLE
            ):
                newblock = BlockPointVariable(block)
                mesh_vars.append(newblock)
            elif blocktype == SdfBlockType.SDF_BLOCKTYPE_POINT_MESH:
                newblock = BlockPointMesh(block)
                mesh_id_map[newblock.id] = newblock
            elif blocktype == SdfBlockType.SDF_BLOCKTYPE_RUN_INFO:
                self.Run_info = get_run_info(block)
            elif blocktype == SdfBlockType.SDF_BLOCKTYPE_STATION:
                sdict = BlockStation(block, name)
                self.__dict__.update({"StationBlocks": sdict})
                self._block_ids.update({block.id.decode(): sdict})
                self._block_names.update({block.name.decode(): sdict})
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
                name = get_member_name(block_mid.name) + "_mid"
                if blocktype == SdfBlockType.SDF_BLOCKTYPE_LAGRANGIAN_MESH:
                    newblock = BlockLagrangianMesh(block_mid, mid=True)
                elif blocktype == SdfBlockType.SDF_BLOCKTYPE_PLAIN_MESH:
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

    def write(self, filename):
        if not self._handle:
            return
        self._clib.sdf_write(self._handle, filename.encode())

    def _set_block_name(self, id, name):
        self._clib.sdf_set_block_name(
            self._handle, id.encode("utf-8"), name.encode("utf-8")
        )

    def _create_id(self, values):
        tmp = self._clib.sdf_create_id(self._handle, values.encode("utf-8"))
        return ct.cast(tmp, ct.c_char_p)

    def _string_array_ctype(self, values):
        strings = [s.encode("utf-8") for s in values]
        strings = [ct.create_string_buffer(s) for s in strings]
        strings = [ct.cast(s, ct.c_char_p) for s in strings]
        strings = (ct.c_char_p * len(values))(*strings)
        return strings

    def _create_id_array(self, values):
        values = self._string_array_ctype(values)
        res = self._clib.sdf_create_id_array(self._handle, len(values), values)
        return res

    def _add_preamble(self, id, name, datatype):
        self._clib.sdf_get_next_block(self._handle)
        h = self._handle.contents
        h.nblocks += 1
        h.nblocks_file += 1
        block = h.current_block.contents
        block._handle = self._handle
        block._blocklist = h.blocklist
        block._data = None
        block.datatype = datatype
        block.in_file = 1
        block.AddBlock = None
        self._set_block_name(id, name)
        return h, block

    def _add_post(self, block):
        if block.AddBlock:
            newblock = block.AddBlock(block)
        else:
            return

        id = block.id.decode()
        name = block.name.decode()
        if not block.dont_display:
            self.__dict__[name] = newblock
        if block._data is not None:
            newblock._data = block._data
        self._block_ids.update({id: newblock})
        self._block_names.update({name: newblock})

    def _add_constant(self, name, value=0, datatype=None, id=None):
        if datatype == SdfDataType.SDF_DATATYPE_CHARACTER:
            print(f'Block "{id}", unsupported datatype: {type(value)}')
            return

        h, block = self._add_preamble(id, name, datatype)
        block.blocktype = SdfBlockType.SDF_BLOCKTYPE_CONSTANT
        block.AddBlock = BlockConstant

        const_value = struct.pack(_st_datatypes[block.datatype], value)
        ct.memmove(block.const_value, const_value, 16)

        self._add_post(block)

    def _add_namevalue(self, name, value={}, datatype=None, id=None):
        h, block = self._add_preamble(id, name, datatype)
        block.blocktype = SdfBlockType.SDF_BLOCKTYPE_NAMEVALUE
        block.AddBlock = BlockNameValue

        nvalue = len(value)
        block.ndims = nvalue
        ctype = _ct_datatypes[block.datatype]
        if block.datatype == SdfDataType.SDF_DATATYPE_CHARACTER:
            vals = self._string_array_ctype(value.values())
        else:
            vals = (ctype * nvalue)(*value.values())
        names = self._string_array_ctype(value.keys())
        vals = ct.cast(vals, ct.POINTER(ct.c_void_p))
        self._clib.sdf_set_namevalue(block, names, vals)

        self._add_post(block)

    def _add_array(self, name, value=(), datatype=None, id=None):
        if datatype == SdfDataType.SDF_DATATYPE_CHARACTER:
            print(f'Block "{id}", unsupported datatype: {type(value[0])}')
            return

        h, block = self._add_preamble(id, name, datatype)
        block.blocktype = SdfBlockType.SDF_BLOCKTYPE_ARRAY
        block.AddBlock = BlockArray

        block._data = np.array(value)
        block.ndims = block._data.ndim
        for i in range(block.ndims):
            block.dims[i] = block._data.shape[i]
        block.data = block._data.ctypes.data_as(ct.c_void_p)

        self._add_post(block)

    def _add_plainvar(
        self,
        name,
        value=(),
        datatype=None,
        id=None,
        mult=None,
        units=None,
        mesh_id=None,
        stagger=None,
    ):
        if datatype == SdfDataType.SDF_DATATYPE_CHARACTER:
            print(f'Block "{id}", unsupported datatype: {type(value[0])}')
            return
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

        h, block = self._add_preamble(id, name, datatype)
        block.blocktype = SdfBlockType.SDF_BLOCKTYPE_PLAIN_VARIABLE
        block.AddBlock = BlockPlainVariable

        block._data = np.array(value, order="F")
        block.ndims = block._data.ndim
        for i in range(block.ndims):
            block.dims[i] = block._data.shape[i]
        block.data = block._data.ctypes.data_as(ct.c_void_p)
        if mult is not None:
            block.mult = mult
        if isinstance(units, str):
            block.units = self._create_id(units)
        if isinstance(mesh_id, str):
            block.mesh_id = self._create_id(mesh_id)
        if stagger:
            block.stagger = stagger

        self._clib.sdf_set_defaults(self._handle, block)
        self._add_post(block)

    def _add_mesh(
        self,
        name,
        value=None,
        datatype=None,
        id=None,
        units=None,
        labels=None,
        geometry=None,
        **kwargs,
    ):
        if datatype == SdfDataType.SDF_DATATYPE_CHARACTER:
            print(f'Block "{id}", unsupported datatype: {type(value[0])}')
            return

        h, block = self._add_preamble(id, name, datatype)

        keys = ["x", "y", "z"]
        keys = [k for k in keys if k in kwargs and kwargs[k] is not None]
        val = np.concatenate([kwargs[k] for k in keys]).flatten()[0]

        block._data = [np.array(kwargs[k], dtype=val.dtype) for k in keys]
        block._data = [np.array(row, order="F") for row in block._data]
        block._data = tuple(block._data)
        block.ndims = len(block._data)
        block.ngrids = block.ndims
        grids = [row.ctypes.data_as(ct.c_void_p) for row in block._data]
        block.grids = (ct.c_void_p * block.ngrids)(*grids)
        if block._data[0].ndim == 1:
            block.blocktype = SdfBlockType.SDF_BLOCKTYPE_PLAIN_MESH
            block.AddBlock = BlockPlainMesh
            for i in range(block.ndims):
                block.dims[i] = block._data[i].shape[0]
        else:
            block.blocktype = SdfBlockType.SDF_BLOCKTYPE_LAGRANGIAN_MESH
            block.AddBlock = BlockLagrangianMesh
            for i in range(block.ndims):
                block.dims[i] = block._data[0].shape[i]
        if isinstance(units, (list, tuple)):
            block.dim_units = self._create_id_array(units)
        if isinstance(labels, (list, tuple)):
            block.dim_labels = self._create_id_array(labels)
        if isinstance(geometry, str):
            if geometry == "rz":
                geometry = SdfGeometry.SDF_GEOMETRY_CYLINDRICAL
        if isinstance(geometry, int):
            block.geometry = geometry

        self._clib.sdf_set_defaults(self._handle, block)
        self._add_post(block)

    def add_block(self, name, value=None, id=None, **kwargs):
        if isinstance(value, dict):
            val = next(iter(value.values()), None)
            add_func = self._add_namevalue
        elif isinstance(value, (tuple, list, np.ndarray)):
            arr = np.array(value)
            if arr.ndim == 1:
                val = value[0]
                add_func = self._add_array
            else:
                val = arr.flatten()[0]
                add_func = self._add_plainvar
        elif value is not None:
            val = value
            add_func = self._add_constant
        else:
            keys = ["x", "y", "z"]
            keys = [k for k in keys if k in kwargs and kwargs[k] is not None]
            val = np.concatenate([kwargs[k] for k in keys]).flatten()[0]
            add_func = self._add_mesh
            if id is None:
                id = "grid"

        if id is None:
            id = name
        if id in self._block_ids:
            print(f'Unable to create block. ID duplicated: "{id}"')
            return

        datatype = None
        if isinstance(val, bool):
            datatype = SdfDataType.SDF_DATATYPE_LOGICAL
        elif isinstance(val, np.int32):
            datatype = SdfDataType.SDF_DATATYPE_INTEGER4
        elif isinstance(val, (int, np.int64)):
            datatype = SdfDataType.SDF_DATATYPE_INTEGER8
        elif isinstance(val, np.float32):
            datatype = SdfDataType.SDF_DATATYPE_REAL4
        elif isinstance(val, float):
            datatype = SdfDataType.SDF_DATATYPE_REAL8
        elif isinstance(val, str):
            datatype = SdfDataType.SDF_DATATYPE_CHARACTER
        else:
            add_func = None

        if add_func:
            add_func(name, value=value, id=id, datatype=datatype, **kwargs)
        else:
            print(f'Block "{id}", unsupported datatype: {type(value)}')
            return

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
        self._id = block.id.decode()
        self._name = block.name.decode()
        self._datatype = _np_datatypes[block.datatype_out]
        self._data_length = block.data_length
        self._dims = tuple(block.dims[: block.ndims])
        self._contents = block
        self._blocklist = block._blocklist
        self._data = None

    def _numpy_from_buffer(self, data, blen):
        buffer_from_memory = ct.pythonapi.PyMemoryView_FromMemory
        buffer_from_memory.restype = ct.py_object
        dtype = self._datatype
        if dtype == np.byte:
            dtype = np.dtype("|S1")
        totype = _ct_datatypes[self._contents.datatype_out]
        cast = ct.cast(data, ct.POINTER(totype))
        buf = buffer_from_memory(cast, blen)
        return np.frombuffer(buf, dtype)

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
            blen = np.dtype(self._datatype).itemsize
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
                blen = np.dtype(self._datatype).itemsize * d
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
            blen = np.dtype(self._datatype).itemsize
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
            if block.datatype == SdfDataType.SDF_DATATYPE_CHARACTER:
                p = ct.cast(block.data, ct.POINTER(ct.c_char_p))
                val = p[n].decode()
            else:
                dt = _ct_datatypes[block.datatype]
                val = ct.cast(block.data, ct.POINTER(dt))[n]
            nid = get_member_name(block.material_names[n])
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
            blen = np.dtype(self._datatype).itemsize
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
            _data = ct.cast(self._contents.data, ct.POINTER(ct.c_char * blen))
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


def BlockStation(block, name):
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
                    self._data.append(self._blocklist._block_ids[vid])
        return self._data


class BlockStitchedPath(BlockStitched):
    """Stitched path block"""

    pass


class BlockStitchedMaterial(BlockStitched):
    """Stitched material block"""

    pass


class BlockStitchedMatvar(BlockStitched):
    """Stitched material variable block"""

    pass


class BlockStitchedSpecies(BlockStitched):
    """Stitched species block"""

    pass


class BlockStitchedTensor(BlockStitched):
    """Stitched tensor block"""

    pass


def get_header(h):
    d = {}
    if h.filename:
        d["filename"] = h.filename.decode()
    d["file_version"] = h.file_version
    d["file_revision"] = h.file_revision
    if h.code_name:
        d["code_name"] = h.code_name.decode()
    d["step"] = h.step
    d["time"] = h.time
    d["jobid1"] = h.jobid1
    d["jobid2"] = h.jobid2
    d["code_io_version"] = h.code_io_version
    d["restart_flag"] = h.restart_flag
    d["other_domains"] = h.other_domains
    d["station_file"] = h.station_file
    return d


def get_run_info(block):
    import datetime
    from datetime import datetime as dtm

    utc = datetime.timezone.utc

    h = ct.cast(block.data, ct.POINTER(RunInfo)).contents
    d = {}
    d["version"] = f"{h.version}.{h.revision}.{h.minor_rev}"
    d["commit_id"] = h.commit_id.decode()
    d["sha1sum"] = h.sha1sum.decode()
    d["compile_machine"] = h.compile_machine.decode()
    d["compile_flags"] = h.compile_flags.decode()
    d["compile_date"] = dtm.fromtimestamp(h.compile_date, utc).strftime("%c")
    d["run_date"] = dtm.fromtimestamp(h.run_date, utc).strftime("%c")
    d["io_data"] = dtm.fromtimestamp(h.io_date, utc).strftime("%c")
    return d


def get_member_name(name):
    sname = name.decode()
    return "".join(
        [
            (
                i
                if (
                    (i >= "a" and i <= "z")
                    or (i >= "A" and i <= "Z")
                    or (i >= "0" and i <= "9")
                )
                else "_"
            )
            for i in sname
        ]
    )


def read(file=None, convert=False, mmap=0, dict=False, derived=True):
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


def new(dict=False, code_name="sdfr", restart=False):
    """Creates a new SDF blocklist and returns a dictionary of NumPy arrays.

    Parameters
    ----------
    dict : bool, optional
        Return file contents as a dictionary rather than member names.
    """

    blocklist = BlockList(mode=SDF_WRITE, code_name=code_name, restart=restart)

    if isinstance(dict, str):
        if dict == "id" or dict == "ids":
            return blocklist._block_ids
    elif isinstance(dict, bool) and dict:
        return blocklist._block_names

    return blocklist
