import logging
import os
import re
import stat
import sys
import time
from ctypes import (
    CDLL,
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_float,
    c_int,
    c_int32,
    c_int64,
    c_uint64,
    sizeof,
)
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np

from .fz_config import FemzipArrayType, FemzipVariableCategory, get_last_int_of_line

# During next refactoring we should take a look at reducing the file size.
# pylint: disable = too-many-lines

# The c-structs python wrappers set variables outside of the init method which
# is okay.
# pylint: disable = attribute-defined-outside-init


class FemzipException(Exception):
    """Custom exception specifically for anything going wrong in femzip"""


class FemzipError(Structure):
    """Struct representing femzip errors in c-code

    Attributes
    ----------
    ier: c_int32
        Error code
    msg: c_char_p
        Error message
    """

    _fields_ = [
        ("ier", c_int32),
        ("msg", c_char_p),
    ]


class VariableInfo(Structure):
    """Struct for details about a single femzip variable

    Attributes
    ----------
    var_index: c_int32
        Index of the variable
    name: c_char_p
        Name from femzip
    var_type: c_int32
        Variable type. See FemzipVariableCategory for translation.
    var_size: c_int32
        Array size of the field variable.
    """

    _fields_ = [
        ("var_index", c_int32),
        ("name", c_char_p),
        ("var_type", c_int32),
        ("var_size", c_int32),
    ]


class FemzipFileMetadata(Structure):
    """This struct contains metadata about femzip files.

    Attributes
    ----------
    version_zip: c_float
    activity_flag: c_int32
    number_of_variables: c_int32
    number_of_nodes: c_int32
    number_of_solid_elements: c_int32
    number_of_thick_shell_elements: c_int32
    number_of_1D_elements: c_int32
    number_of_tool_elements: c_int32
    number_of_shell_elements: c_int32
    number_of_solid_element_neighbors: c_int32
    number_of_rbe_element_neighbors: c_int32
    number_of_bar_elements: c_int32
    number_of_beam_elements: c_int32
    number_of_plotel_elements: c_int32
    number_of_spring_elements: c_int32
    number_of_damper_elements: c_int32
    number_of_joint_elements: c_int32
    number_of_joint_element_neighbors: c_int32
    number_of_bar_element_neighbors: c_int32
    number_of_beamcross_elements: c_int32
    number_of_spotweld_elements: c_int32
    number_of_rbe_elements: c_int32
    number_of_hexa20_elements: c_int32
    number_of_rigid_shell_elements: c_int32
    number_of_timesteps: c_int32
    variable_infos: POINTER(VariableInfo)
    """

    _fields_ = [
        ("version_zip", c_float),
        ("activity_flag", c_int32),
        ("number_of_variables", c_int32),
        ("number_of_nodes", c_int32),
        ("number_of_solid_elements", c_int32),
        ("number_of_thick_shell_elements", c_int32),
        ("number_of_1D_elements", c_int32),
        ("number_of_tool_elements", c_int32),
        ("number_of_shell_elements", c_int32),
        ("number_of_solid_element_neighbors", c_int32),
        ("number_of_rbe_element_neighbors", c_int32),
        ("number_of_bar_elements", c_int32),
        ("number_of_beam_elements", c_int32),
        ("number_of_plotel_elements", c_int32),  # NOTE typo?
        ("number_of_spring_elements", c_int32),
        ("number_of_damper_elements", c_int32),
        ("number_of_joint_elements", c_int32),
        ("number_of_joint_element_neighbors", c_int32),
        ("number_of_bar_element_neighbors", c_int32),
        ("number_of_beamcross_elements", c_int32),
        ("number_of_spotweld_elements", c_int32),
        ("number_of_rbe_elements", c_int32),
        ("number_of_hexa20_elements", c_int32),
        ("number_of_rigid_shell_elements", c_int32),
        ("number_of_timesteps", c_int32),
        ("variable_infos", POINTER(VariableInfo)),
    ]


class FemzipBufferInfo(Structure):
    """This struct describes necessary buffer sizes for reading the file

    Attributes
    ----------
    n_timesteps: c_uint64
        Number of timesteps
    timesteps: POINTER(c_float)
        Time for each timestep
    size_geometry: c_uint64
        Size of the geometry buffer
    size_state: c_uint64
        Size of the state buffer
    size_displacement: c_uint64
        Size for displacement array
    size_activity: c_uint64
        Size for activity array (deletion stuff)
    size_post: c_uint64
        Size of the post region of which I currently don't know anymore what it
        was.
    size_titles: c_uint64
        Size of the titles region behind the geomtry.
    """

    _fields_ = [
        ("n_timesteps", c_uint64),
        ("timesteps", POINTER(c_float)),
        ("size_geometry", c_uint64),
        ("size_state", c_uint64),
        ("size_displacement", c_uint64),
        ("size_activity", c_uint64),
        ("size_post", c_uint64),
        ("size_titles", c_uint64),
    ]


class FemzipAPIStatus(Structure):
    """This struct summarizes the state of the femzip API library. The library
    has a shared, global state which is stored in static variables. The state
    of the gloval vars is tracked by this struct.

    Attributes
    ----------
    is_file_open: c_int32
        Whether a femzip file is opened and being processed.
    is_geometry_read: c_int32
        Whether the geometry was already read.
    is_states_open: c_int32
        Whether processing of the states was started.
    i_timestep_state: c_int32
        Counter of timestep processing.
    i_timestep_activity: c_int32
        Counter of activity data for timesteps.
    """

    _fields_ = [
        ("is_file_open", c_int32),
        ("is_geometry_read", c_int32),
        ("is_states_open", c_int32),
        ("i_timestep_state", c_int32),
        ("i_timestep_activity", c_int32),
    ]


class FemzipAPI:
    """FemzipAPI contains wrapper functions around the femzip library."""

    _api: Union[None, CDLL] = None

    @staticmethod
    def load_dynamic_library(path: str) -> CDLL:
        """Load a library and check for correct execution

        Parameters
        ----------
        path: str
            path to the library

        Returns
        -------
        library: CDLL
            loaded library
        """

        # check executable rights
        if not os.access(path, os.X_OK) or not os.access(path, os.R_OK):
            os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IREAD)
            if not os.access(path, os.X_OK) or not os.access(path, os.R_OK):
                err_msg = "Library '{0}' is not executable and couldn't change execution rights."
                raise RuntimeError(err_msg.format(path))

        return CDLL(path)

    @property
    def api(self) -> CDLL:
        """Returns the loaded, shared object library of the native interface

        Returns
        -------
        shared_object_lib: CDLL
            Loaded shared object library.
        """

        # pylint: disable = too-many-statements

        if self._api is None:

            bin_dirpath = (
                os.path.abspath(os.path.dirname(sys.executable))
                if hasattr(sys, "frozen")
                else os.path.dirname(os.path.abspath(__file__))
            )

            # Flexlm Settings
            # prevent flexlm gui to pop up
            os.environ["FLEXLM_BATCH"] = "1"
            # set a low timeout from originally 10 seconds
            if "FLEXLM_TIMEOUT" not in os.environ:
                os.environ["FLEXLM_TIMEOUT"] = "200000"

            # windows
            if "win32" in sys.platform:

                shared_lib_name = "api_extended.dll"
                self.load_dynamic_library(os.path.join(bin_dirpath, "libmmd.dll"))
                self.load_dynamic_library(os.path.join(bin_dirpath, "libifcoremd.dll"))
                self.load_dynamic_library(os.path.join(bin_dirpath, "libifportmd.dll"))
                self.load_dynamic_library(os.path.join(bin_dirpath, "libiomp5md.dll"))
                self.load_dynamic_library(
                    os.path.join(bin_dirpath, "femzip_a_dyna_sidact_generic.dll")
                )
                self.load_dynamic_library(
                    os.path.join(bin_dirpath, "libfemzip_post_licgenerator_ext_flexlm.dll")
                )
            # linux hopefully
            else:
                shared_lib_name = "api_extended.so"
                self.load_dynamic_library(os.path.join(bin_dirpath, "libiomp5.so"))
                self.load_dynamic_library(os.path.join(bin_dirpath, "libintlc.so.5"))
                self.load_dynamic_library(os.path.join(bin_dirpath, "libirng.so"))
                self.load_dynamic_library(os.path.join(bin_dirpath, "libimf.so"))
                self.load_dynamic_library(os.path.join(bin_dirpath, "libsvml.so"))
                self.load_dynamic_library(
                    os.path.join(bin_dirpath, "libfemzip_a_dyna_sidact_generic.so")
                )
                self.load_dynamic_library(
                    os.path.join(bin_dirpath, "libfemzip_post_licgenerator_ext_flexlm.so")
                )

            filepath = os.path.join(bin_dirpath, shared_lib_name)
            self._api = self.load_dynamic_library(filepath)

            # license check
            self._api.has_femunziplib_license.restype = c_int

            # file check
            self._api.is_sidact_file.argtypes = (c_char_p,)
            self._api.is_sidact_file.restype = c_int

            # content infos
            self._api.get_file_metadata.argtypes = (c_char_p, POINTER(FemzipFileMetadata))
            self._api.get_file_metadata.restype = FemzipError

            # free
            self._api.free_variable_array.argtypes = (POINTER(FemzipFileMetadata),)
            self._api.free_variable_array.restype = c_int32

            # get buffer dims
            self._api.get_buffer_info.argtypes = (c_char_p, POINTER(FemzipBufferInfo))
            self._api.get_buffer_info.restype = FemzipError

            # read geom
            self._api.read_geometry.argtypes = (
                c_char_p,
                POINTER(FemzipBufferInfo),
                POINTER(c_int32),
                c_int32,
            )
            self._api.read_geometry.restype = FemzipError

            # read var
            self._api.read_variables.argtypes = (
                POINTER(c_float),
                c_int,
                c_int,
                POINTER(FemzipFileMetadata),
            )
            self._api.read_variables.restype = FemzipError

            # femunzip version
            self._api.is_femunzip_version_ok.argtypes = (c_char_p, POINTER(c_int))
            self._api.is_femunzip_version_ok.restype = FemzipError

            # femzip status
            self._api.get_femzip_status.argtypes = tuple()
            self._api.get_femzip_status.restype = FemzipAPIStatus

            # get part titles
            self._api.get_part_titles.argtypes = (c_char_p, POINTER(c_int32), c_int32)
            self._api.get_part_titles.restype = FemzipError

            # finish reading states
            self._api.finish_reading_states.argtypes = (POINTER(c_int32), c_int64)
            self._api.finish_reading_states.restype = FemzipError

            # close file
            self._api.close_current_file.argtypes = tuple()
            self._api.close_current_file.restype = FemzipError

            # read single state
            self._api.read_single_state.argtypes = (c_int32, c_int32, POINTER(c_float), c_int64)
            self._api.read_single_state.restype = FemzipError

            # read state activity
            self._api.read_activity.argtypes = (c_int32, c_int32, POINTER(c_float))
            self._api.read_activity.restype = FemzipError

            # free buffer info
            self._api.free_buffer_info.argtypes = (POINTER(FemzipBufferInfo),)
            self._api.free_buffer_info.restype = c_int32

        return self._api

    @staticmethod
    def _parse_state_filter(state_filter: Union[Set[int], None], n_timesteps: int) -> Set[int]:

        # convert negative indexes
        state_filter_parsed = (
            {entry if entry >= 0 else entry + n_timesteps for entry in state_filter}
            if state_filter is not None
            else set(range(n_timesteps))
        )

        # filter invalid indexes
        state_filter_valid = {entry for entry in state_filter_parsed if 0 <= entry < n_timesteps}

        return state_filter_valid

    @staticmethod
    def _check_femzip_error(err: FemzipError) -> None:
        """Checks a femzip error coming from C (usually)

        Parameters
        ----------
        err: FemzipError
            c struct error

        Raises
        ------
        FemzipException
            If the error flag is set with the corresponding
            error message.
        """
        if err.ier != 0:
            fz_error_msg = "Unknown"
            try:
                fz_error_msg = err.msg.decode("ascii")
            except ValueError:
                pass

            err_msg = "Error Code '{0}': {1}"
            raise FemzipException(err_msg.format(err.ier, fz_error_msg))

    @staticmethod
    def struct_to_dict(struct: Structure) -> Dict[str, Any]:
        """Converts a ctypes struct into a dict

        Parameters
        ----------
        struct: Structure

        Returns
        -------
        fields: Dict[str, Any]
            struct as dict

        Examples
        --------
            >>> api.struct_to_dict(api.get_femzip_status())
            {'is_file_open': 1, 'is_geometry_read': 1, 'is_states_open': 0,
            'i_timestep_state': -1, 'i_timestep_activity': -1}
        """
        # We access some internal members to do some magic.
        # pylint: disable = protected-access
        return {field_name: getattr(struct, field_name) for field_name, _ in struct._fields_}

    @staticmethod
    def copy_struct(src: Structure, dest: Structure):
        """Copies all fields from src struct to dest

        Parameters
        ----------
        src: Structure
            src struct
        src: Structure
            destination struct

        Examples
        --------
            >>> err1 = FemzipError()
            >>> err1.ier = -1
            >>> err1.msg = b"Oops"
            >>> err2 = FemzipError()
            >>> api.copy_struct(err1, err2)
            >>> err2.ier
            -1
            >>> err2.msg
            b'Oops'
        """
        # We access some internal members to do some magic.
        # pylint: disable = protected-access
        assert src._fields_ == dest._fields_

        for field_name, _ in src._fields_:
            setattr(dest, field_name, getattr(src, field_name))

    def get_part_titles(
        self, filepath: str, buffer_info: Union[None, FemzipBufferInfo] = None
    ) -> memoryview:
        """Get the part title section

        Parameters
        ----------
        filepath: str
            path to femzip file
        buffer_info: Union[None, FemzipBufferInfo]
            buffer info if previously fetched

        Returns
        -------
        mview: memoryview
            memory of the part title section
        """

        # find out how much memory to allocate
        buffer_info_parsed = self.get_buffer_info(filepath) if buffer_info is None else buffer_info

        # allocate memory
        # pylint: disable = invalid-name
        BufferType = c_int32 * (buffer_info_parsed.size_titles)
        buffer = BufferType()

        # do the thing
        err = self.api.get_part_titles(
            filepath.encode("utf-8"),
            buffer,
            buffer_info_parsed.size_titles,
        )
        self._check_femzip_error(err)

        return memoryview(buffer).cast("B")

    def read_state_deletion_info(
        self, buffer_info: FemzipBufferInfo, state_filter: Union[Set[int], None] = None
    ) -> np.ndarray:
        """Get information which elements are alive

        Parameters
        ----------
        buffer_info: FemzipBufferInfo
            infos about buffer sizes
        state_filter: Union[Set[int], None]
            usable to read only specific states

        Notes
        -----
            The `buffer` must have the size of at least
            `buffer_info.size_activity`.

        Examples
        --------
            >>> # get info about required memory
            >>> buffer_info = api.get_buffer_info(filepath)

            >>> # first read geometry and leave file open!
            >>> mview_geom = api.read_geometry(filepath, buffer_info, False)

            >>> # now read deletion info
            >>> array_deletion = api.read_state_activity(buffer_info)

            >>> # close file
            >>> api.close_current_file()
        """

        logging.debug("FemzipAPI.read_state_deletion_info start")

        # filter timesteps
        state_filter_valid = self._parse_state_filter(state_filter, buffer_info.n_timesteps)
        logging.debug("state filter: %s", state_filter_valid)

        # allocate memory
        # pylint: disable = invalid-name
        StateBufferType = c_float * buffer_info.size_activity
        BufferType = c_float * (buffer_info.size_activity * len(state_filter_valid))
        buffer_c = BufferType()

        # major looping
        n_timesteps_read = 0
        for i_timestep in range(buffer_info.n_timesteps):
            logging.debug("i_timestep %d", i_timestep)

            # walk forward in buffer
            state_buffer_ptr = StateBufferType.from_buffer(
                buffer_c, sizeof(c_float) * buffer_info.size_activity * n_timesteps_read
            )

            # do the thing
            err = self.api.read_activity(i_timestep, buffer_info.size_activity, state_buffer_ptr)
            self._check_femzip_error(err)

            # increment buffer ptr if we needed this one
            if i_timestep in state_filter_valid:
                logging.debug("saved")
                n_timesteps_read += 1
                state_filter_valid.remove(i_timestep)

            # we processe what we need
            if not state_filter_valid:
                break

        # convert buffer into array
        array = np.frombuffer(buffer_c, dtype=np.float32).reshape(
            (n_timesteps_read, buffer_info.size_activity)
        )

        logging.debug("FemzipAPI.read_state_deletion_info end")

        return array

        # return memoryview(buffer_c).cast('B')

    def read_single_state(
        self,
        i_timestep: int,
        buffer_info: FemzipBufferInfo,
        state_buffer: Union[None, memoryview] = None,
    ) -> memoryview:
        """Read a single state

        Parameters
        ----------
        i_timestep: int
            timestep to be read
        buffer_info: FemzipBufferInfo
            infos about buffer sizes
        state_buffer: Union[None, memoryview]
            buffer in which the states are stored

        Notes
        -----
            It is unclear to us why the state buffer needs to be given
            in order to terminate state reading.

        Examples
        --------
            >>> # get info about required memory
            >>> buffer_info = api.get_buffer_info(filepath)

            >>> # first read geometry and leave file open
            >>> mview_geom = api.read_geometry(filepath, buffer_info, False)

            >>> # now read a state
            >>> mview_state = api.read_single_state(0, buffer_info=buffer_info)

            >>> # close file
            >>> api.close_current_file()
        """

        if state_buffer is not None and "f" not in state_buffer.format:
            err_msg = "The state buffer must have a float format '<f' instead of '{0}'."
            raise ValueError(err_msg.format(state_buffer.format))

        # pylint: disable = invalid-name
        StateBufferType = c_float * buffer_info.size_state
        state_buffer_c = (
            StateBufferType() if state_buffer is None else StateBufferType.from_buffer(state_buffer)
        )

        err = self.api.read_single_state(
            i_timestep, buffer_info.n_timesteps, state_buffer_c, buffer_info.size_state
        )
        self._check_femzip_error(err)

        return memoryview(state_buffer_c).cast("B")

    def close_current_file(self) -> None:
        """Closes the current file handle(use not recommended)

        Notes
        -----
            Closes a currently opened file by the API. There
            is no arg because femzip can process only one file
            at a time.
            This can also be used in case of bugs.

        Examples
        --------
            >>> api.close_current_file()
        """
        err = self.api.close_current_file()
        self._check_femzip_error(err)

    def get_femzip_status(self) -> FemzipAPIStatus:
        """Check the status of the femzip api

        Returns
        -------
        femzip_status: FemzipAPIStatus
            c struct with info about femzip API

        Notes
        -----
            This reports whether a file is currently
            opened and how far it was processed. This
            internal state is used to avoid internal
            conflicts and crashes, thus is useful for
            debugging.

        Examples
        --------
            >>> print(api.struct_to_dict(api.get_femzip_status()))
            {'is_file_open': 0, 'is_geometry_read': 0, 'is_states_open': 0,
            'i_timestep_state': -1, 'i_timestep_activity': -1}
        """
        return self.api.get_femzip_status()

    def is_femunzip_version_ok(self, filepath: str) -> bool:
        """Checks if the femunzip version can be handled

        Parameters
        ----------
        filepath: str
            path to the femzpi file

        Returns
        -------
        version_ok: bool

        Examples
        --------
            >>> api.is_femunzip_version_ok("path/to/d3plot.fz")
            True
        """
        is_ok = c_int(-1)
        err = self.api.is_femunzip_version_ok(filepath.encode("ascii"), byref(is_ok))
        self._check_femzip_error(err)
        return is_ok.value == 1

    def has_femunziplib_license(self) -> bool:
        """Checks whether the extended libraries are available

        Returns
        -------
        has_license: bool

        Examples
        --------
            >>> api.has_femunziplib_license()
            False
        """
        start_time = time.time()
        has_license = self.api.has_femunziplib_license() == 1
        logging.debug("License check duration: %fs", (time.time() - start_time))
        return has_license

    def is_sidact_file(self, filepath: str) -> bool:
        """Tests if a filepath points at a sidact file

        Parameters
        ----------
        filepath: path to file

        Returns
        -------
        is_sidact_file: bool

        Examples
        --------
            >>> api.is_sidact_file("path/to/d3plot.fz")
            True
            >>> api.is_sidact_file("path/to/d3plot")
            False
            >>> api.is_sidact_file("path/to/non/existing/file")
            False
        """
        return self.api.is_sidact_file(filepath.encode("ascii")) == 1

    def get_buffer_info(self, filepath: str) -> FemzipBufferInfo:
        """Get the dimensions of the buffers for femzip

        Parameters
        ----------
        filepath: str
            path to femzip file

        Returns
        -------
        buffer_info: FemzipBufferInfo
            c struct with infos about the memory required by femzip

        Examples
        --------
            >>> # read memory demand info first
            >>> buffer_info = api.get_buffer_info(filepath)
            >>> # buffer info is a c struct, but we can print it
            >>> api.struct_to_dict(buffer_info)
            {'n_timesteps': 12,
            'timesteps': <lasso.femzip.femzip_api.LP_c_float object at 0x0000028A8F6B21C0>,
            'size_geometry': 537125, 'size_state': 1462902, 'size_displacement': 147716,
            'size_activity': 47385, 'size_post': 1266356, 'size_titles': 1448}
            >>> for i_timestep in range(buffer_info.n_timesteps):
            >>>     print(buffer_info.timesteps[i_timestep])
            0.0
            0.9998100399971008
            1.9998900890350342
            2.9999701976776123
            3.9997801780700684
        """
        buffer_info = FemzipBufferInfo()

        err = self.api.get_buffer_info(
            filepath.encode("ascii"),
            byref(buffer_info),
        )
        self._check_femzip_error(err)

        # we need to copy the timesteps from C to Python
        buffer_info_2 = FemzipBufferInfo()

        # pylint: disable = invalid-name
        TimestepsType = c_float * buffer_info.n_timesteps
        timesteps_buffer = TimestepsType()
        for i_timestep in range(buffer_info.n_timesteps):
            timesteps_buffer[i_timestep] = buffer_info.timesteps[i_timestep]
        buffer_info_2.timesteps = timesteps_buffer

        self.copy_struct(buffer_info, buffer_info_2)
        buffer_info_2.timesteps = timesteps_buffer

        # free C controlled memory
        self.api.free_buffer_info(byref(buffer_info))

        return buffer_info_2

    def read_geometry(
        self,
        filepath: str,
        buffer_info: Union[FemzipBufferInfo, None] = None,
        close_file: bool = True,
    ) -> memoryview:
        """Read the geometry buffer from femzip

        Parameters
        ----------
        filepath: str
            path to femzpi file
        buffer_info: Union[FemzipBufferInfo, None]
            struct with info regarding required memory for femzip
        close_file: bool
            it is useful to leave the file open if
            states are processed right afterwards

        Returns
        -------
        buffer: memoryview
            memoryview of buffer

        Notes
        -----
            If the file isn't closed appropriately bugs and crashes
            might occur.

        Examples
        --------
            >>> mview = api.read_geometry(filepath, buffer_info)
        """

        # find out how much memory to allocate
        buffer_info = self.get_buffer_info(filepath) if buffer_info is None else buffer_info

        # allocate memory
        # pylint: disable = invalid-name
        GeomBufferType = c_int * (buffer_info.size_geometry + buffer_info.size_titles)
        buffer = GeomBufferType()

        # read geometry
        err = self.api.read_geometry(
            filepath.encode("ascii"),
            byref(buffer_info),
            buffer,
            c_int32(close_file),
        )

        self._check_femzip_error(err)

        return memoryview(buffer).cast("B")

    def read_states(
        self,
        filepath: str,
        buffer_info: Union[FemzipBufferInfo, None] = None,
        state_filter: Union[Set[int], None] = None,
    ) -> np.ndarray:
        """Reads all femzip state information

        Parameters
        ----------
        filepath: str
            path to femzip file
        buffer_info: Union[FemzipBufferInfo, None]
            struct with info regarding required memory for femzip
        state_filter: Union[Set[int], None]
            usable to load only specific states

        Returns
        -------
        buffer: memoryview
            buffer containing all state data

        Examples
        --------
            >>> buffer_info = api.get_buffer_info("path/to/d3plot.fz")
            >>> array_states = api.read_states("path/to/d3plot.fz", buffer_info)
        """

        buffer_info_parsed = self.get_buffer_info(filepath) if buffer_info is None else buffer_info

        # filter invalid indexes
        state_filter_valid = self._parse_state_filter(state_filter, buffer_info_parsed.n_timesteps)

        n_states_to_allocate = (
            buffer_info_parsed.n_timesteps if state_filter is None else len(state_filter_valid)
        )

        # allocate buffer
        # pylint: disable = invalid-name
        BufferType = c_float * (buffer_info_parsed.size_state * n_states_to_allocate)
        buffer = BufferType()

        n_timesteps_read = 0
        for i_timestep in range(buffer_info_parsed.n_timesteps):

            # forward pointer in buffer
            buffer_state = buffer[buffer_info.size_state * n_timesteps_read]

            # read state data
            self.read_single_state(i_timestep, buffer_info_parsed, buffer_state)

            if i_timestep in state_filter_valid:
                n_timesteps_read += 1
                state_filter_valid.remove(i_timestep)

            if not state_filter_valid:
                break

        array = np.from_buffer(buffer, dtype=np.float32).reshape(
            (n_timesteps_read, buffer_info_parsed.size_state)
        )

        return array

    def get_file_metadata(self, filepath: str) -> FemzipFileMetadata:
        """Get infos about the femzip variables in the file

        Parameters
        ----------
        filepath: str
            path to femzip file

        Returns
        -------
        file_metadata: FemzipFileMetadata
            c struct with infos about the femzip file

        Notes
        -----
            This is for direct interaction with the C-API, thus should
            not be used by users.

        Examples
        --------
            >>> file_metadata = api.get_file_metadata("path/to/d3plot.fz")
            >>> # print general internals
            >>> api.struct_to_dict(file_metadata)
            {'version_zip': 605.0, 'activity_flag': 1, 'number_of_variables': 535, ...}

            >>> # We can iterate the variable names contained in the file
            >>> print(
                [file_metadata.variable_infos[i_var].name.decode("utf8").strip()
                for i_var in range(file_metadata.number_of_variables)]
            )
            ['global', 'Parts: Energies and others', 'coordinates', 'velocities', ...]
        """
        file_metadata = FemzipFileMetadata()

        # get variable infos
        err = self.api.get_file_metadata(filepath.encode("ascii"), byref(file_metadata))
        self._check_femzip_error(err)

        # transfer memory to python
        file_metadata2 = self._copy_variable_info_array(file_metadata)

        # release c memory
        self.api.free_variable_array(byref(file_metadata))

        return file_metadata2

    def _get_variables_state_buffer_size(
        self,
        n_parts: int,
        n_rigid_walls: int,
        n_rigid_wall_vars: int,
        n_airbag_particles: int,
        n_airbags: int,
        file_metadata: FemzipFileMetadata,
    ) -> int:
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements

        buffer_size_state = 0
        var_indexes_to_remove: Set[int] = set()
        for i_var in range(file_metadata.number_of_variables):
            var_info = file_metadata.variable_infos[i_var]
            variable_name = var_info.name.decode("utf-8")
            variable_category = FemzipVariableCategory.from_int(var_info.var_type)
            if variable_category == FemzipVariableCategory.NODE:

                variable_multiplier = 1
                if (
                    FemzipArrayType.NODE_DISPLACEMENT.value in variable_name
                    or FemzipArrayType.NODE_VELOCITIES.value in variable_name
                    or FemzipArrayType.NODE_ACCELERATIONS.value in variable_name
                ):
                    variable_multiplier = 3

                array_size = file_metadata.number_of_nodes * variable_multiplier
                buffer_size_state += array_size
                file_metadata.variable_infos[i_var].var_size = array_size

            elif variable_category == FemzipVariableCategory.SHELL:
                array_size = (
                    file_metadata.number_of_shell_elements
                    - file_metadata.number_of_rigid_shell_elements
                )
                file_metadata.variable_infos[i_var].var_size = array_size
                buffer_size_state += array_size
            elif variable_category == FemzipVariableCategory.SOLID:
                array_size = file_metadata.number_of_solid_elements
                file_metadata.variable_infos[i_var].var_size = array_size
                buffer_size_state += array_size
            elif variable_category == FemzipVariableCategory.BEAM:
                array_size = file_metadata.number_of_1D_elements
                file_metadata.variable_infos[i_var].var_size = array_size
                buffer_size_state += file_metadata.number_of_1D_elements
            elif variable_category == FemzipVariableCategory.THICK_SHELL:
                array_size = file_metadata.number_of_thick_shell_elements
                file_metadata.variable_infos[i_var].var_size = array_size
                buffer_size_state += file_metadata.number_of_thick_shell_elements
            elif variable_category == FemzipVariableCategory.GLOBAL:
                array_size = 6
                file_metadata.variable_infos[i_var].var_size = array_size
                buffer_size_state += array_size
            elif variable_category == FemzipVariableCategory.PART:
                logging.debug("n_parts: %d", n_parts)
                array_size = n_parts * 7 + n_rigid_walls * n_rigid_wall_vars
                file_metadata.variable_infos[i_var].var_size = array_size
                buffer_size_state += array_size
            elif variable_category == FemzipVariableCategory.CPM_FLOAT_VAR:
                array_size = n_airbag_particles
                file_metadata.variable_infos[i_var].var_size = array_size
                buffer_size_state += array_size
            elif variable_category == FemzipVariableCategory.CPM_INT_VAR:
                array_size = n_airbag_particles
                file_metadata.variable_infos[i_var].var_size = array_size
                buffer_size_state += array_size
            elif variable_category == FemzipVariableCategory.CPM_AIRBAG:
                array_size = n_airbags * 2
                file_metadata.variable_infos[i_var].var_size = array_size
                buffer_size_state += array_size
            else:
                warn_msg = "Femzip variable category '%s' is not supported"
                logging.warning(warn_msg, variable_category)
                var_indexes_to_remove.add(i_var)

        # one more for end marker
        buffer_size_state += 1

        return buffer_size_state

    def _decompose_read_variables_array(
        self,
        n_parts: int,
        n_rigid_walls: int,
        n_rigid_wall_vars: int,
        n_airbag_particles: int,
        n_airbags: int,
        all_vars_array: np.ndarray,
        n_timesteps_read: int,
        file_metadata: FemzipFileMetadata,
    ) -> Dict[Tuple[int, str, FemzipVariableCategory], np.ndarray]:

        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements

        # decompose array
        result_arrays: Dict[Tuple[int, str, FemzipVariableCategory], np.ndarray] = {}
        var_pos = 0
        for i_var in range(file_metadata.number_of_variables):

            var_info: VariableInfo = file_metadata.variable_infos[i_var]
            variable_name: str = var_info.name.decode("utf-8")
            variable_index: int = var_info.var_index
            variable_type = FemzipArrayType.from_string(variable_name)
            variable_category = FemzipVariableCategory.from_int(var_info.var_type)

            if variable_category == FemzipVariableCategory.NODE:
                if variable_type.value in (
                    FemzipArrayType.NODE_DISPLACEMENT.value,
                    FemzipArrayType.NODE_VELOCITIES.value,
                    FemzipArrayType.NODE_ACCELERATIONS.value,
                ):
                    array_size = file_metadata.number_of_nodes * 3
                    var_array = all_vars_array[:, var_pos : var_pos + array_size].reshape(
                        (n_timesteps_read, file_metadata.number_of_nodes, 3)
                    )
                    var_pos += array_size
                    result_arrays[
                        (variable_index, variable_name, FemzipVariableCategory.NODE)
                    ] = var_array
                else:
                    array_size = file_metadata.number_of_nodes
                    var_array = all_vars_array[:, var_pos : var_pos + array_size]
                    var_pos += array_size
                    result_arrays[
                        (variable_index, variable_name, FemzipVariableCategory.NODE)
                    ] = var_array

            elif variable_category == FemzipVariableCategory.SHELL:
                array_size = (
                    file_metadata.number_of_shell_elements
                    - file_metadata.number_of_rigid_shell_elements
                )
                var_array = all_vars_array[:, var_pos : var_pos + array_size]
                var_pos += array_size
                result_arrays[
                    (variable_index, variable_name, FemzipVariableCategory.SHELL)
                ] = var_array
            elif variable_category == FemzipVariableCategory.SOLID:
                array_size = file_metadata.number_of_solid_elements
                var_array = all_vars_array[:, var_pos : var_pos + array_size]
                var_pos += array_size
                result_arrays[
                    (variable_index, variable_name, FemzipVariableCategory.SOLID)
                ] = var_array
            elif variable_category == FemzipVariableCategory.BEAM:
                array_size = file_metadata.number_of_1D_elements
                var_array = all_vars_array[:, var_pos : var_pos + array_size]
                var_pos += array_size
                result_arrays[
                    variable_index, variable_name, FemzipVariableCategory.BEAM
                ] = var_array
            elif variable_category == FemzipVariableCategory.THICK_SHELL:
                array_size = file_metadata.number_of_thick_shell_elements
                var_array = all_vars_array[:, var_pos : var_pos + array_size]
                var_pos += array_size
                result_arrays[
                    variable_index, variable_name, FemzipVariableCategory.THICK_SHELL
                ] = var_array
            elif variable_category == FemzipVariableCategory.GLOBAL:
                array_size = 6
                var_array = all_vars_array[:, var_pos : var_pos + array_size]
                var_pos += array_size
                result_arrays[
                    variable_index, variable_name, FemzipVariableCategory.GLOBAL
                ] = var_array
            elif variable_category == FemzipVariableCategory.PART:
                array_size = n_parts * 7 + n_rigid_walls * n_rigid_wall_vars
                var_array = all_vars_array[:, var_pos : var_pos + array_size]
                var_pos += array_size
                result_arrays[
                    variable_index, variable_name, FemzipVariableCategory.PART
                ] = var_array
            elif variable_category == FemzipVariableCategory.CPM_FLOAT_VAR:
                array_size = n_airbag_particles
                var_array = all_vars_array[:, var_pos : var_pos + array_size]
                var_pos += array_size
                result_arrays[variable_index, variable_name, variable_category] = var_array
            elif variable_category == FemzipVariableCategory.CPM_INT_VAR:
                array_size = n_airbag_particles
                var_array = all_vars_array[:, var_pos : var_pos + array_size].view(np.int32)
                var_pos += array_size
                result_arrays[variable_index, variable_name, variable_category] = var_array
            elif variable_category == FemzipVariableCategory.CPM_AIRBAG:
                n_airbag_vars = 2
                array_size = n_airbags * n_airbag_vars
                var_array = all_vars_array[:, var_pos : var_pos + array_size]
                var_array = var_array.reshape((var_array.shape[0], n_airbags, n_airbag_vars))
                var_pos += array_size
                result_arrays[variable_index, variable_name, variable_category] = var_array
            else:
                err_msg = "Femzip variable category '{0}' is not supported"
                raise RuntimeError(err_msg)

        return result_arrays

    def read_variables(
        self,
        file_metadata: FemzipFileMetadata,
        n_parts: int,
        n_rigid_walls: int,
        n_rigid_wall_vars: int,
        n_airbag_particles: int,
        n_airbags: int,
        state_filter: Union[Set[int], None] = None,
    ) -> Dict[Tuple[int, str, FemzipVariableCategory], np.ndarray]:
        """Read specific variables from Femzip

        Parameters
        ----------
        file_metadata: FemzipFileMetadata
            metadata of file including which variables to read
        n_parts: int
            number of parts in the file
        n_rigid_walls: int
            number of rigid walls
        n_rigid_wall_vars: int
            number of rigid wall variables
        n_airbag_particles: int
            number of airbag particles in the file
        n_airbags: int
        state_filter: Union[Set[int], None]
            used to read specific arrays

        Returns
        -------
        arrays: dict
            dictionary with d3plot arrays

        Notes
        -----
            Uses extended femzip library and requires a license
            for 'FEMUNZIPLIB_DYNA'. Please contact sidact if
            required.
        """

        # pylint: disable = too-many-arguments
        # pylint: disable = too-many-locals

        # fetch metadata if required
        n_timesteps = file_metadata.number_of_timesteps
        logging.info("file_metadata: %s", self.struct_to_dict(file_metadata))

        # log variable names
        for i_var in range(file_metadata.number_of_variables):
            var_info = file_metadata.variable_infos[i_var]
            logging.debug("%s", self.struct_to_dict(var_info))

        # estimate float buffer size
        buffer_size_state = self._get_variables_state_buffer_size(
            n_parts=n_parts,
            n_rigid_walls=n_rigid_walls,
            n_rigid_wall_vars=n_rigid_wall_vars,
            n_airbag_particles=n_airbag_particles,
            n_airbags=n_airbags,
            file_metadata=file_metadata,
        )
        logging.info("buffer_size_state: %s", buffer_size_state)

        # specify which states to read
        states_to_copy = (
            {i_timestep for i_timestep in state_filter if i_timestep < n_timesteps + 1}
            if state_filter is not None
            else set(range(n_timesteps))
        )
        logging.info("states_to_copy: %s", states_to_copy)

        # take timesteps into account
        buffer_size = len(states_to_copy) * buffer_size_state
        logging.info("buffer_size: %s", buffer_size)

        # allocate memory
        # pylint: disable = invalid-name
        BufferType = c_float * buffer_size
        buffer = BufferType()

        # do the thing
        # pylint: disable = invalid-name
        BufferStateType = c_float * buffer_size_state
        n_timesteps_read = 0
        for i_timestep in range(n_timesteps):
            logging.info("timestep: %d", i_timestep)

            buffer_ptr_state = BufferStateType.from_buffer(
                buffer, sizeof(c_float) * n_timesteps_read * buffer_size_state
            )

            # read the variables into the buffer
            fortran_offset = 1
            err = self.api.read_variables(
                buffer_ptr_state,
                buffer_size_state,
                i_timestep + fortran_offset,
                byref(file_metadata),
            )
            self._check_femzip_error(err)

            # check if there is nothing to read anymore
            # thus we can terminate earlier
            if i_timestep in states_to_copy:
                states_to_copy.remove(i_timestep)
                n_timesteps_read += 1

            if not states_to_copy:
                logging.info("All states processed")
                break

        array = np.ctypeslib.as_array(buffer, shape=(buffer_size,)).reshape((n_timesteps_read, -1))

        # decompose total array into array pieces again
        result_arrays = self._decompose_read_variables_array(
            n_parts=n_parts,
            n_rigid_walls=n_rigid_walls,
            n_rigid_wall_vars=n_rigid_wall_vars,
            n_airbag_particles=n_airbag_particles,
            n_airbags=n_airbags,
            all_vars_array=array,
            n_timesteps_read=n_timesteps_read,
            file_metadata=file_metadata,
        )

        return result_arrays

    def _copy_variable_info_array(self, file_metadata: FemzipFileMetadata) -> FemzipFileMetadata:
        """Copies a variable info array into python memory

        Parameters
        ----------
        file_metadata: FemzipFileMetadata
            metadata object for femzip file

        Returns
        -------
        file_metadata2: FemzipFileMetadata
            very same data object but the data in
            variable_infos is now managed by python and
            not C anymore
        """
        file_metadata2 = FemzipFileMetadata()

        # allocate memory on python side
        data2 = (VariableInfo * file_metadata.number_of_variables)()

        # copy data
        for i_var in range(file_metadata.number_of_variables):
            var1 = file_metadata.variable_infos[i_var]
            var2 = data2[i_var]
            self.copy_struct(var1, var2)

        # assign
        self.copy_struct(file_metadata, file_metadata2)
        file_metadata2.variable_infos = data2
        return file_metadata2


class FemzipD3plotArrayMapping:
    """Contains information about how to map femzip arrays to d3plot arrays"""

    d3plot_array_type: str
    i_integration_point: Union[int, None]
    i_var_index: Union[int, None]

    fz_array_slices = Tuple[slice]

    def __init__(
        self,
        d3plot_array_type: str,
        fz_array_slices: Tuple[slice] = (slice(None),),
        i_integration_point: Union[int, None] = None,
        i_var_index: Union[int, None] = None,
    ):
        self.d3plot_array_type = d3plot_array_type
        self.fz_array_slices = fz_array_slices
        self.i_integration_point = i_integration_point
        self.i_var_index = i_var_index


class FemzipArrayMetadata:
    """Contains metadata about femzip arrays"""

    array_type: FemzipArrayType
    category: FemzipVariableCategory
    d3plot_mappings: List[FemzipD3plotArrayMapping]
    # set when parsed
    fz_var_index: Union[int, None] = None

    def __init__(
        self,
        array_type: FemzipArrayType,
        category: FemzipVariableCategory,
        d3plot_mappings: List[FemzipD3plotArrayMapping],
    ):
        self.array_type = array_type
        self.category = category
        self.d3plot_mappings = d3plot_mappings

    def match(self, fz_name: str) -> bool:
        """Checks if the given name matches the array

        Parameters
        ----------
        fz_name: str
            femzip array name

        Returns
        -------
        match: bool
            If the array metadata instance matches the given array
        """
        return self.array_type.value in fz_name

    def parse(self, fz_var_name: str, fz_var_index: int) -> None:
        """Parses the incoming femzip variable name and extracts infos

        Parameters
        ----------
        fz_var_name: str
            variable name from femzip
        fz_var_index: int
            variable index from femzip
        """
        # matches anything until brackets start
        pattern = re.compile(r"(^[^\(\n]+)(\([^\)]+\))*")

        matches = pattern.findall(fz_var_name)

        if not len(matches) == 1:
            err_msg = f"Could not match femzip array name: {fz_var_name}"
            raise RuntimeError(err_msg)
        if not len(matches[0]) == 2:
            err_msg = f"Could not match femzip array name: {fz_var_name}"
            raise RuntimeError(err_msg)

        # first group contains
        # - var name
        # - var index (if existing)
        # second group contains
        # - integration layer index
        (first_grp, second_grp) = matches[0]
        _, var_index = get_last_int_of_line(first_grp)

        # the slice 1:-1 leaves out the brackets '(' and ')'
        second_grp = second_grp[1:-1]
        if "inner" in second_grp:
            i_integration_point = 0
        elif "outer" in second_grp:
            i_integration_point = 1
        else:
            _, i_integration_point = get_last_int_of_line(second_grp)

        # setters
        self.fz_var_index = fz_var_index
        for mapping in self.d3plot_mappings:
            mapping.i_integration_point = i_integration_point
            mapping.i_var_index = var_index
