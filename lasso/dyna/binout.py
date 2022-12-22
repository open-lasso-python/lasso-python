import glob
from typing import List, Union

import h5py
import numpy as np
import pandas as pd

from .lsda_py3 import Lsda


class Binout:
    """This class is meant to read binouts from LS-Dyna

    Parameters
    ----------
    filepath: str
        Path to the binout to read. May contain * (glob) for selecting multiple
        files.

    Attributes
    ----------
        filelist: List[str]
            List of files which are opened.
        lsda: Lsda
            The underlying LS-Dyna binout reader instance from code from LSTC.
        lsda_root: Symbol
            Root lsda symbol which is like a root directory to traverse the
            content of the binout file.

    Notes
    -----
        This class is only a utility wrapper for Lsda from LSTC.

    Examples
    --------
        >>> binout = Binout("path/to/binout")
    """

    def __init__(self, filepath: str):
        """Constructor for a binout

        Parameters
        ----------
        filepath: str
            path to the binout or pattern

        Notes
        -----
            The class loads the file given in the filepath. By giving a
            search pattern such as: "binout*", all files with that
            pattern will be loaded.

        Examples
        --------
            >>> # reads a single binout
            >>> binout = Binout("path/to/binout0000")
            >>> binout.filelist
            ['path/to/binout0000']

            >>> # reads multiple files
            >>> binout = Binout("path/to/binout*")
            >>> binout.filelist
            ['path/to/binout0000','path/to/binout0001']
        """

        self.filelist = glob.glob(filepath)

        # check file existance
        if not self.filelist:
            raise IOError("No file was found.")

        # open lsda buffer
        self.lsda = Lsda(self.filelist, "r")
        self.lsda_root = self.lsda.root

    def read(self, *path) -> Union[List[str], str, np.ndarray]:
        """Read all data from Binout (top to low level)

        Parameters
        ----------
        path: Union[Tuple[str, ...], List[str], str]
            internal path in the folder structure of the binout

        Returns
        -------
        ret: Union[List[str], str, np.ndarray]
            list of subdata within the folder or data itself (array or string)

        Notes
        -----
            This function is used to read any data from the binout. It has been used
            to make the access to the data more comfortable. The return type depends
            on the given path:

             - `binout.read()`: `List[str] names of directories (in binout)
             - `binout.read(dir)`: `List[str]` names of variables or subdirs
             - `binout.read(dir1, ..., variable)`: np.array data

            If you have multiple outputs with different ids (e.g. in nodout for
            multiple nodes) then don't forget to read the id array for
            identification or id-labels.

        Examples
        --------
            >>> from lasso.dyna import Binout
            >>> binout = Binout("test/binout")
            >>> # get top dirs
            >>> binout.read()
            ['swforc']
            >>> binout.read("swforc")
            ['title', 'failure', 'ids', 'failure_time', ...]
            >>> binout.read("swforc","shear").shape
            (321L, 26L)
            >>> binout.read("swforc","ids").shape
            (26L,)
            >>> binout.read("swforc","ids")
            array([52890, 52891, 52892, ...])
            >>> # read a string value
            >>> binout.read("swforc","date")
            '11/05/2013'
        """

        return self._decode_path(path)

    def as_df(self, *args) -> pd.DataFrame:
        """read data and convert to pandas dataframe if possible

        Parameters
        ----------
        *args: Union[Tuple[str, ...], List[str], str]
            internal path in the folder structure of the binout

        Returns
        -------
        df: pandas.DataFrame
            data converted to pandas dataframe

        Raises
        ------
        ValueError
            if the data cannot be converted to a pandas dataframe

        Examples
        --------
            >>> from lasso.dyna import Binout
            >>> binout = Binout('path/to/binout')

            Read a time-dependent array.

            >>> binout.as_df('glstat', 'eroded_kinetic_energy')
            time
            0.00000        0.000000
            0.19971        0.000000
            0.39942        0.000000
            0.59976        0.000000
            0.79947        0.000000
                            ...
            119.19978    105.220786
            119.39949    105.220786
            119.59983    105.220786
            119.79954    105.220786
            119.99988    105.220786
            Name: eroded_kinetic_energy, Length: 601, dtype: float64

            Read a time and id-dependent array.

            >>> binout.as_df('secforc', 'x_force')
                                  1             2             3  ...            33            34
            time                                                 .
            0.00063    2.168547e-16  2.275245e-15 -3.118639e-14  ... -5.126108e-13  4.592941e-16
            0.20034    3.514243e-04  3.797908e-04 -1.701294e-03  ...  2.530416e-11  2.755493e-07
            0.40005    3.052490e-03  3.242951e-02 -2.699926e-02  ...  6.755315e-06 -2.608923e-03
            0.60039   -1.299816e-02  4.930999e-02 -1.632376e-02  ...  8.941705e-05 -2.203455e-02
            0.80010    1.178485e-02  4.904512e-02 -9.740204e-03  ...  5.648263e-05 -6.999854e-02
            ...                 ...           ...           ...  ...           ...           ...
            119.00007  9.737679e-01 -8.833702e+00  1.298964e+01  ... -9.977377e-02  7.883521e+00
            119.20041  7.421170e-01 -8.849411e+00  1.253505e+01  ... -1.845916e-01  7.791409e+00
            119.40012  9.946615e-01 -8.541475e+00  1.188757e+01  ... -3.662228e-02  7.675800e+00
            119.60046  9.677638e-01 -8.566695e+00  1.130774e+01  ...  5.144208e-02  7.273052e+00
            119.80017  1.035165e+00 -8.040828e+00  1.124044e+01  ... -1.213450e-02  7.188395e+00
        """

        data = self.read(*args)

        # validate time-based data
        if not isinstance(data, np.ndarray):
            err_msg = "data is not a numpy array but has type '{0}'"
            raise ValueError(err_msg.format(type(data)))

        time_array = self.read(*args[:-1], "time")
        if data.shape[0] != time_array.shape[0]:
            raise ValueError("data series length does not match time array length")

        time_pdi = pd.Index(time_array, name="time")

        # create dataframe
        if data.ndim > 1:
            df = pd.DataFrame(index=time_pdi)

            if args[0] == "rcforc":
                ids = [
                    (str(i) + "m") if j else (str(i) + "s")
                    for i, j in zip(self.read("rcforc", "ids"), self.read("rcforc", "side"))
                ]
            else:
                ids = self.read(*args[:-1], "ids")

            for i, j in enumerate(ids):
                df[str(j)] = data.T[i]

        else:
            df = pd.Series(data, index=time_pdi, name=args[-1])

        return df

    def _decode_path(self, path):
        """Decode a path and get whatever is inside.

        Parameters
        ----------
        path: List[str]
            path within the binout

        Notes
        -----
            Usually returns the folder children. If there are variables in the folder
            (usually also if a subfolder metadata exists), then the variables will
            be printed from these directories.

        Returns
        -------
        ret: Union[List[str], np.ndarray]
            either sub folder list or data array
        """

        i_level = len(path)

        if i_level == 0:  # root subfolders
            return self._bstr_to_str(list(self.lsda_root.children.keys()))

        # some subdir
        # try if path can be resolved (then it's a dir)
        # in this case print the subfolders or subvars
        try:
            dir_symbol = self._get_symbol(self.lsda_root, path)

            if "metadata" in dir_symbol.children:
                return self._collect_variables(dir_symbol)
            return self._bstr_to_str(list(dir_symbol.children.keys()))

        # an error is risen, if the path is not resolvable
        # this could be, because we want to read a var
        except ValueError:
            return self._get_variable(path)

    def _get_symbol(self, symbol, path):
        """Get a symbol from a path via lsda

        Parameters
        ----------
        symbol: Symbol
            current directory which is a Lsda.Symbol

        Returns
        -------
        symbol: Symbol
            final symbol after recursive search of path
        """

        # check
        if symbol is None:
            raise ValueError("Symbol may not be none.")

        # no further path, return current symbol
        if len(path) == 0:
            return symbol

        # more subsymbols to search for
        sub_path = list(path)  # copy
        next_symbol_name = sub_path.pop(0)

        next_symbol = symbol.get(next_symbol_name)
        if next_symbol is None:
            raise ValueError(f"Cannot find: {next_symbol_name}")

        return self._get_symbol(next_symbol, sub_path)

    def _get_variable(self, path):
        """Read a variable from a given path

        Parameters
        ----------
        path: List[str]
            path to the variable

        Returns
        -------
        data: np.ndarray
        """

        dir_symbol = self._get_symbol(self.lsda_root, path[:-1])
        # variables are somehow binary strings ... dirs not
        variable_name = self._str_to_bstr(path[-1])

        # var in metadata
        if ("metadata" in dir_symbol.children) and (
            variable_name in dir_symbol.get("metadata").children
        ):
            var_symbol = dir_symbol.get("metadata").get(variable_name)
            var_type = var_symbol.type

            # symbol is a string
            if var_type == 1:
                return self._to_string(var_symbol.read())

            # symbol is numeric data
            return np.asarray(var_symbol.read())

        # var in state data ... hopefully
        time = []
        data = []
        for subdir_name, subdir_symbol in dir_symbol.children.items():

            # skip metadata
            if subdir_name == "metadata":
                continue

            # read data
            if variable_name in subdir_symbol.children:
                state_data = subdir_symbol.get(variable_name).read()
                if len(state_data) == 1:
                    data.append(state_data[0])
                else:  # more than one data entry
                    data.append(state_data)

                time_symbol = subdir_symbol.get(b"time")
                if time_symbol:
                    time += time_symbol.read()

        # return sorted by time
        if len(time) == len(data):
            return np.array(data)[np.argsort(time)]

        return np.array(data)

    def _collect_variables(self, symbol):
        """Collect all variables from a symbol

        Parameters
        ----------
        symbol: Symbol

        Returns
        -------
        variable_names: List[str]

        Notes
        -----
            This function collect all variables from the state dirs and metadata.
        """

        var_names = set()
        for _, subdir_symbol in symbol.children.items():
            var_names = var_names.union(subdir_symbol.children.keys())

        return self._bstr_to_str(list(var_names))

    def _to_string(self, data_array):
        """Convert a data series of numbers (usually ints) to a string

        Parameters
        ----------
        data_array: Union[int, np.ndarray]
            some data array

        Returns
        -------
        string: str
            data array converted to characters

        Notes
        -----
            This is needed for the reason that sometimes the binary data
            within the files are strings.
        """

        return "".join([chr(entry) for entry in data_array])

    def _bstr_to_str(self, arg):
        """Encodes or decodes a string correctly regarding python version

        Parameters
        ----------
        arg: Union[str, bytes]

        Returns
        -------
        string: str
            converted to python version
        """

        # in case of a list call this function with its atomic strings
        if isinstance(arg, (list, tuple)):
            return [self._bstr_to_str(entry) for entry in arg]

        # convert a string (dependent on python version)
        if not isinstance(arg, str):
            return arg.decode("utf-8")

        return arg

    def _str_to_bstr(self, string):
        """Convert a string to a binary string python version independent

        Parameters
        ----------
        string: str

        Returns
        -------
        string: bytes
        """

        if not isinstance(string, bytes):
            return string.encode("utf-8")

        return string

    def save_hdf5(self, filepath, compression="gzip"):
        """Save a binout as HDF5

        Parameters
        ----------
        filepath: str
            path where the HDF5 shall be saved
        compression: str
            compression technique (see h5py docs)

        Examples
        --------
            >>> binout = Binout("path/to/binout")
            >>> binout.save_hdf5("path/to/binout.h5")
        """

        with h5py.File(filepath, "w") as fh:
            self._save_all_variables(fh, compression)

    def _save_all_variables(self, hdf5_grp, compression, *path):
        """Iterates through all variables in the Binout

        Parameters
        ----------
        hdf5_grp: Group
            group object in the HDF5, where all the data
            shall be saved into (of course in a tree like
            manner)
        compression: str
            compression technique (see h5py docs)
        path: Tuple[str, ...]
            entry path in the binout
        """

        ret = self.read(*path)
        path_str = "/".join(path)

        # iterate through subdirs
        if isinstance(ret, list):

            if path_str:
                hdf5_grp = hdf5_grp.create_group(path_str)

            for entry in ret:
                path_child = path + (entry,)
                self._save_all_variables(hdf5_grp, compression, *path_child)
        # children are variables
        else:
            # can not save strings, only list of strings ...
            if isinstance(ret, str):
                ret = np.array([ret], dtype=np.dtype("S"))
            hdf5_grp.create_dataset(path[-1], data=ret, compression=compression)
