
import mmap
import os
from typing import Any, List, Union

import numpy as np


class BinaryBuffer:
    '''This class is used to handle binary data
    '''

    def __init__(self, filepath: Union[str, None] = None, n_bytes: int = 0):
        '''Buffer used to read binary files

        Parameters
        ----------
        filepath: Union[str, None]
            path to a binary file
        n_bytes: int
            how many bytes to load (uses memory mapping)

        Returns
        -------
        instance: BinaryBuffer
        '''
        self.filepath_ = None
        self.sizes_ = []
        self.load(filepath, n_bytes)

    @property
    def memoryview(self) -> memoryview:
        '''Get the underlying memoryview of the binary buffer

        Returns
        -------
        mv_: memoryview
            memoryview used to store the data
        '''
        return self.mv_

    @memoryview.setter
    def memoryview(self, new_mv):
        '''Set the memoryview of the binary buffer manually

        Parameters
        ----------
        new_mv: memoryview
            memoryview used to store the bytes
        '''
        assert(isinstance(new_mv, memoryview))
        self.mv_ = new_mv
        self.sizes_ = [len(self.mv_)]

    def get_slice(self,
                  start: int,
                  end=Union[None, int],
                  step: int = 1) -> 'BinaryBuffer':
        '''Get a slice of the binary buffer

        Parameters
        ----------
        start: int
            start position in bytes
        end: Union[int, None]
            end position
        step: int
            step for slicing (default 1)

        Returns
        -------
        new_buffer: BinaryBuffer
            the slice as a new buffer
        '''

        assert(start < len(self))
        assert(end is None or end < len(self))

        end = len(self) if end is None else end

        new_binary_buffer = BinaryBuffer()
        new_binary_buffer.memoryview = self.mv_[start:end:step]

        return new_binary_buffer

    def __len__(self) -> int:
        '''Get the length of the byte buffer

        Returns
        -------
        len: int
        '''
        return len(self.mv_)

    @property
    def size(self) -> int:
        '''Get the size of the byte buffer

        Returns
        -------
        size: int
            size of buffer in bytes
        '''
        return len(self.mv_)

    @size.setter
    def size(self, size: int):
        '''Set the length of the byte buffer

        Parameters
        ----------
        size: int
            new size of the buffer
        '''

        if len(self.mv_) > size:
            self.mv_ = self.mv_[:size]
        elif len(self.mv_) < size:
            buffer = bytearray(self.mv_) + bytearray(b'0' * (size - len(self.mv_)))
            self.mv_ = memoryview(buffer)

    def read_number(self, start: int, dtype: np.dtype) -> Union[float, int]:
        '''Read a number from the buffer

        Parameters
        ----------
        start: int
            at which byte to start reading
        dtype: np.dtype
            type of the number to read

        Returns
        -------
        number: np.dtype
            number with the type specified
        '''
        return np.frombuffer(self.mv_,
                             dtype=dtype,
                             count=1,
                             offset=start)[0]

    def write_number(self, start: int, value: Any, dtype: np.dtype):
        '''Write a number to the buffer

        Parameters
        ----------
        start: int
            at which byte to start writing
        value: Any
            value to write
        dtype: np.dtype
            type of the number to write
        '''

        wrapper = np.frombuffer(self.mv_[start:], dtype=dtype)
        wrapper[0] = value

    def read_ndarray(self, start: int, length: int, step: int, dtype: np.dtype) -> np.ndarray:
        '''Read a numpy array from the buffer

        Parameters
        ----------
        start: int
            at which byte to start reading
        len: int
            length in bytes to read
        step: int
            byte step size (how many bytes to skip)
        dtype: np.dtype
            type of the number to read

        Returns
        -------
        array: np.andrray
        '''

        return np.frombuffer(self.mv_[start:start + length:step],
                             dtype=dtype)

    def write_ndarray(self, array: np.ndarray, start: int, step: int):
        '''Write a numpy array to the buffer

        Parameters
        ----------
        array: np.ndarray
            array to save to the file
        start: int
            start in bytes
        step: int
            byte step size (how many bytes to skip)
        '''

        wrapper = np.frombuffer(self.mv_[start::step],
                                dtype=array.dtype)

        np.copyto(wrapper[:array.size], array, casting='no')

    def read_text(self, start: int, length: int, step: int = 1, encoding: str = 'utf8') -> str:
        '''Read text from the binary buffer

        Parameters
        ----------
        start: int
            start in bytes
        length: int
            length in bytes to read
        step: int
            byte step size
        encoding: str
            encoding used
        '''
        return self.mv_[start:start + length:step].tobytes().decode(encoding)

    def save(self, filepath: Union[str, None] = None):
        '''Save the binary buffer to a file

        Parameters
        ----------
        filepath: str
            path where to save the data

        Notes
        -----
            Overwrites to original file if no filepath
            is specified.
        '''

        filepath_parsed = filepath if filepath else (self.filepath_[0] if self.filepath_ else None)

        if filepath_parsed is None:
            return

        with open(filepath_parsed, "wb") as fp:
            fp.write(self.mv_)

        self.filepath_ = filepath_parsed

    def load(self, filepath: Union[List[str], str, None] = None, n_bytes: int = 0):
        '''load a file

        Parameters
        ----------
        filepath: Union[str, None]
            path to the file to load
        n_bytes: int
            number of bytes to load (uses memory mapping if nonzero)

        Notes
        -----
            If not filepath is specified, then the opened file is simply
            reloaded.
        '''

        filepath = filepath if filepath else self.filepath_

        if not filepath:
            return

        # convert to a list if only a single file is given
        filepath_parsed = [filepath] if isinstance(filepath, str) else filepath

        # get size of all files
        sizes = [os.path.getsize(path) for path in filepath_parsed]

        # reduce memory if required
        sizes = [entry if n_bytes == 0 else min(n_bytes, entry) for entry in sizes]

        memorysize = sum(sizes)

        # allocate memory
        buffer = memoryview(bytearray(b'0' * memorysize))

        # read files and concatenate them
        sizes_tmp = [0] + sizes
        for i_path, path in enumerate(filepath_parsed):
            with open(path, "br") as fp:
                if n_bytes:
                    mm = mmap.mmap(fp.fileno(),
                                   sizes[i_path],
                                   access=mmap.ACCESS_READ)
                    buffer[sizes_tmp[i_path]:] = mm[:sizes[i_path]]
                else:
                    fp.readinto(buffer[sizes_tmp[i_path]:])

        self.filepath_ = filepath_parsed
        self.sizes_ = sizes
        self.mv_ = buffer

    def append(self, binary_buffer: 'BinaryBuffer'):
        '''Append another binary buffer to this one

        Parameters
        ----------
        binary_buffer: BinaryBuffer
            buffer to append
        '''

        assert(isinstance(binary_buffer, BinaryBuffer))

        self.mv_ = memoryview(bytearray(self.mv_) +
                              bytearray(binary_buffer.mv_))
        self.sizes_.append(len(binary_buffer))
