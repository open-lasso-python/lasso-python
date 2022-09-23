
import contextlib
import glob
import os
import typing
from typing import Iterator, List, Union


@contextlib.contextmanager
def open_file_or_filepath(path_or_file: Union[str, typing.BinaryIO],
                          mode: str) -> Iterator[typing.BinaryIO]:
    """ This function accepts a file or filepath and handles closing correctly

    Parameters
    ----------
    path_or_file: Union[str, typing.IO]
        path or file
    mode: str
        filemode

    Yields
    ------
    f: file object
    """
    if isinstance(path_or_file, str):
        f = file_to_close = open(path_or_file, mode)
    else:
        f = path_or_file
        file_to_close = None
    try:
        yield f
    finally:
        if file_to_close:
            file_to_close.close()


def collect_files(dirpath: Union[str, List[str]],
                  patterns: Union[str, List[str]],
                  recursive: bool = False):
    ''' Collect files from directories

    Parameters
    ----------
    dirpath: Union[str, List[str]]
        path to one or multiple directories to search through
    patterns: Union[str, List[str]]
        patterns to search for
    recursive: bool
        whether to also search subdirs

    Returns
    -------
    found_files: Union[List[str], List[List[str]]]
        returns the list of files found for every pattern specified

    Examples
    --------
        >>> png_images, jpeg_images = collect_files('./folder', ['*.png', '*.jpeg'])
    '''

    if not isinstance(dirpath, (list, tuple)):
        dirpath = [dirpath]
    if not isinstance(patterns, (list, tuple)):
        patterns = [patterns]

    found_files = []
    for pattern in patterns:

        files_with_pattern = []
        for current_dir in dirpath:
            # files in root dir
            files_with_pattern += glob.glob(
                os.path.join(current_dir, pattern))
            # subfolders
            if recursive:
                files_with_pattern += glob.glob(
                    os.path.join(current_dir, '**', pattern))

        found_files.append(sorted(files_with_pattern))

    if len(found_files) == 1:
        return found_files[0]
    else:
        return found_files
