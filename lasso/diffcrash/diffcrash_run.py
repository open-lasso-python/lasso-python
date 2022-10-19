import argparse
import glob
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import typing
from concurrent import futures
from typing import List, Union

import psutil

from ..logging import str_error, str_info, str_running, str_success, str_warn

# pylint: disable = too-many-lines

DC_STAGE_SETUP = "SETUP"
DC_STAGE_IMPORT = "IMPORT"
DC_STAGE_MATH = "MATH"
DC_STAGE_EXPORT = "EXPORT"
DC_STAGE_MATRIX = "MATRIX"
DC_STAGE_EIGEN = "EIGEN"
DC_STAGE_MERGE = "MERGE"
DC_STAGES = [
    DC_STAGE_SETUP,
    DC_STAGE_IMPORT,
    DC_STAGE_MATH,
    DC_STAGE_EXPORT,
    DC_STAGE_MATRIX,
    DC_STAGE_EIGEN,
    DC_STAGE_MERGE,
]


def get_application_header():
    """Prints the header of the command line tool"""

    return """

       ==== D I F F C R A S H ====

       an open-lasso-python utility script
    """


def str2bool(value) -> bool:
    """Converts some value from the cmd line to a boolean

    Parameters
    ----------
    value: `str` or `bool`

    Returns
    -------
    bool_value: `bool`
        value as boolean
    """

    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_diffcrash_args():
    """Parse the arguments from the command line

    Returns
    -------
    args : `argparse.Namespace`
        parsed arguments
    """

    # print title
    print(get_application_header())

    parser = argparse.ArgumentParser(
        description="Python utility script for Diffcrash written by LASSO GmbH."
    )

    parser.add_argument(
        "--reference-run", type=str, required=True, help="filepath of the reference run."
    )
    parser.add_argument(
        "--exclude-runs", type=str, nargs="*", default=[], help="Runs to exclude from the analysis."
    )
    parser.add_argument(
        "--crash-code",
        type=str,
        required=True,
        help="Which crash code is used ('dyna', 'pam' or 'radioss').",
    )
    parser.add_argument(
        "--start-stage",
        type=str,
        nargs="?",
        default=DC_STAGES[0],
        help=f"At which specific stage to start the analysis ({', '.join(DC_STAGES)}).",
    )
    parser.add_argument(
        "--end-stage",
        type=str,
        nargs="?",
        default=DC_STAGES[-1],
        help=f"At which specific stage to stop the analysis ({', '.join(DC_STAGES)}).",
    )
    parser.add_argument(
        "--diffcrash-home",
        type=str,
        default=os.environ["DIFFCRASHHOME"] if "DIFFCRASHHOME" in os.environ else "",
        nargs="?",
        required=False,
        help=(
            "Home directory where Diffcrash is installed."
            " Uses environment variable 'DIFFCRASHHOME' if unspecified."
        ),
    )
    parser.add_argument(
        "--use-id-mapping",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to use id-based mapping (default is nearest neighbour).",
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        nargs="?",
        default="project",
        help="Project dir to use for femzip.",
    )
    parser.add_argument(
        "--config-file", type=str, nargs="?", default="", help="Path to the config file."
    )
    parser.add_argument(
        "--parameter-file", type=str, nargs="?", default="", help="Path to the parameter file."
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        nargs="?",
        default=max(1, psutil.cpu_count() - 1),
        help="Number of processes to use (default: max-1).",
    )
    parser.add_argument(
        "simulation_runs",
        type=str,
        nargs="*",
        help="Simulation runs or patterns used to search for simulation runs.",
    )

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    return parser.parse_args(sys.argv[1:])


def run_subprocess(args):
    """Run a subprocess with the specified arguments

    Parameters:
    -----------
        args : `list` of `str`

    Returns
    -------
    rc : `int`
        process return code

    Notes
    -----
        Suppresses stderr.
    """
    return subprocess.Popen(args, stderr=subprocess.DEVNULL).wait()


class DiffcrashRun:
    """Class for handling the settings of a diffcrash run"""

    # pylint: disable = too-many-instance-attributes

    # pylint: disable = too-many-arguments
    def __init__(
        self,
        project_dir: str,
        crash_code: str,
        reference_run: str,
        simulation_runs: typing.Sequence[str],
        exclude_runs: typing.Sequence[str],
        diffcrash_home: str = "",
        use_id_mapping: bool = False,
        config_file: str = None,
        parameter_file: str = None,
        n_processes: int = 1,
        logfile_dir: str = None,
    ):
        """Object handling a diffcrash run

        Parameters
        ----------
        project_dir : `str`
            directory to put all buffer files etc., in
        crash_code : `str`
            crash code to use.
        reference_run : `str`
            filepath to the reference run
        simulation_runs: `list` of `str`
            patterns used to search for simulation runs
        diffcrash_home : `str`
            home directory of diffcrash installation. Uses environment
            variable DIFFCRASHHOME if not set.
        use_id_mapping : `bool`
            whether to use id mapping instead of nearest neighbor mapping
        config_file : `str`
            filepath to a config file
        parameter_file : `str`
            filepath to the parameter file
        n_processes : `int`
            number of processes to spawn for worker pool
        logfile_dir : `str`
            directory to put logfiles in
        """

        # settings
        self._msg_option = "{:16s}: {}"
        self._log_formatter = logging.Formatter("%(levelname)s:%(asctime)s   %(message)s")

        # logdir
        if logfile_dir is not None:
            self.logfile_dir = logfile_dir
        else:
            self.logfile_dir = os.path.join(project_dir, "Log")
        self.logfile_filepath = os.path.join(self.logfile_dir, "DiffcrashRun.log")

        # logger
        self.logger = self._setup_logger()

        # make some space in the log
        self.logger.info(get_application_header())

        # diffcrash home
        self.diffcrash_home = self._parse_diffcrash_home(diffcrash_home)
        self.diffcrash_home = os.path.join(self.diffcrash_home, "bin")
        self.diffcrash_lib = os.path.join(os.path.dirname(self.diffcrash_home), "lib")

        if platform.system() == "Linux":
            os.environ["PATH"] = (
                os.environ["PATH"] + ":" + self.diffcrash_home + ":" + self.diffcrash_lib
            )
        if platform.system() == "Windows":
            os.environ["PATH"] = (
                os.environ["PATH"] + ";" + self.diffcrash_home + ";" + self.diffcrash_lib
            )

        # project dir
        self.project_dir = self._parse_project_dir(project_dir)

        # crashcode
        self.crash_code = self._parse_crash_code(crash_code)

        # reference run
        self.reference_run = self._parse_reference_run(reference_run)

        # mapping
        self.use_id_mapping = self._parse_use_id_mapping(use_id_mapping)

        # exlude runs
        self.exclude_runs = exclude_runs

        # simulation runs
        self.simulation_runs = self._parse_simulation_runs(
            simulation_runs, self.reference_run, self.exclude_runs
        )

        # config file
        self.config_file = self._parse_config_file(config_file)

        # parameter file
        self.parameter_file = self._parse_parameter_file(parameter_file)

        # n processes
        self.n_processes = self._parse_n_processes(n_processes)

    def _setup_logger(self) -> logging.Logger:

        # better safe than sorry
        os.makedirs(self.logfile_dir, exist_ok=True)

        # create console log channel
        # streamHandler = logging.StreamHandler(sys.stdout)
        # streamHandler.setLevel(logging.INFO)
        # streamHandler.setFormatter(self._log_formatter)

        # create file log channel
        file_handler = logging.FileHandler(self.logfile_filepath)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(self._log_formatter)

        # create logger
        logger = logging.getLogger("DiffcrashRun")
        logger.setLevel(logging.INFO)
        # logger.addHandler(streamHandler)
        logger.addHandler(file_handler)

        return logger

    def _parse_diffcrash_home(self, diffcrash_home) -> str:

        diffcrash_home_ok = len(diffcrash_home) != 0

        msg = self._msg_option.format("diffcrash-home", diffcrash_home)
        print(str_info(msg))
        self.logger.info(msg)

        if not diffcrash_home_ok:
            err_msg = (
                "Specify the path to the Diffcrash installation either "
                + "with the environment variable 'DIFFCRASHHOME' or the option --diffcrash-home."
            )
            self.logger.error(err_msg)
            raise RuntimeError(str_error(err_msg))

        return diffcrash_home

    def _parse_crash_code(self, crash_code) -> str:

        # these guys are allowed
        valid_crash_codes = ["dyna", "radioss", "pam"]

        # do the thing
        crash_code_ok = crash_code in valid_crash_codes

        print(str_info(self._msg_option.format("crash-code", crash_code)))
        self.logger.info(self._msg_option.format("crash-code", crash_code))

        if not crash_code_ok:
            err_msg = (
                f"Invalid crash code '{crash_code}'. "
                f"Please use one of: {str(valid_crash_codes)}"
            )
            self.logger.error(err_msg)
            raise RuntimeError(str_error(err_msg))

        return crash_code

    def _parse_reference_run(self, reference_run) -> str:

        reference_run_ok = os.path.isfile(reference_run)

        msg = self._msg_option.format("reference-run", reference_run)
        print(str_info(msg))
        self.logger.info(msg)

        if not reference_run_ok:
            err_msg = f"Filepath '{reference_run}' is not a file."
            self.logger.error(err_msg)
            raise RuntimeError(str_error(err_msg))

        return reference_run

    def _parse_use_id_mapping(self, use_id_mapping) -> bool:

        msg = self._msg_option.format("use-id-mapping", use_id_mapping)
        print(str_info(msg))
        self.logger.info(msg)

        return use_id_mapping

    def _parse_project_dir(self, project_dir):
        project_dir = os.path.abspath(project_dir)

        msg = self._msg_option.format("project-dir", project_dir)
        print(str_info(msg))
        self.logger.info(msg)

        return project_dir

    def _parse_simulation_runs(
        self,
        simulation_run_patterns: typing.Sequence[str],
        reference_run: str,
        exclude_runs: typing.Sequence[str],
    ):

        # search all denoted runs
        simulation_runs = []
        for pattern in simulation_run_patterns:
            simulation_runs += glob.glob(pattern)
        simulation_runs = [filepath for filepath in simulation_runs if os.path.isfile(filepath)]

        # search all excluded runs
        runs_to_exclude = []
        for pattern in exclude_runs:
            runs_to_exclude += glob.glob(pattern)
        runs_to_exclude = [filepath for filepath in runs_to_exclude if os.path.isfile(filepath)]

        n_runs_before_filtering = len(simulation_runs)
        simulation_runs = [
            filepath for filepath in simulation_runs if filepath not in runs_to_exclude
        ]
        n_runs_after_filtering = len(simulation_runs)

        # remove the reference run
        if reference_run in simulation_runs:
            simulation_runs.remove(reference_run)

        # sort it because we can!
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [atoi(c) for c in re.split(r"(\d+)", text)]

        simulation_runs = sorted(simulation_runs, key=natural_keys)

        # check
        simulation_runs_ok = len(simulation_runs) != 0

        msg = self._msg_option.format("# simul.-files", len(simulation_runs))
        print(str_info(msg))
        self.logger.info(msg)

        msg = self._msg_option.format(
            "# excluded files", (n_runs_before_filtering - n_runs_after_filtering)
        )
        print(str_info(msg))
        self.logger.info(msg)

        if not simulation_runs_ok:
            err_msg = (
                "No simulation files could be found with the specified patterns. "
                "Check the argument 'simulation_runs'."
            )
            self.logger.error(err_msg)
            raise RuntimeError(str_error(err_msg))

        return simulation_runs

    def _parse_config_file(self, config_file) -> Union[str, None]:

        _msg_config_file = ""
        if len(config_file) > 0 and not os.path.isfile(config_file):
            config_file = None
            _msg_config_file = f"Can not find config file '{config_file}'"

        # missing config file
        else:

            config_file = None
            _msg_config_file = (
                "Config file missing. "
                "Consider specifying the path with the option '--config-file'."
            )

        msg = self._msg_option.format("config-file", config_file)
        print(str_info(msg))
        self.logger.info(msg)

        if _msg_config_file:
            print(str_warn(_msg_config_file))
            self.logger.warning(_msg_config_file)

        return config_file

    def _parse_parameter_file(self, parameter_file) -> Union[None, str]:

        _msg_parameter_file = ""
        if len(parameter_file) > 0 and not os.path.isfile(parameter_file):
            parameter_file = None
            _msg_parameter_file = f"Can not find parameter file '{parameter_file}'"
        # missing parameter file
        else:
            parameter_file = None
            _msg_parameter_file = (
                "Parameter file missing. Consider specifying the "
                "path with the option '--parameter-file'."
            )

        msg = self._msg_option.format("parameter-file", parameter_file)
        print(str_info(msg))
        self.logger.info(msg)

        if _msg_parameter_file:
            print(str_warn(_msg_parameter_file))
            self.logger.warning(_msg_parameter_file)

        return parameter_file

    def _parse_n_processes(self, n_processes) -> int:

        print(str_info(self._msg_option.format("n-processes", n_processes)))

        if n_processes <= 0:
            err_msg = f"n-processes is '{n_processes}' but must be at least 1."
            self.logger.error(err_msg)
            raise ValueError(str_error(err_msg))

        return n_processes

    def create_project_dirs(self):
        """Creates all project relevant directores

        Notes
        -----
            Created dirs:
             - logfile_dir
             - project_dir
        """
        os.makedirs(self.project_dir, exist_ok=True)
        os.makedirs(self.logfile_dir, exist_ok=True)

    def run_setup(self, pool: futures.ThreadPoolExecutor):
        """Run diffcrash setup

        Parameters
        ----------
        pool : `concurrent.futures.ThreadPoolExecutor`
            multiprocessing pool
        """

        # SETUP
        msg = "Running Setup ... "
        print(str_running(msg) + "\r", end="", flush="")
        self.logger.info(msg)

        args = []
        if self.config_file is None and self.parameter_file is None:
            args = [
                os.path.join(self.diffcrash_home, "DFC_Setup_" + self.crash_code + "_fem"),
                self.reference_run,
                self.project_dir,
            ]
        elif self.config_file is not None and self.parameter_file is None:
            args = [
                os.path.join(self.diffcrash_home, "DFC_Setup_" + self.crash_code + "_fem"),
                self.reference_run,
                self.project_dir,
                "-C",
                self.config_file,
            ]
        elif self.config_file is None and self.parameter_file is not None:
            if ".fz" in self.reference_run:
                args = [
                    os.path.join(self.diffcrash_home, "DFC_Setup_" + self.crash_code + "_fem"),
                    self.reference_run,
                    self.project_dir,
                    "-P",
                    self.parameter_file,
                ]
            else:
                args = [
                    os.path.join(self.diffcrash_home, "DFC_Setup_" + self.crash_code),
                    self.reference_run,
                    self.project_dir,
                    "-P",
                    self.parameter_file,
                ]
        elif self.config_file is not None and self.parameter_file is not None:
            if ".fz" in self.reference_run:
                args = [
                    os.path.join(self.diffcrash_home, "DFC_Setup_" + self.crash_code + "_fem"),
                    self.reference_run,
                    self.project_dir,
                    "-C",
                    self.config_file,
                    "-P",
                    self.parameter_file,
                ]
            else:
                args = [
                    os.path.join(self.diffcrash_home, "DFC_Setup_" + self.crash_code),
                    self.reference_run,
                    self.project_dir,
                    "-C",
                    self.config_file,
                    "-P",
                    self.parameter_file,
                ]
        start_time = time.time()

        # submit task
        return_code_future = pool.submit(run_subprocess, args)
        return_code = return_code_future.result()

        # check return code
        if return_code != 0:
            err_msg = f"Running Setup ... done in {time.time() - start_time:.2f}s"
            print(str_error(err_msg))
            self.logger.error(err_msg)

            err_msg = "Process somehow failed."
            self.logger.error(err_msg)
            raise RuntimeError(str_error(err_msg))

        # check log
        messages = self.check_if_logfiles_show_success("DFC_Setup.log")
        if messages:
            err_msg = f"Running Setup ... done in {time.time() - start_time:.2f}s"
            print(str_error(err_msg))
            self.logger.error(err_msg)

            # print failed logs
            for msg in messages:
                print(str_error(msg))
                self.logger.error(msg)

            err_msg = "Setup failed."
            self.logger.error(err_msg)
            raise RuntimeError(str_error(err_msg))

        # print success
        err_msg = f"Running Setup ... done in {time.time() - start_time:.2f}s"
        print(str_success(msg))
        self.logger.info(msg)

    def run_import(self, pool: futures.ThreadPoolExecutor):
        """Run diffcrash import of runs

        Parameters
        ----------
        pool : `concurrent.futures.ThreadPoolExecutor`
            multiprocessing pool
        """

        # pylint: disable = too-many-locals, too-many-branches, too-many-statements

        # list of arguments to run in the command line
        import_arguments = []

        # id 1 is the reference run
        # id 2 and higher are the imported runs
        counter_offset = 2

        # assemble arguments for running the import
        # entry 0 is the reference run, thus we start at 1
        # pylint: disable = consider-using-enumerate
        for i_filepath in range(len(self.simulation_runs)):

            # parameter file missing
            if self.parameter_file is None:
                if self.use_id_mapping:
                    args = [
                        os.path.join(self.diffcrash_home, "DFC_Import_" + self.crash_code + "_fem"),
                        "-id",
                        self.simulation_runs[i_filepath],
                        self.project_dir,
                        str(i_filepath + counter_offset),
                    ]
                else:
                    args = [
                        os.path.join(self.diffcrash_home, "DFC_Import_" + self.crash_code + "_fem"),
                        self.simulation_runs[i_filepath],
                        self.project_dir,
                        str(i_filepath + counter_offset),
                    ]
            # indeed there is a parameter file
            else:
                if self.use_id_mapping:
                    args = [
                        os.path.join(self.diffcrash_home, "DFC_Import_" + self.crash_code),
                        "-ID",
                        self.simulation_runs[i_filepath],
                        self.project_dir,
                        str(i_filepath + counter_offset),
                    ]
                else:
                    args = [
                        os.path.join(self.diffcrash_home, "DFC_Import_" + self.crash_code),
                        self.simulation_runs[i_filepath],
                        self.project_dir,
                        str(i_filepath + counter_offset),
                    ]

            # append args to list
            import_arguments.append(args)

        # do the thing
        msg = "Running Imports ...\r"
        print(str_running(msg), end="", flush=True)
        self.logger.info(msg)
        start_time = time.time()
        return_code_futures = [pool.submit(run_subprocess, args) for args in import_arguments]

        # wait for imports to finish (with a progressbar)
        n_imports_finished = sum(
            return_code_future.done() for return_code_future in return_code_futures
        )
        while n_imports_finished != len(return_code_futures):

            # check again
            n_new_imports_finished = sum(
                return_code_future.done() for return_code_future in return_code_futures
            )

            # print
            percentage = n_new_imports_finished / len(return_code_futures) * 100

            if n_imports_finished != n_new_imports_finished:
                # pylint: disable = consider-using-f-string
                msg = "Running Imports ... [{0}/{1}] - {2:3.2f}%\r".format(
                    n_new_imports_finished, len(return_code_futures), percentage
                )
                print(str_running(msg), end="", flush=True)
                self.logger.info(msg)

            n_imports_finished = n_new_imports_finished

            # wait a little bit
            time.sleep(0.25)

        return_codes = [return_code_future.result() for return_code_future in return_code_futures]

        # print failure
        if any(return_code != 0 for return_code in return_codes):

            n_failed_runs = 0
            for i_run, return_code in enumerate(return_codes):
                if return_code != 0:
                    _err_msg = str_error(
                        f"Run {i_run} failed to import with error code '{return_code}'."
                    )
                    print(str_error(_err_msg))
                    self.logger.error(_err_msg)
                    n_failed_runs += 1

            err_msg = f"Running Imports ... done in {time.time() - start_time:.2f}s   "
            print(str_error(err_msg))
            self.logger.error(err_msg)

            err_msg = f"Import of {n_failed_runs} runs failed."
            self.logger.error(err_msg)
            raise RuntimeError(str_error(err_msg))

        # check log files
        messages = self.check_if_logfiles_show_success("DFC_Import_*.log")
        if messages:

            # print failure
            msg = f"Running Imports ... done in {time.time() - start_time:.2f}s   "
            print(str_error(msg))
            self.logger.info(msg)

            # print failed logs
            for msg in messages:
                self.logger.error(msg)
                print(str_error(msg))

            err_msg = (
                f"At least one import failed. Please check the log files in '{self.logfile_dir}'."
            )
            self.logger.error(err_msg)
            raise RuntimeError(str_error(err_msg))

        # print success
        print(str_success(f"Running Imports ... done in {time.time() - start_time:.2f}s   "))

    def run_math(self, pool: futures.ThreadPoolExecutor):
        """Run diffcrash math

        Parameters
        ----------
        pool : `concurrent.futures.ThreadPoolExecutor`
            multiprocessing pool
        """

        msg = "Running Math ... \r"
        print(str_running(msg), end="", flush=True)
        self.logger.info(msg)

        start_time = time.time()
        return_code_future = pool.submit(
            run_subprocess,
            [os.path.join(self.diffcrash_home, "DFC_Math_" + self.crash_code), self.project_dir],
        )
        return_code = return_code_future.result()

        # check return code
        if return_code != 0:

            msg = f"Running Math ... done in {time.time() - start_time:.2f}s   "
            print(str_error(msg))
            self.logger.error(msg)

            err_msg = f"Caught a nonzero return code '{return_code}'"
            self.logger.error(err_msg)
            raise RuntimeError(str_error(err_msg))

        # check logs
        messages = self.check_if_logfiles_show_success("DFC_MATH*.log")
        if messages:

            # print failure
            msg = f"Running Math ... done in {time.time() - start_time:.2f}s   "
            print(str_error(msg))
            self.logger.error(msg)

            # print failed logs
            for msg in messages:
                print(str_error(msg))
                self.logger.error(msg)

            err_msg = (
                "Logfile does indicate a failure. "
                f"Please check the log files in '{self.logfile_dir}'."
            )
            self.logger.error(err_msg)
            raise RuntimeError(str_error(err_msg))

        # print success
        msg = f"Running Math ... done in {time.time() - start_time:.2f}s   "
        print(str_success(msg))
        self.logger.info(msg)

    def run_export(self, pool: futures.ThreadPoolExecutor):
        """Run diffcrash export

        Parameters
        ----------
        pool : `concurrent.futures.ThreadPoolExecutor`
            multiprocessing pool
        """

        msg = "Running Export ... "
        print(str_running(msg) + "\r", end="", flush=True)
        self.logger.info(msg)

        if self.config_file is None:
            export_item_list = []

            # check for pdmx
            pdmx_filepath_list = glob.glob(os.path.join(self.project_dir, "*_pdmx"))
            if pdmx_filepath_list:
                export_item_list.append(os.path.basename(pdmx_filepath_list[0]))

            # check for pdij
            pdij_filepath_list = glob.glob(os.path.join(self.project_dir, "*_pdij"))
            if pdij_filepath_list:
                export_item_list.append(os.path.basename(pdij_filepath_list[0]))

        else:
            export_item_list = self.read_config_file(self.config_file)

        # remove previous existing exports
        for export_item in export_item_list:
            export_item_filepath = os.path.join(self.project_dir, export_item + ".d3plot.fz")
            if os.path.isfile(export_item_filepath):
                os.remove(export_item_filepath)

        # do the thing
        start_time = time.time()
        return_code_futures = [
            pool.submit(
                run_subprocess,
                [
                    os.path.join(self.diffcrash_home, "DFC_Export_" + self.crash_code),
                    self.project_dir,
                    export_item,
                ],
            )
            for export_item in export_item_list
        ]

        return_codes = [result_future.result() for result_future in return_code_futures]

        # check return code
        if any(rc != 0 for rc in return_codes):
            msg = f"Running Export ... done in {time.time() - start_time:.2f}s   "
            print(str_error(msg))
            self.logger.error(msg)

            for i_export, export_return_code in enumerate(return_codes):
                if export_return_code != 0:
                    msg = (
                        f"Return code of export '{export_item_list[i_export]}' "
                        f"was nonzero: '{export_return_code}'"
                    )
                    self.logger.error(msg)
                    print(str_error(msg))

            msg = "At least one export process failed."
            self.logger.error(msg)
            raise RuntimeError(str_error(msg))

        # check logs
        messages = self.check_if_logfiles_show_success("DFC_Export_*")
        if messages:

            # print failure
            msg = f"Running Export ... done in {time.time() - start_time:.2f}s   "
            print(str_error(msg))
            self.logger.error(msg)

            # print logs
            for msg in messages:
                print(str_error(msg))
                self.logger.error(msg)

            msg = (
                "At least one export failed. "
                f"Please check the log files in '{self.logfile_dir}'."
            )
            self.logger.error(msg)
            raise RuntimeError(str_error(msg))

        # print success
        msg = f"Running Export ... done in {time.time() - start_time:.2f}s   "
        print(str_success(msg))
        self.logger.info(msg)

    def run_matrix(self, pool: futures.ThreadPoolExecutor):
        """Run diffcrash matrix

        Parameters
        ----------
        pool : `concurrent.futures.ThreadPoolExecutor`
            multiprocessing pool
        """

        msg = "Running Matrix ... "
        print(str_running(msg) + "\r", end="", flush=True)
        self.logger.info(msg)

        start_time = time.time()

        # create the input file for the process
        matrix_inputfile = self._create_matrix_input_file(self.project_dir)

        # run the thing
        return_code_future = pool.submit(
            run_subprocess,
            [
                os.path.join(self.diffcrash_home, "DFC_Matrix_" + self.crash_code),
                self.project_dir,
                matrix_inputfile,
            ],
        )

        # please hold the line ...
        return_code = return_code_future.result()

        # check return code
        if return_code != 0:

            # print failure
            msg = f"Running Matrix ... done in {time.time() - start_time:.2f}s   "
            print(str_error(msg))
            self.logger.error(msg)

            msg = "The DFC_Matrix process failed somehow."
            self.logger.error(msg)
            raise RuntimeError(str_error(msg))

        # check log file
        messages = self.check_if_logfiles_show_success("DFC_Matrix_*")
        if messages:

            # print failure
            msg = f"Running Matrix ... done in {time.time() - start_time:.2f}s   "
            print(str_error(msg))
            self.logger.info(msg)

            # print why
            for msg in messages:
                print(str_error(msg))
                self.logger.error(msg)

            msg = f"DFC_Matrix failed. Please check the log files in '{self.logfile_dir}'."
            self.logger.error(msg)
            raise RuntimeError(str_error(msg))

        # print success
        msg = f"Running Matrix ... done in {time.time() - start_time:.2f}s   "
        print(str_success(msg))
        self.logger.info(msg)

    def run_eigen(self, pool: futures.ThreadPoolExecutor):
        """Run diffcrash eigen

        Parameters
        ----------
        pool : `concurrent.futures.ThreadPoolExecutor`
            multiprocessing pool
        """

        msg = "Running Eigen ... "
        print(str_running(msg) + "\r", end="", flush=True)
        self.logger.info(msg)

        # create input file for process
        eigen_inputfile = self._create_eigen_input_file(self.project_dir)

        # run the thing
        start_time = time.time()
        return_code_future = pool.submit(
            run_subprocess,
            [
                os.path.join(self.diffcrash_home, "DFC_Eigen_" + self.crash_code),
                self.project_dir,
                eigen_inputfile,
            ],
        )

        # please hold the line ...
        return_code = return_code_future.result()

        # check return code
        if return_code != 0:
            msg = f"Running Eigen ... done in {time.time() - start_time:.2f}s   "
            print(str_error(msg))
            self.logger.error(msg)

            msg = "The process failed somehow."
            self.logger.error(msg)
            raise RuntimeError(str_error(msg))

        # check log file
        messages = self.check_if_logfiles_show_success("DFC_Matrix_*")
        if messages:

            # print failure
            msg = f"Running Eigen ... done in {time.time() - start_time:.2f}s   "
            print(str_error(msg))
            self.logger.error(msg)

            # print why
            for msg in messages:
                print(str_error(msg))
                self.logger.error(msg)

            msg = f"DFC_Eigen failed. Please check the log files in '{self.logfile_dir}'."
            self.logger.error(msg)
            raise RuntimeError(str_error(msg))

        # print success
        msg = f"Running Eigen ... done in {time.time() - start_time:.2f}s   "
        print(str_success(msg))
        self.logger.info(msg)

    def run_merge(self, pool: futures.ThreadPoolExecutor):
        """Run diffcrash merge

        Parameters
        ----------
        pool : `concurrent.futures.ThreadPoolExecutor`
            multiprocessing pool
        """

        msg = "Running Merge ... "
        print(str_running(msg) + "\r", end="", flush=True)
        self.logger.info(msg)

        # create ionput file for merge
        merge_inputfile = self._create_merge_input_file(self.project_dir)

        # clear previous merges
        for filepath in glob.glob(os.path.join(self.project_dir, "mode_*")):
            if os.path.isfile(filepath):
                os.remove(filepath)

        # run the thing
        start_time = time.time()
        return_code_future = pool.submit(
            run_subprocess,
            [
                os.path.join(self.diffcrash_home, "DFC_Merge_All_" + self.crash_code),
                self.project_dir,
                merge_inputfile,
            ],
        )
        return_code = return_code_future.result()

        # check return code
        if return_code != 0:
            msg = f"Running Merge ... done in {time.time() - start_time:.2f}s   "
            print(str_error(msg))
            self.logger.info(msg)

            msg = "The process failed somehow."
            self.logger.error(msg)
            raise RuntimeError(str_error(msg))

        # check logfiles
        messages = self.check_if_logfiles_show_success("DFC_Merge_All.log")
        if messages:
            msg = f"Running Merge ... done in {time.time() - start_time:.2f}s   "
            print(str_error(msg))
            self.logger.error(msg)

            for msg in messages:
                print(str_error(msg))
                self.logger.info(msg)

            msg = "DFC_Merge_All failed. Please check the log files in '{self.logfile_dir}'."
            self.logger.error(msg)
            raise RuntimeError(str_error(msg))

        # print success
        msg = f"Running Merge ... done in {time.time() - start_time:.2f}s   "
        print(str_success(msg))
        self.logger.info(msg)

    def is_logfile_successful(self, logfile: str) -> bool:
        """Checks if a logfile indicates a success

        Parameters
        ----------
        logfile : `str`
            filepath to the logfile

        Returns
        -------
        success : `bool`
        """

        with open(logfile, "r", encoding="utf-8") as fp:
            for line in fp:
                if "successfully" in line:
                    return True
        return False

    def _create_merge_input_file(self, directory: str) -> str:
        """Create an input file for the merge executable

        Notes
        -----
            From the official diffcrash docs.
        """

        # creates default inputfile for DFC_Merge
        filepath = os.path.join(directory, "merge_all.txt")
        with open(filepath, "w", encoding="utf-8") as merge_input_file:
            merge_input_file.write("eigen_all        ! Name of eigen input file\n")
            merge_input_file.write(
                "mode_            ! Name of Output file "
                + "(string will be apended with mode information)\n"
            )
            merge_input_file.write("1 1              ! Mode number to be generated\n")
            merge_input_file.write("'d+ d-'          ! Mode type to be generated\n")
            # TIMESTEPSFILE         optional
            merge_input_file.write(
                "                 ! Optional: Timestepfile (specify timesteps used for merge)\n"
            )
            # PARTSFILE             optional
            merge_input_file.write(
                "                 ! Optional: Partlistfile (specify parts used for merge)\n"
            )

        return filepath

    def _create_eigen_input_file(self, directory: str) -> str:
        """Create an input file for the eigen executable

        Notes
        -----
            From the official diffcrash docs.
        """

        # creates default inputfile for DFC_Eigen
        filepath = os.path.join(directory, "eigen_all.txt")
        with open(filepath, "w", encoding="utf-8") as eigen_input_file:
            eigen_input_file.write("matrix_all\n")
            eigen_input_file.write('""\n')
            eigen_input_file.write("1 1000\n")
            eigen_input_file.write('""\n')
            eigen_input_file.write("0 0\n")
            eigen_input_file.write('""\n')
            eigen_input_file.write("eigen_all\n")
            eigen_input_file.write('""\n')
            eigen_input_file.write("0 0\n")

        return filepath

    def _create_matrix_input_file(self, directory: str) -> str:
        """Create an input file for the matrix executable

        Notes
        -----
            From the official diffcrash docs.
        """

        # creates default inputfile for DFC_Matrix
        filepath = os.path.join(directory, "matrix.txt")
        with open(filepath, "w", encoding="utf-8") as matrix_input_file:
            matrix_input_file.write("0 1000        !    Initial and final time stept to consider\n")
            matrix_input_file.write('""          !    not used\n')
            matrix_input_file.write('""          !    not used\n')
            matrix_input_file.write("matrix_all    !    Name of matrix file set (Output)\n")

        return filepath

    def clear_project_dir(self):
        """Clears the entire project dir"""

        # disable logging
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)

        # delete folder
        if os.path.exists(self.project_dir):
            shutil.rmtree(self.project_dir)

        # reinit logger
        self.logger = self._setup_logger()

    def read_config_file(self, config_file: str) -> List[str]:
        """Read a diffcrash config file

        Parameters
        ----------
        config_file : `str`
            path to the config file

        Notes
        -----
            From the official diffcrash docs ... seriously.
        """

        # Just to make it clear, this is not code from LASSO
        # ...

        # pylint: disable = too-many-locals
        # pylint: disable = consider-using-enumerate
        # pylint: disable = too-many-nested-blocks
        # pylint: disable = too-many-branches
        # pylint: disable = too-many-statements

        with open(config_file, "r", encoding="utf-8") as conf:
            conf_lines = conf.readlines()
        line = 0

        for i in range(0, len(conf_lines)):
            if conf_lines[i].find("FUNCTION") >= 0:
                line = i + 1
                break

        export_item_list = []
        j = 1
        if line != 0:
            while 1:
                while 1:
                    for i in range(0, len(conf_lines[line])):
                        if conf_lines[line][i] == "<":
                            element_start = i + 1
                        if conf_lines[line][i] == ">":
                            element_end = i
                    elem = conf_lines[line][element_start:element_end]
                    check = conf_lines[line + j][:-1]

                    if check.find(elem) >= 0:
                        line = line + j + 1
                        j = 1
                        break
                    j += 1
                    items = check.split(" ")
                    pos = -1
                    for n in range(0, len(items)):
                        if items[n].startswith("!"):
                            msg = f"FOUND at {n}"
                            print(msg)
                            self.logger.info(msg)
                            pos = n
                            break
                        pos = len(items)

                    for n in range(0, pos):
                        if items[n] == "PDMX" or items[n] == "pdmx":
                            break
                        if items[n] == "PDXMX" or items[n] == "pdxmx":
                            break
                        if items[n] == "PDYMX" or items[n] == "pdymx":
                            break
                        if items[n] == "PDZMX" or items[n] == "pdzmx":
                            break
                        if items[n] == "PDIJ" or items[n] == "pdij":
                            break
                        if items[n] == "STDDEV" or items[n] == "stddev":
                            break
                        if items[n] == "NCOUNT" or items[n] == "ncount":
                            break
                        if items[n] == "MISES_MX" or items[n] == "mises_mx":
                            break
                        if items[n] == "MISES_IJ" or items[n] == "mises_ij":
                            break

                    for k in range(n, pos):
                        postval = None
                        for m in range(0, n):
                            if items[m] == "coordinates":
                                items[m] = "geometry"
                            if postval is None:
                                postval = items[m]
                            else:
                                postval = postval + "_" + items[m]
                        postval = postval.strip("_")

                        # hotfix
                        # sometimes the engine writes 'Geometry' instead of 'geometry'
                        postval = postval.lower()

                        items[k] = items[k].strip()

                        if items[k] != "" and items[k] != "\r":
                            if postval.lower() == "sigma":
                                export_item_list.append(
                                    elem + "_" + postval + "_" + "001_" + items[k].lower()
                                )
                                export_item_list.append(
                                    elem + "_" + postval + "_" + "002_" + items[k].lower()
                                )
                                export_item_list.append(
                                    elem + "_" + postval + "_" + "003_" + items[k].lower()
                                )
                            else:
                                export_item_list.append(
                                    elem + "_" + postval + "_" + items[k].lower()
                                )
                        if export_item_list[-1].endswith("\r"):
                            export_item_list[-1] = export_item_list[-1][:-1]

                if conf_lines[line].find("FUNCTION") >= 0:
                    break
        else:
            export_item_list = ["NODE_geometry_pdmx", "NODE_geometry_pdij"]

        return export_item_list

    def check_if_logfiles_show_success(self, pattern: str) -> List[str]:
        """Check if a logfiles with given pattern show success

        Parameters
        ----------
        pattern : `str`
            file pattern used to search for logfiles

        Returns
        -------
        messages : `list`
            list with messages of failed log checks
        """

        _msg_logfile_nok = str_error("Logfile '{0}' reports no success.")
        messages = []

        logfiles = glob.glob(os.path.join(self.logfile_dir, pattern))
        for filepath in logfiles:
            if not self.is_logfile_successful(filepath):
                messages.append(_msg_logfile_nok.format(filepath))

        return messages
