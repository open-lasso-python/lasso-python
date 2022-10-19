import argparse
import enum
import glob
import os
import re
import shutil
import sys
import time
from concurrent.futures.process import ProcessPoolExecutor
from typing import Sequence, Tuple, Union

import h5py
import numpy as np
import psutil
from rich.console import Console
from rich.progress import BarColumn, Progress
from rich.table import Table
from rich.text import Text

from ..utils.rich_progress_bars import PlaceHolderBar, WorkingDots
from .svd.clustering_betas import create_cluster_arg_dict, create_detector_arg_dict, group_betas
from .svd.plot_beta_clusters import plot_clusters_js
from .svd.pod_functions import calculate_v_and_betas
from .svd.subsampling_methods import create_reference_subsample, remap_random_subsample

# pylint: disable = too-many-lines


class DimredRunError(Exception):
    """Custom exception for errors during the dimensionality reduction"""

    def __init__(self, msg):
        self.message = msg


def get_application_header():
    """Prints the header of the command line tool"""

    return """

       ==== LASSO - AI ====

       visit us: [link=http://www.lasso.de/en]www.lasso.de/en[/link]
       mail: lasso@lasso.de
    """


def timestamp() -> str:
    """Get current timestamp as string

    Returns
    -------
    timestamp : str
        current timestamp as string
    """

    def add_zero(in_str) -> str:
        if len(in_str) == 1:
            return "0" + in_str
        return in_str

    loc_time = time.localtime()[3:6]
    h_str = add_zero(str(loc_time[0]))
    m_str = add_zero(str(loc_time[1]))
    s_str = add_zero(str(loc_time[2]))
    t_str = "[" + h_str + ":" + m_str + ":" + s_str + "]"
    return t_str


def parse_dimred_args():
    """Parse the arguments from the command line

    Returns
    -------
    args : `argparse.Namespace`
        parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="Python utility script for dimensionality reduction written by LASSO GmbH."
    )

    parser.add_argument(
        "simulation_runs",
        type=str,
        nargs="*",
        help="Simulation runs or patterns used to search for simulation runs.",
    )
    parser.add_argument(
        "--reference-run",
        type=str,
        help="Optional. Set the reference run instead of using the first entry in simulation runs.",
    )
    parser.add_argument(
        "--exclude-runs",
        type=str,
        nargs="*",
        default=[],
        help="Optional. Runs to exclude from the analysis.",
    )
    parser.add_argument(
        "--start-stage",
        type=str,
        nargs="?",
        default=DIMRED_STAGES[0],
        help="Optional. "
        f"At which specific stage to start the analysis ({', '.join(DIMRED_STAGES)}).",
    )
    parser.add_argument(
        "--end-stage",
        type=str,
        nargs="?",
        default=DIMRED_STAGES[-1],
        help="Optional. "
        f"At which specific stage to stop the analysis ({', '.join(DIMRED_STAGES)}).",
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        required=True,
        help="Project dir for temporary files. Must be specified to allow"
        + " restart at specific steps",
    )
    parser.add_argument(
        "--embedding-images",
        type=str,
        default="",
        help="Optional. Path to folder containing images of runs. Sample names must be numbers",
    )
    parser.add_argument(
        "--logfile-filepath",
        type=str,
        nargs="?",
        default="",
        help="Optional. Path for the logfile. A file will be created automatically"
        + "in the project dir if not specified.",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        nargs="?",
        default=max(1, psutil.cpu_count() - 1),
        help="Optional. Number of processes to use (default: n_cpu-1).",
    )
    parser.add_argument(
        "--part-ids",
        type=str,
        nargs="*",
        default=[],
        help="Optional. Part ids to process. By default all are taken.",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default="-1",
        help="Optional. Sets timestep to analyse. Uses last timestep if not set.",
    )
    parser.add_argument(
        "--html-name",
        type=str,
        default="3d-beta-plot",
        help="Optional. Sets the name of the generated 3D visualization. "
        + "Default is '3d_beta_plot'",
    )
    parser.add_argument(
        "--html-timestamp",
        action="store_true",
        help="""Optional. If set, the visualization will include a timestamp of yymmdd_hhmmss,
                         else the previous file will be overwritten""",
    )
    parser.add_argument(
        "--cluster-args",
        type=str,
        nargs="*",
        help="Optional. Arguments for clustering algorithms. "
        + "If not set, clustering will be skipped.",
    )
    parser.add_argument(
        "--outlier-args",
        type=str,
        nargs="*",
        help="Optional. Arguments for outlier detection before clustering.",
    )

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    return parser.parse_args(sys.argv[1:])


class DimredStage(enum.Enum):
    """Enum for all stages of the dimenstionality reduction"""

    REFERENCE_RUN = "REFERENCE_RUN"
    IMPORT_RUNS = "IMPORT_RUNS"
    REDUCTION = "REDUCTION"
    CLUSTERING = "CLUSTERING"
    EXPORT_PLOT = "EXPORT_PLOT"


DIMRED_STAGES = (
    DimredStage.REFERENCE_RUN.value,
    DimredStage.IMPORT_RUNS.value,
    DimredStage.REDUCTION.value,
    DimredStage.CLUSTERING.value,
    DimredStage.EXPORT_PLOT.value,
)


class HDF5FileNames(enum.Enum):
    """Enum for arrays in the hdf5 file"""

    SUBSAMPLE_SAVE_NAME = "subsample"
    SUBSAMPLED_GROUP_NAME = "subsampled_runs"
    BETAS_GROUP_NAME = "betas"
    V_ROB_SAVE_NAME = "v_rob"
    PLOT_LOAD_TIME = "t_load"
    SUBSAMPLE_PROCESS_TIME = "t_total"
    NR_CLUSTER = "nr_clusters"
    HAS_OUTLIERS = "has_outliers"
    OUTLIERS = "outlier"
    CLUSTER = "cluster"


class DimredRun:
    """Class to control and run the dimensionality reduction process"""

    # pylint: disable = too-many-instance-attributes

    reference_run: str
    simulation_runs: Sequence[str]
    exclude_runs: Sequence[str]
    project_dir: str
    img_path: Union[None, str]
    logfile_filepath: str
    n_processes: int
    part_ids: Sequence[int]
    timestep: int
    start_stage_index: int
    end_stage_index: int
    skip_valid: bool
    html_name: str
    html_set_timestamp: bool
    show_output: bool
    cluster_type: Union[None, str]
    detector_type: Union[None, str]
    cluster_args: Union[None, dict]
    detector_args: Union[None, dict]
    h5file: Union[None, h5py.File]
    use_folder_name: bool

    def __init__(
        self,
        simulation_runs: Sequence[str],
        start_stage: str,
        end_stage: str,
        project_dir: str,
        html_name: str = "3d-beta-plot",
        html_set_timestamp: bool = False,
        reference_run: Union[str, None] = None,
        console: Union[Console, None] = None,
        img_path: Union[None, str] = None,
        exclude_runs: Union[None, Sequence[str]] = None,
        logfile_filepath: Union[str, None] = None,
        n_processes: int = 1,
        part_id_filter: Union[None, Sequence[int]] = None,
        timestep: int = -1,
        show_output: bool = True,
        cluster_args: Union[None, Sequence[str]] = None,
        outlier_args: Union[None, Sequence[str]] = None,
    ):
        """Class handling a dimensionality reduction

        Parameters
        ----------
        simulation_runs : Sequence[str]
            simulation runs to analyze
        start_stage: str
            where to start
        end_stage: str
            where to stop
        project_dir : Union[None, str]
            required project directory for creation of buffer files. Allows restart in between.
        html_name: str
            Name of the output .html file
        html_set_timestamp: bool
            If true, the output .html will include a timestamp (hh_mm_ss) at the end of the filename
        reference_run : str
            filepath to the reference run.
            If not set, first entry in simulation_runs will be used as reference run.
        console: Union[rich.console.Console, None], default: None
            Console for information printing and logging.
            Rich offers pretty text printing, syntax highlighting etc.
        img_path: Union[None, str]
            optional image directory to show images in visualization.
        exclude_runs: Union[Sequence[str], None]
            optional list of runs to exclude from processing
        logfile_filepath : Union[str, None]
            path of the log file (always appends)
        n_processes: int
            number of processes to use during execution
        part_id_filter: Union[Sequence[int], None]
            which part ids to process
        timestep: int, default: -1
            specifies timestep to analyze in clustering and show in output visualization
        show_output: bool, default: True
            Set to false not to show the output html in the browser
        cluster_args: Union[None, [str]], default: None
            Arguments for cluster algorithm
        outlier_args: Union[None, [str]], default: None
            Arguments for outlier detection algorithm

        Notes
        -----
            Using a project directory allows to restart stages of the entire
            process.
        """

        # pylint: disable = too-many-arguments, too-many-locals

        # settings
        # Set up Rich Console and Rich logging
        self.console = console
        if self.console:
            self.console.print(get_application_header(), style="success", highlight=True)

        self.logfile_filepath = (
            logfile_filepath
            if logfile_filepath
            else os.path.join(project_dir, "logfile")
            if project_dir
            else ""
        )

        self._msg_option = "{:16s}: {}"

        # run variables
        # table is a rich format containing information of the variables
        table = Table(show_header=False)
        self.n_processes = self._parse_n_processes(n_processes, table)

        # check for correctly parsed simulation-runs
        if len(simulation_runs) == 0:
            err_msg = "No entries in positional argument 'simulation-runs'."
            err_msg += "\nIt is recommended to set the 'simulation-runs' arguments first!"
            self.raise_error(err_msg)

        # parse simulation and reference run
        # if no reference run was set use first simulation run
        (
            self.simulation_runs,
            self.reference_run,
            self.exclude_runs,
        ) = self._parse_simulation_and_reference_runs(
            simulation_runs, reference_run, tuple() if not exclude_runs else exclude_runs, table
        )

        # check if basename or foldername serves as unique identifier
        self.use_folder_name = os.path.basename(self.simulation_runs[0]) == os.path.basename(
            self.simulation_runs[1]
        )

        # set project dir and simulation runs
        self.project_dir = self._parse_project_dir(project_dir, table)
        self.part_ids = part_id_filter if part_id_filter is not None else tuple()
        if self.part_ids is not None and len(self.part_ids) != 0:
            table.add_row("selected parts", ",".join(str(entry) for entry in self.part_ids))
        self.timestep = timestep
        if timestep != -1:
            table.add_row("Timestep: ", str(timestep))

        # check if start_stage_index and end_stage_index are valid
        self.start_stage_index, self.end_stage_index = self._parse_stages(start_stage, end_stage)
        if self.console:
            self.console.print(table)

        # check valid image path
        self.img_path = self._check_img_path(img_path) if img_path else None

        # set cluster and outlier arguments
        self._parse_cluster_and_outlier_args(cluster_args, outlier_args)

        self.html_name = self._parse_html_name(html_name)
        self.html_set_timestamp = html_set_timestamp
        self.show_output = show_output

        self.pool = None

    def log(self, msg: str, style: Union[str, None] = None, highlight: bool = False):
        """Log a message

        Parameters
        ----------
        msg : str
            message to log
        style : Union[str, None]
            style of the message
        highlight : bool
            whether to highlight the message or not
        """
        if self.console:
            self.console.print(timestamp() + msg, style=style, highlight=highlight)

    def raise_error(self, err_msg: str):
        """
        Parameters
        ----------
        err_msg : str
            error message to be raised

        Raises
        ------
        RuntimeError
            raises an exception with error msg

        Notes
        -----
            Logs correctly and deals with open file handles.
        """

        err_msg_text = Text(err_msg, style="error")

        if not self.console:
            raise DimredRunError(err_msg)

        try:
            self.h5file.close()
            self.console.print("closed hdf5 file")
        except AttributeError:
            self.console.print("no hdf5 file to close")

        self.console.print(err_msg_text, style="error")
        if self.logfile_filepath:
            self.console.save_html(self.logfile_filepath)

        raise DimredRunError(err_msg)

    # pylint believes this function has different return statements
    # whereas it only has one.
    # pylint: disable = inconsistent-return-statements
    def _check_img_path(self, img_path: str) -> str:
        """checks if provided image path is valid"""

        if os.path.isdir(img_path):
            abs_path = os.path.abspath(img_path)
            js_path = re.sub(r"\\", "/", abs_path)
            return js_path

        err_msg = "provided argument --embedding.images is not a folder"
        self.raise_error(err_msg)

    def _parse_stages(self, start_stage: str, end_stage: str):

        # check validity
        if start_stage not in DIMRED_STAGES:
            err_msg = f"{start_stage} is not a valid stage. Try: {', '.join(DIMRED_STAGES)}."
            self.raise_error(err_msg)

        if end_stage not in DIMRED_STAGES:
            err_msg = f"{end_stage} is not a valid stage. Try: {', '.join(DIMRED_STAGES)}."
            self.raise_error(err_msg)

        # get indexes
        start_stage_index = DIMRED_STAGES.index(start_stage)
        end_stage_index = DIMRED_STAGES.index(end_stage)

        # check if start and end are in correct order
        if start_stage_index > end_stage_index:
            err_msg = (
                f"The specified end stage '{end_stage}' "
                f"comes before the start stage ({start_stage}). "
                f"Try the order: {', '.join(DIMRED_STAGES)}"
            )
            self.raise_error(err_msg)

        return start_stage_index, end_stage_index

    def _check_valid_stage_skip(self):
        # check if stage skip is valid
        if self.start_stage_index == DIMRED_STAGES.index(DimredStage.IMPORT_RUNS.value):
            self.log("Skipped setup stage", style="warning")
            if HDF5FileNames.SUBSAMPLE_SAVE_NAME.value not in self.h5file:  # type: ignore
                msg = "no reference sample found"
                self.raise_error(msg)
        elif self.start_stage_index == DIMRED_STAGES.index(DimredStage.REDUCTION.value):
            self.log("Skipped import stage", style="warning")
            if HDF5FileNames.SUBSAMPLED_GROUP_NAME.value not in self.h5file:  # type: ignore
                msg = "no subsampled samples found"
                self.raise_error(msg)
        elif self.start_stage_index == DIMRED_STAGES.index(DimredStage.CLUSTERING.value):
            self.log("Skipped reduction stage", style="warning")
            if (
                HDF5FileNames.V_ROB_SAVE_NAME.value not in self.h5file  # type: ignore
                or HDF5FileNames.BETAS_GROUP_NAME.value not in self.h5file
            ):  # type: ignore
                err_msg = "Could not find reduced betas and V_ROB"
                self.raise_error(err_msg)
        elif self.start_stage_index == DIMRED_STAGES.index(DimredStage.CLUSTERING.value):
            self.log("Skipped clustering stage", style="warning")

    def _parse_part_ids(self, part_ids: Union[Sequence[int], None]) -> Sequence[int]:

        if not part_ids:
            return tuple()

        assert all(isinstance(pid, int) for pid in part_ids), "All part ids must be of type 'int'"

        return part_ids

    def _parse_project_dir(self, project_dir: Union[str, None], table: Table):

        if not project_dir:
            return ""

        project_dir = os.path.abspath(project_dir)

        if os.path.isfile(project_dir):
            err_msg = (
                f"The project path '{project_dir}' is pointing at an existing file."
                " Change either the project path or move the file."
            )
            self.raise_error(err_msg)

        if not os.path.exists(project_dir):
            os.makedirs(project_dir, exist_ok=True)

        table.add_row("project-dir", project_dir)
        return project_dir

    def _parse_simulation_and_reference_runs(
        self,
        simulation_run_patterns: Sequence[str],
        reference_run_pattern: Union[None, str],
        exclude_runs: Sequence[str],
        table: Table,
    ) -> Tuple[Sequence[str], str, Sequence[str]]:

        # pylint: disable = too-many-locals

        # search all denoted runs
        simulation_runs = []
        for pattern in simulation_run_patterns:
            simulation_runs += glob.glob(pattern)
        simulation_runs = [
            os.path.normpath(filepath) for filepath in simulation_runs if os.path.isfile(filepath)
        ]

        # search all excluded runs
        runs_to_exclude = []
        for pattern in exclude_runs:
            runs_to_exclude += glob.glob(pattern)
        runs_to_exclude = [
            os.path.normpath(filepath) for filepath in runs_to_exclude if os.path.isfile(filepath)
        ]

        n_runs_before_filtering = len(simulation_runs)
        simulation_runs = [
            filepath for filepath in simulation_runs if filepath not in runs_to_exclude
        ]
        n_runs_after_filtering = len(simulation_runs)

        # check if simulation runs are valid

        simulation_runs_ok = len(simulation_runs) != 0

        if not simulation_runs_ok:
            err_msg = (
                "No simulation files could be found with the specified patterns. "
                "Check the argument 'simulation_runs'."
            )
            self.raise_error(err_msg)

        table.add_row("# simul.-files", str(len(simulation_runs)))

        table.add_row("# excluded files", f"{n_runs_before_filtering - n_runs_after_filtering}")

        # check for valid reference run
        reference_run = ""
        if reference_run_pattern:

            reference_run_ok = os.path.isfile(reference_run_pattern)
            if not reference_run_ok:
                err_msg = f"Filepath '{reference_run_pattern}' is not a file."
                self.raise_error(err_msg)

            reference_run = os.path.normpath(reference_run_pattern)
        else:
            # use first simulation run if no reference run was provided
            # check if enough simulation runs remain
            if len(simulation_runs) > 1:
                reference_run = simulation_runs[0]
            else:
                err_msg = "Number of Simulation runs after using first as reference run is zero."
                self.raise_error(err_msg)

        # add to table
        table.add_row("reference-run", reference_run)

        # remove the reference run from simulation runs
        if reference_run and reference_run in simulation_runs:
            simulation_runs.remove(reference_run)

        # sort it because we can!
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [atoi(c) for c in re.split(r"(\d+)", text)]

        simulation_runs = sorted(simulation_runs, key=natural_keys)

        return simulation_runs, reference_run, runs_to_exclude

    def _parse_cluster_and_outlier_args(
        self, cluster_args: Union[Sequence[str], None], outlier_args: Union[Sequence[str], None]
    ):
        """verifies correct oultier and cluster args, if provided"""

        # creates a valid argument dict for clustering arguments
        if cluster_args is None:
            self.cluster_type = None
            self.cluster_args = None
        else:
            result = create_cluster_arg_dict(cluster_args)

            # check for errors
            if isinstance(result, str):
                self.raise_error(result)
            else:
                self.cluster_type, self.cluster_args = result[0], result[1]

        # creates a valid argument dict for outlier detection arguments
        self.detector_type = None
        self.detector_args = {}

        if outlier_args:
            result = create_detector_arg_dict(outlier_args)
            # check for errors
            if isinstance(result, str):
                self.raise_error(result)
            self.detector_type = result[0]
            self.detector_args = result[1]

    def _parse_n_processes(self, n_processes: int, table: Table) -> int:

        if n_processes <= 0:
            err_msg = f"n-processes is '{n_processes}' but must be at least 1."
            self.raise_error(err_msg)

        table.add_row("n-processes", str(n_processes))
        return n_processes

    def _parse_html_name(self, html_name_string: str) -> str:

        html_name, replace_count = re.subn(r"[!§$%&/()=?\"\[\]{}\\.,;:<>|]", "", html_name_string)
        html_name = html_name.replace(" ", "-")

        if replace_count > 0:
            msg = (
                f"Replaced {replace_count} invalid characters for the html file name. "
                f"The new hmtl name is: {html_name}"
            )
            self.log(msg)

        return html_name

    def __enter__(self):
        self.pool = ProcessPoolExecutor(max_workers=self.n_processes)
        self.h5file = h5py.File(os.path.join(self.project_dir, "project_buffer.hdf5"), "a")
        self._check_valid_stage_skip()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.pool = None
        self.h5file.close()
        self.h5file = None

    def _perform_context_check(self):
        if self.pool is None:
            msg = "The class function can only be used in a 'with' block on the instance itself."
            self.raise_error(msg)

    def reset_project_dir(self):
        """resets the project directory entirely"""

        # delete folder
        if os.path.exists(self.project_dir):
            shutil.rmtree(self.project_dir)

        if self.project_dir:
            os.makedirs(self.project_dir, exist_ok=True)

    def process_reference_run(self):
        """Process the reference run"""

        # is a process pool up
        self._perform_context_check()

        msg = "Reference Subsample"
        self.log(msg)

        # init progress bar
        if self.console:
            prog = Progress("", WorkingDots())
        else:
            prog = PlaceHolderBar()
        with prog:
            ref_task = prog.add_task("", total=1)

            # Delete existing reference subsample
            if HDF5FileNames.SUBSAMPLE_SAVE_NAME.value in self.h5file:  # type: ignore
                del self.h5file[HDF5FileNames.SUBSAMPLE_SAVE_NAME.value]

            reference_sample = create_reference_subsample(self.reference_run, self.part_ids)

            prog.advance(ref_task)  # type: ignore

        if isinstance(reference_sample, str):
            self.raise_error(reference_sample)

        # create dataset in h5file
        h5_ref = self.h5file.create_dataset(
            HDF5FileNames.SUBSAMPLE_SAVE_NAME.value, data=reference_sample[0]
        )
        h5_ref.attrs[HDF5FileNames.PLOT_LOAD_TIME.value] = reference_sample[2]
        h5_ref.attrs[HDF5FileNames.SUBSAMPLE_PROCESS_TIME.value] = reference_sample[1]

        # log time and success
        self.log("Loadtime Reference subsample: " + str(reference_sample[2])[:5])
        self.log("Total time for Reference subsample: " + str(reference_sample[1])[:5])
        self.log("Reference subsample completed", style="success")

    def subsample_to_reference_run(self):
        """Subsamples all runs"""

        # pylint: disable = too-many-branches,too-many-locals

        self._perform_context_check()
        self.log("Subsampling")

        # init progress bar
        if self.console:
            prog = Progress(
                "[progress.description]{task.description}",
                WorkingDots(),
                BarColumn(),
                "{task.completed} of {task.total};",
                # SubsamplingWaitTime(self.n_processes)
            )
        else:
            prog = PlaceHolderBar()

        with prog:

            # define progressbar task
            task1 = prog.add_task(
                "[cyan]Subsampling plots [/cyan]", total=len(self.simulation_runs)
            )
            h5_ref = self.h5file[HDF5FileNames.SUBSAMPLE_SAVE_NAME.value]
            # prog.columns[4].update_avrg(h5_ref.attrs[HDF5FileNames.plot_load_time.value])

            submitted_samples = []

            # delete previous subsample entries
            if HDF5FileNames.SUBSAMPLED_GROUP_NAME.value in self.h5file:  # type: ignore
                del self.h5file[HDF5FileNames.SUBSAMPLED_GROUP_NAME.value]

            # submit all simulation runs
            for _, entry in enumerate(self.simulation_runs):
                name = "overwrite_this"
                if self.use_folder_name:
                    name = os.path.basename(os.path.split(entry)[0])
                else:
                    name = os.path.basename(entry)

                try:
                    future = self.pool.submit(
                        remap_random_subsample, entry, self.part_ids, h5_ref[:]
                    )

                    submitted_samples.append(np.array([name, future]))
                except Exception:
                    break

            # check if an error occurred
            # pylint: disable = protected-access, undefined-loop-variable
            if self.pool._broken and "entry" in locals():
                msg = f"Failed to load file: {entry}"
                self.raise_error(msg)

            # we measure required time here
            t_cum = 0
            t_cum_io = 0

            # prepare hdf5 file
            self.h5file.create_group(HDF5FileNames.SUBSAMPLED_GROUP_NAME.value)
            # This isn't very elegant, there must be a better way
            while not prog.finished:
                for i, sub in enumerate(submitted_samples):
                    if sub[1].done():
                        try:
                            if isinstance(sub[1].result()[0], str):
                                self.raise_error(sub[1].result())
                            h5_sample = self.h5file[
                                HDF5FileNames.SUBSAMPLED_GROUP_NAME.value
                            ].create_dataset(sub[0], data=sub[1].result()[0])
                            h5_sample.attrs[HDF5FileNames.PLOT_LOAD_TIME.value] = sub[1].result()[2]
                            h5_sample.attrs[HDF5FileNames.SUBSAMPLE_PROCESS_TIME.value] = sub[
                                1
                            ].result()[1]
                            submitted_samples.pop(i)
                            prog.advance(task1)  # type: ignore
                            t_cum_io += sub[1].result()[2]
                            t_cum += sub[1].result()[1]
                        except RuntimeError:
                            err_msg = f"Error while loading {sub}"
                            self.raise_error(err_msg)
                time.sleep(0.5)

            # calculate required time
            t_avrg = t_cum / len(self.simulation_runs)
            t_avrg_io = t_cum_io / len(self.simulation_runs)

        # log results
        self.log("Average Time per Subsampling Process: " + str(t_avrg)[0:5])
        self.log("Average Loadtime per sample: " + str(t_avrg_io)[0:5])

        self.log("Subsampling completed", style="success")

        # Finished: We either have all sub-sampled runs in the project_dir,
        # or a list containing all sub-sampled runs
        # Problem: we might be running into issues with available RAM?
        # 1000 runs, 30 timesteps, sub-sampled onto 2000 points -> 1,34GB

    def dimension_reduction_svd(self):
        """Calculate V_ROB and Betas"""

        # pylint: disable = too-many-locals

        # applying pod_functions.py
        # (TODO: lots of stuff in the pod_functions.py has to be overhauled)
        # save if appropriate into project_dir
        self.log("Dimension Reduction")

        if self.console:
            # prog = Progress("", WorkingDots())
            prog = Progress(
                "[progress.description]{task.description}",
                WorkingDots(),
                BarColumn(),
                "{task.completed} of {task.total} timesteps;",
                # SubsamplingWaitTime(self.n_processes)
            )
        else:
            prog = PlaceHolderBar()
        with prog:
            # deletes old files
            if HDF5FileNames.BETAS_GROUP_NAME.value in self.h5file:  # type: ignore
                del self.h5file[HDF5FileNames.BETAS_GROUP_NAME.value]
            if HDF5FileNames.V_ROB_SAVE_NAME.value in self.h5file:  # type: ignore
                del self.h5file[HDF5FileNames.V_ROB_SAVE_NAME.value]

            beta_group = self.h5file.create_group(HDF5FileNames.BETAS_GROUP_NAME.value)

            excluded_entries = [
                os.path.basename(os.path.split(entry)[0])
                if self.use_folder_name
                else os.path.basename(entry)
                for entry in self.exclude_runs
            ]

            valid_entries = [
                entry
                for entry in self.h5file[HDF5FileNames.SUBSAMPLED_GROUP_NAME.value].keys()
                if entry not in excluded_entries
            ]

            run_timesteps = np.array(
                [
                    self.h5file[HDF5FileNames.SUBSAMPLED_GROUP_NAME.value][entry].shape[0]
                    for entry in valid_entries
                ]
            )

            min_step = np.min(run_timesteps)
            max_step = np.max(run_timesteps)

            if min_step != max_step:
                warn_msg = (
                    "The timesteps fo the samples don't match, only "
                    + "processing up to timestep {}. Skipped {} timesteps"
                )
                warn_msg = warn_msg.format(min_step, max_step - min_step)
                self.log(warn_msg, style="warning")

            # add task after checking condition, else output looks wonky
            beta_task = prog.add_task("[cyan]Reducing Plots [/cyan]", total=int(min_step))

            sub_displ = np.stack(
                [
                    self.h5file[HDF5FileNames.SUBSAMPLED_GROUP_NAME.value][entry][:min_step, :]
                    for entry in valid_entries
                ]
            )

            result = calculate_v_and_betas(
                sub_displ, progress_bar=prog, task_id=beta_task
            )  # type: ignore
            # returns string if samplesize to small
            if isinstance(result, str):
                self.raise_error(result)

            v_rob, betas = result
            for i, sample in enumerate(
                self.h5file[HDF5FileNames.SUBSAMPLED_GROUP_NAME.value].keys()
            ):
                beta_group.create_dataset(sample, data=betas[i])

            self.h5file.create_dataset(HDF5FileNames.V_ROB_SAVE_NAME.value, data=v_rob)

        self.log("Dimension Reduction completed", style="success")

    def clustering_results(self):
        """clustering results"""

        # pylint: disable = too-many-locals

        self._perform_context_check()
        # delete old entries
        betas_group = self.h5file[HDF5FileNames.BETAS_GROUP_NAME.value]
        if HDF5FileNames.HAS_OUTLIERS.value in betas_group.attrs:
            del betas_group.attrs[HDF5FileNames.HAS_OUTLIERS.value]

        if HDF5FileNames.NR_CLUSTER.value in betas_group.attrs:
            del betas_group.attrs[HDF5FileNames.NR_CLUSTER.value]

        if not self.cluster_type and not self.detector_type:
            msg = "No arguments provided for clustering, clustering aborted"
            self.log(msg)
            return

        self.log("Clustering")

        # init progress bar
        if self.console:
            prog = Progress("", WorkingDots())
        else:
            prog = PlaceHolderBar()
        with prog:
            cluster_task = prog.add_task("", total=1)

            # performs clustering with provided arguments

            excluded_entries = [
                os.path.basename(os.path.split(entry)[0])
                if self.use_folder_name
                else os.path.basename(entry)
                for entry in self.exclude_runs
            ]

            beta_index = np.stack(
                [key for key in betas_group.keys() if key not in excluded_entries]
            )
            try:
                betas = np.stack(
                    [betas_group[entry][self.timestep, :3] for entry in beta_index]
                )  # betas_group.keys()])
            except ValueError:
                log_msg = (
                    "Invalid parameter for timestep. Set a valid timestep with --timestep.\n"
                    "To save time, you can restart the tool with --start-stage CLUSTERING."
                )
                self.log(log_msg, style="warning")
                t_max = betas_group[beta_index[0]][:].shape[0]
                err_msg = (
                    f"Timestep {self.timestep} is not a valid timestep. "
                    f"Samples have {t_max} timesteps. "
                    f"Choose a timestep between 0 and {t_max - 1}"
                )
                self.raise_error(err_msg)

            result = group_betas(
                beta_index,
                betas,
                cluster=self.cluster_type,
                cluster_params=self.cluster_args,
                detector=self.detector_type,
                detector_params=self.detector_args,
            )

            if isinstance(result, str):
                self.raise_error(result)

            id_cluster = result[1]

            # Save clusters
            if len(id_cluster) > 1:
                betas_group.attrs.create(HDF5FileNames.NR_CLUSTER.value, len(id_cluster))
            if self.detector_type is not None:
                # if attribute has_outliers is set, the first cluster contains the outliers
                # so all outliers can be found by searching for the cluster attribute "0"
                betas_group.attrs.create(HDF5FileNames.HAS_OUTLIERS.value, len(id_cluster[0]))
            for index, cluster in enumerate(id_cluster):
                for entry in cluster:
                    # Enter appropriate cluster as attribute
                    sample = betas_group[entry]
                    sample.attrs.create(HDF5FileNames.CLUSTER.value, index)

            prog.advance(cluster_task)  # type: ignore

        self.log("Clustering completed", style="success")

    def visualize_results(self):
        """creates an output .html file"""

        self._perform_context_check()
        self.log("Creating .html viz")
        betas_group = self.h5file[HDF5FileNames.BETAS_GROUP_NAME.value]
        mark_outliers = False

        excluded_entries = [
            os.path.basename(os.path.split(entry)[0])
            if self.use_folder_name
            else os.path.basename(entry)
            for entry in self.exclude_runs
        ]

        # check if clustering was performed, else load all betas into one pseudo-cluster
        if HDF5FileNames.NR_CLUSTER.value not in betas_group.attrs:

            # plotfunction expects list of cluster
            # we have no clusters -> we claim all is in one cluster

            # Create and load ids
            id_cluster = [
                np.stack([key for key in betas_group.keys() if key not in excluded_entries])
            ]

            # Create and load betas
            beta_cluster = [np.stack([betas_group[entry][-1] for entry in id_cluster[0]])]

        else:
            # check if outlier where detected
            if HDF5FileNames.HAS_OUTLIERS.value in betas_group.attrs:
                mark_outliers = True

            # index of all runs
            id_data = np.stack([key for key in betas_group.keys() if key not in excluded_entries])

            # create an index referencing each run to a cluster
            cluster_index = np.stack(
                [betas_group[entry].attrs[HDF5FileNames.CLUSTER.value] for entry in id_data]
            )

            # load betas & ids
            beta_data = np.stack([betas_group[entry][-1] for entry in id_data])

            # create list containing list of clusters
            beta_cluster = []
            id_cluster = []
            for i, cluster in enumerate(range(betas_group.attrs[HDF5FileNames.NR_CLUSTER.value])):
                chosen = np.where(cluster_index == cluster)[0]
                if len(chosen) > 0:
                    beta_cluster.append(beta_data[chosen])
                    id_cluster.append(id_data[chosen])
                elif len(chosen) == 0 and i == 0:
                    mark_outliers = False

        plot_clusters_js(
            beta_cluster,
            id_cluster,
            save_path=self.project_dir,
            img_path=self.img_path,
            mark_outliers=mark_outliers,
            mark_timestamp=self.html_set_timestamp,
            filename=self.html_name,
            show_res=self.show_output,
        )
        self.log("Finished creating viz", style="success")
