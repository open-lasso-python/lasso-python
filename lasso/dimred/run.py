from rich.console import Console
from rich.theme import Theme

from lasso.dimred.dimred_run import (
    DIMRED_STAGES,
    DimredRun,
    DimredRunError,
    DimredStage,
    parse_dimred_args,
)


def main():
    """Runs the dimensionality reduction CLI"""

    # parse command line stuff
    parser = parse_dimred_args()
    log_theme = Theme(
        {"info": "royal_blue1", "success": "green", "warning": "dark_orange3", "error": "bold red"}
    )
    console = Console(theme=log_theme, record=True, highlight=False)

    try:
        # parse settings from command line
        dimred_run = DimredRun(
            reference_run=parser.reference_run,
            simulation_runs=parser.simulation_runs,
            console=console,
            exclude_runs=parser.exclude_runs,
            project_dir=parser.project_dir,
            html_name=parser.html_name,
            html_set_timestamp=parser.html_timestamp,
            img_path=parser.embedding_images,
            logfile_filepath=parser.logfile_filepath,
            n_processes=parser.n_processes,
            part_id_filter=parser.part_ids,
            start_stage=parser.start_stage,
            end_stage=parser.end_stage,
            timestep=parser.timestep,
            cluster_args=parser.cluster_args,
            outlier_args=parser.outlier_args,
        )

        # do the thing
        console.print()
        console.print("   ---- Running Routines ----   ")
        console.print()

        # initiate threading pool for handling jobs
        with dimred_run:

            # setup
            if (
                dimred_run.start_stage_index
                <= DIMRED_STAGES.index(DimredStage.REFERENCE_RUN.value)
                <= dimred_run.end_stage_index
            ):
                dimred_run.process_reference_run()

            # import
            if (
                dimred_run.start_stage_index
                <= DIMRED_STAGES.index(DimredStage.IMPORT_RUNS.value)
                <= dimred_run.end_stage_index
            ):
                dimred_run.subsample_to_reference_run()

            # math
            if (
                dimred_run.start_stage_index
                <= DIMRED_STAGES.index(DimredStage.REDUCTION.value)
                <= dimred_run.end_stage_index
            ):
                dimred_run.dimension_reduction_svd()

            # clustering
            if (
                dimred_run.start_stage_index
                <= DIMRED_STAGES.index(DimredStage.CLUSTERING.value)
                <= dimred_run.end_stage_index
            ):
                dimred_run.clustering_results()

            # export
            if (
                dimred_run.start_stage_index
                <= DIMRED_STAGES.index(DimredStage.EXPORT_PLOT.value)
                <= dimred_run.end_stage_index
            ):
                dimred_run.visualize_results()

            # print logfile
            console.save_html(dimred_run.logfile_filepath)

    # Catch if DimredrunError was called
    except DimredRunError as err:
        print(err)


if __name__ == "__main__":
    main()
