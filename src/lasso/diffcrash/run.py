from concurrent import futures

from lasso.diffcrash.diffcrash_run import (
    DC_STAGE_EIGEN,
    DC_STAGE_EXPORT,
    DC_STAGE_IMPORT,
    DC_STAGE_MATH,
    DC_STAGE_MATRIX,
    DC_STAGE_MERGE,
    DC_STAGE_SETUP,
    DC_STAGES,
    DiffcrashRun,
    parse_diffcrash_args,
)

from ..logging import str_error


def _parse_stages(start_stage: str, end_stage: str):

    # check validity
    if start_stage not in DC_STAGES or end_stage not in DC_STAGES:
        raise ValueError(
            str_error(f"{start_stage} is not a valid stage. Try: {', '.join(DC_STAGES)}.")
        )

    # get indexes
    start_stage_index = DC_STAGES.index(start_stage)
    end_stage_index = DC_STAGES.index(end_stage)

    # check if start and end are in correct order
    if start_stage_index > end_stage_index:
        raise ValueError(
            str_error(
                f"The specified end stage '{end_stage}' comes before "
                f"the start stage ({start_stage}). "
                f"Try the order: {', '.join(DC_STAGES)}"
            )
        )

    return start_stage_index, end_stage_index


def main():
    """Main function for running diffcrash"""

    # parse command line stuff
    parser = parse_diffcrash_args()

    # parse settings from command line
    diffcrash_run = DiffcrashRun(
        project_dir=parser.project_dir,
        crash_code=parser.crash_code,
        reference_run=parser.reference_run,
        exclude_runs=parser.exclude_runs,
        simulation_runs=parser.simulation_runs,
        diffcrash_home=parser.diffcrash_home,
        use_id_mapping=parser.use_id_mapping,
        config_file=parser.config_file,
        parameter_file=parser.parameter_file,
        n_processes=parser.n_processes,
    )

    # determine start and end stages
    start_stage_index, end_stage_index = _parse_stages(parser.start_stage, parser.end_stage)

    # remove old stuff
    if start_stage_index == 0:
        diffcrash_run.clear_project_dir()
    diffcrash_run.create_project_dirs()

    # do the thing
    print()
    print("   ---- Running Routines ----   ")
    print()

    # initiate threading pool for handling jobs
    with futures.ThreadPoolExecutor(max_workers=diffcrash_run.n_processes) as pool:

        # setup
        if start_stage_index <= DC_STAGES.index(DC_STAGE_SETUP) <= end_stage_index:
            diffcrash_run.run_setup(pool)

        # import
        if start_stage_index <= DC_STAGES.index(DC_STAGE_IMPORT) <= end_stage_index:
            diffcrash_run.run_import(pool)

        # math
        if start_stage_index <= DC_STAGES.index(DC_STAGE_MATH) <= end_stage_index:
            diffcrash_run.run_math(pool)

        # export
        if start_stage_index <= DC_STAGES.index(DC_STAGE_EXPORT) <= end_stage_index:
            diffcrash_run.run_export(pool)

        # matrix
        if start_stage_index <= DC_STAGES.index(DC_STAGE_MATRIX) <= end_stage_index:
            diffcrash_run.run_matrix(pool)

        # eigen
        if start_stage_index <= DC_STAGES.index(DC_STAGE_EIGEN) <= end_stage_index:
            diffcrash_run.run_eigen(pool)

        # merge
        if start_stage_index <= DC_STAGES.index(DC_STAGE_MERGE) <= end_stage_index:
            diffcrash_run.run_merge(pool)

    # final spacing
    print()


if __name__ == "__main__":
    main()
