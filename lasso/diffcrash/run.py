

from concurrent import futures
from lasso.logging import str_error
from lasso.diffcrash.DiffcrashRun import DiffcrashRun, parse_diffcrash_args, DC_STAGES, DC_STAGE_SETUP, DC_STAGE_IMPORT, DC_STAGE_MATH, DC_STAGE_EXPORT, DC_STAGE_MATRIX, DC_STAGE_EIGEN, DC_STAGE_MERGE


def parse_stages(start_stage: str, end_stage: str):

    # check validity
    if start_stage not in DC_STAGES or end_stage not in DC_STAGES:
        raise ValueError(str_error("{0} is not a valid stage. Try: {1}.".format(
            start_stage, ", ".join(DC_STAGES))))

    # get indexes
    start_stage_index = DC_STAGES.index(start_stage)
    end_stage_index = DC_STAGES.index(end_stage)

    # check if start and end are in correct order
    if start_stage_index > end_stage_index:
        raise ValueError(str_error("The specified end stage '{0}' comes before the start stage ({1}). Try the order: {2}".format(
            end_stage, start_stage, ', '.join(DC_STAGES))))

    return start_stage_index, end_stage_index


def main():

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
        n_processes=parser.n_processes
    )

    # determine start and end stages
    start_stage_index, end_stage_index = parse_stages(
        parser.start_stage, parser.end_stage)

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

        # TODO EXPORT_add
        '''
        if returnCode == 0:
            if len(export_item_list) > 1:
                print("Export add ...")
                exportadd_functionals = [export_item_list[1]]
                if len(export_item_list) > 2:
                    for i in range(2, len(export_item_list)):
                        exportadd_functionals.append(export_item_list[i])
                exportadd_args = [os.path.join(DIFFCRASHHOME, "DFC_Export_add_" + CRASHCODE), project_dir, os.path.join(
                    project_dir, export_item_list[0] + file_extension), os.path.join(project_dir, "EXPORT_ADD") + file_extension]
                for i in range(0, len(exportadd_functionals)):
                    exportadd_args.append(exportadd_functionals[i])
                returnCode = startproc(exportadd_args)
            else:
                for i in range(1, len(export_item_list)):
                    print("Export", export_item_list[i], "...")
                    returnCode = startproc(
                        [os.path.join(DIFFCRASHHOME, "DFC_Export_" + CRASHCODE), project_dir, export_item_list[i]])
        '''

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
