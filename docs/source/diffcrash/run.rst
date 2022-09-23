
run
---

What is Diffcrash?
``````````````````

    Diffcrash_ is a software from Sidact_ which is designed
    for robustness analysis of simulation runs. It can be used as 
    a set of independent executables or as a postprocessor plugin.
    Diffcrash_ itself must be licensed. Please therefore contact either
    Sidact_ directly or LASSO_. This commmand line utility makes 
    running a Diffcrash analysis much easier.

    .. _LASSO: https://www.lasso.de/en
    .. _Sidact: http://www.sidact.com/
    .. _Diffcrash: http://www.sidact.com/diffcrash.html

How to use the utility?
```````````````````````

    You can get the run info by performing:

    ::

        python -m lasso.diffcrash.run --help


       ==== D I F F C R A S H ====

       a LASSO GmbH utility script

        usage: run.py [-h] --reference-run REFERENCE_RUN
                    [--exclude-runs [EXCLUDE_RUNS [EXCLUDE_RUNS ...]]] --crash-code
                    CRASH_CODE [--start-stage [START_STAGE]]
                    [--end-stage [END_STAGE]] [--diffcrash-home [DIFFCRASH_HOME]]
                    [--use-id-mapping [USE_ID_MAPPING]]
                    [--project-dir [PROJECT_DIR]] [--config-file [CONFIG_FILE]]
                    [--parameter-file [PARAMETER_FILE]]
                    [--n-processes [N_PROCESSES]]
                    [simulation_runs [simulation_runs ...]]

        Python utility script for Diffcrash written by LASSO GmbH.

        positional arguments:
        simulation_runs       Simulation runs or patterns used to search for
                                simulation runs.

        optional arguments:
        -h, --help            show this help message and exit
        --reference-run REFERENCE_RUN
                                filepath of the reference run.
        --exclude-runs [EXCLUDE_RUNS [EXCLUDE_RUNS ...]]
                                Runs to exclude from the analysis.
        --crash-code CRASH_CODE
                                Which crash code is used ('dyna', 'pam' or 'radioss').
        --start-stage [START_STAGE]
                                At which specific stage to start the analysis (SETUP,
                                IMPORT, MATH, EXPORT, MATRIX, EIGEN, MERGE).
        --end-stage [END_STAGE]
                                At which specific stage to stop the analysis (SETUP,
                                IMPORT, MATH, EXPORT, MATRIX, EIGEN, MERGE).
        --diffcrash-home [DIFFCRASH_HOME]
                                Home directory where Diffcrash is installed. Uses
                                environment variable 'DIFFCRASHHOME' if unspecified.
        --use-id-mapping [USE_ID_MAPPING]
                                Whether to use id-based mapping (default is nearest
                                neighbour).
        --project-dir [PROJECT_DIR]
                                Project dir to use for femzip.
        --config-file [CONFIG_FILE]
                                Path to the config file.
        --parameter-file [PARAMETER_FILE]
                                Path to the parameter file.
        --n-processes [N_PROCESSES]
                                Number of processes to use (default: max-1).

    It is important to specify a `--reference-run` for the analysis. 
    If the reference run is contained within the rest of the 
    `simulation_runs`, it is automatically removed from that list. 
    `simulation_runs` can be either tagged individually or by using 
    placeholders for entire directories (e.g. '\*.fz') and 
    subdirectories (e.g. '/\*\*/\*.fz').

    .. WARNING::
        Every run clears the project directory entirely!

Example
```````

    ::

        python -m lasso.diffcrash.run
            --reference-run ./runs/run_1.fz 
            --crash-code dyna 
            --project-dir diffcrash_project  
            ./runs/*.fz

            
            ==== D I F F C R A S H ==== 

            a LASSO GmbH utility script
            
        [/] diffcrash-home  : /sw/Linux/diffcrash/V6.1.24
        [/] project-dir     : test-example-project
        [/] crash-code      : dyna
        [/] reference-run   : bus/run_1.fz
        [/] use-id-mapping  : False
        [/] # simul.-files  : 37
        [/] # excluded files: 0
        [/] config-file     : None
        [!] Config file missing. Consider specifying the path with the option '--config-file'.
        [/] parameter-file  : None
        [!] Parameter file missing. Consider specifying the path with the option '--parameter-file'.
        [/] n-processes     : 4

        ---- Running Routines ----   

        [✔] Running Setup ... done in 3.88s
        [✔] Running Imports ... done in 58.20s   
        [✔] Running Math ... done in 56.22s   
        [✔] Running Export ... done in 2.22s   
        [✔] Running Matrix ... done in 9.78s   
        [✔] Running Eigen ... done in 0.46s   
        [✔] Running Merge ... done in 23.29s
