#!/usr/bin/env python3
"""
                SEISFLOWS SPECFEM2D WORKSTATION EXAMPLE 1

This example will run two iterations of an inversion to assess misfit between
a homogeneous halfspace model and a slightly perturbed homogeneous halfspace
model using 3 events and 1 receiver.

.. note::
    You can change the number of events (NTASK) and iterations (NITER) by
    changing the constants below the import statements

.. warning::
    Because we are using 3 events and only 1 receiver, the results of this
    inversion will be of questionable quality. This example is only meant
    to highlight how SeisFlows operates during an inversion workflow.

.. note::
    The tasks involved include:
    1. Download, configure and compile SPECFEM2D
    2. Set up a SPECFEM2D working directory
    3. Generate starting model from Tape2007 example
    4. Generate target model w/ perturbed starting model
    5. Set up a SeisFlows working directory
    6. Run two iterations of an inversion workflow

.. rubric::
    $ seisflows examples run 1
"""
import os
import sys
import glob
import subprocess
import numpy as np

from seisflows.tools import msg
from seisflows.tools.config import Dict
from seisflows.seisflows import SeisFlows
from seisflows.tools.unix import cd, cp, rm, ln, mv, mkdir


class SFExample2D:
    """
    A class for running SeisFlows examples. Simplifies calls structure so that
    multiple example runs can benefit from the code written here
    """
    def __init__(self, ntask=None, niter=None, nsta=None, nproc=None,
                 method="run", specfem2d_repo=None):
        """
        Set path structure which is used to navigate around SPECFEM repositories
        and the example working directory

        :type ntask: int
        :param ntask: number of events to use in inversion, between 1 and 25.
            defaults to 3
        :type niter: int
        :param niter: number of iterations to run. defaults to 2
        :type nsta: int
        :param nsta: number of stations to include in inversion, between 1 and
            131
        :type method: str
        :param method: method for running the example problem, can be:
            * 'run': setup and run the example problem
            * 'setup': only setup the example problem, do not `submit` job
        :type specfem2d_repo: str
        :param specfem2d_repo: path to the SPECFEM2D directory which should
            contain binary executables. If not given, SPECFEM2D will be
            downloaded configured and compiled automatically.
        """
        self.cwd = os.getcwd()
        self.sem2d_paths, self.workdir_paths = self.define_dir_structures(
            cwd=self.cwd, specfem2d_repo=specfem2d_repo
        )
        # We set defaults here because `seisflows examples` may input these
        # values as NoneType which would override __init__ defaults.
        self.ntask = ntask or 1
        assert(1 <= self.ntask <= 25), \
            f"number of tasks/events must be between 1 and 25, not {self.ntask}"
        self.niter = niter or 1
        assert(1 <= self.niter <= np.inf), \
            f"number of iterations must be between 1 and inf, not {self.niter}"
        self.nsta = nsta or 1
        # -1 because it represents index but we need to talk in terms of count
        assert(1 <= self.nsta <= 132), \
            f"number of stations must be between 1 and 131, not {self.nsta}"
        self.nproc = nproc or 1  # must be 1 for Examples 1-3

        # This bool information is provided by the User running 'setup' or 'run'
        self.run_example = bool(method == "run")

        # Command line tool to use $ seisflows <cmd> from inside Python
        # Zero out sys.argv to ensure that no arguments are given to the CLI
        sys.argv = [sys.argv[0]]
        self.sf = SeisFlows()

        # Set the main SeisFlows modules prior to running the configure() cmd.
        self._modules = {
            "workflow": "inversion",
        }
        
        # SeisFlows parameters are set as an attribute so that other examples 
        # can overwrite or override
        self._parameters = {
            "ntask": self.ntask,  # default 3 sources for this example
            "materials": "elastic",  # how velocity model parameterized
            "density": False,  # update density or keep constant
            "data_format": "ascii",  # how to output synthetic seismograms
            "start": 1,  # first iteration
            "end": self.niter,  # final iteration -- we will run 2
            "step_count_max": 5,  # will cause iteration 2 to fail
            "data_case": "synthetic",  # synthetic-synthetic inversion
            "components": "Y",  # only Y component seismograms avail.
            "attenuation": False,
            "misfit": "traveltime",  # cross-correlation phase measure
            "adjoint": "traveltime",  # cross-correlation phase measure
            "path_specfem_bin": self.workdir_paths.bin,
            "path_specfem_data": self.workdir_paths.data,
            "path_model_init": self.workdir_paths.model_init,
            "path_model_true": self.workdir_paths.model_true,
        }

        # Used to configure SPECFEM2D binaries
        self._configure_cmd = "./configure"

    def print_dialogue(self):
        """
        Print help/system dialogue message that explains the setup of th
        this workflow
        """
        print(msg.ascii_logo_small)
        print(msg.cli(
            f"This is a [SPECFEM2D] [WORKSTATION] example, which will "
            f"run an inversion to assess misfit between two homogeneous "
            f"halfspace models with slightly different velocities. "
            f"[{self.ntask} events, 1 station, {self.niter} iterations]. "
            f"The tasks involved include: ",
            items=["1. (optional) Download, configure, compile SPECFEM2D",
                   "2. [Setup] a SPECFEM2D working directory",
                   "3. [Setup] starting model from 'Tape2007' example",
                   "4. [Setup] target model w/ perturbed starting model",
                   "5. [Setup] a SeisFlows working directory",
                   "6. [Run] the inversion workflow"],
            header="seisflows example 1",
            border="=")
        )

    @staticmethod
    def define_dir_structures(cwd, specfem2d_repo, ex="Tape2007"):
        """
        Define the example directory structure, which will contain abridged
        versions of the SPECFEM2D working directory

        :type cwd: str
        :param cwd: current working directory
        :type specfem2d_repo: str
        :param specfem2d_repo: location of the SPECFEM2D repository
        :type ex: str
        :type ex: The name of the example problem inside SPECFEM2D/EXAMPLES
        """
        if specfem2d_repo is None or not os.path.exists(specfem2d_repo):
            print(f"No existing SPECFEM2D repo given, default to: "
                  f"{cwd}/specfem2d")
            specfem2d_repo = os.path.join(cwd, "specfem2d")

        # This defines required structures from the SPECFEM2D repository
        sem2d = {
            "repo": specfem2d_repo,
            "bin": os.path.join(specfem2d_repo, "bin"),
            "data": os.path.join(specfem2d_repo, "DATA"),
            "example": os.path.join(specfem2d_repo, "EXAMPLES", ex),
            "example_data": os.path.join(specfem2d_repo, "EXAMPLES", ex, "DATA")
        }
        # This defines a working directory structure which we will create
        working_directory = os.path.join(cwd, "specfem2d_workdir")
        workdir = {
            "workdir": os.path.join(working_directory),
            "bin": os.path.join(working_directory, "bin"),
            "data": os.path.join(working_directory, "DATA"),
            "output": os.path.join(working_directory, "OUTPUT_FILES"),
            "model_init": os.path.join(working_directory, "OUTPUT_FILES_INIT"),
            "model_true": os.path.join(working_directory, "OUTPUT_FILES_TRUE"),
        }
        return Dict(sem2d), Dict(workdir)

    def download_specfem2d(self):
        """
        Download the latest version of SPECFEM2D from GitHub, devel branch.
        Last successfully tested 4/28/22
        """
        if not os.path.exists(self.sem2d_paths.repo):
            cmd = ("git clone --recursive --branch devel --depth=1 " 
                   "https://github.com/geodynamics/specfem2d.git")

            print(f"Downloading SPECFEM2D with command: {cmd}")
            subprocess.run(cmd, shell=True, check=True)

        assert self.sem2d_paths.repo, (
            f"User supplied SPECFEM2D directory '{self.sem2d_paths.repo}' "
            f"does not exist, please check your path and try again."
        )

    def configure_specfem2d(self):
        """
        Run ./configure within the SPECFEM2D repo directory.
        This function assumes it is being run from inside the repo. Should guess
        all the configuration options. Probably the least stable part of the
        example
        """
        cd(self.sem2d_paths.repo)
        try:
            if not os.path.exists("./config.log"):
                print(f"Configuring SPECFEM2D with command: "
                      f"{self._configure_cmd}")
                # Ignore the configure outputs from SPECFEM
                subprocess.run(self._configure_cmd, shell=True, check=True,
                               stdout=subprocess.DEVNULL)
            else:
                print("SPECFEM2D already configured, skipping 'configure'")
        except subprocess.CalledProcessError as e:
            print(f"SPECFEM `configure` step has failed, please check and "
                  f"retry. If this command repeatedly fails, you may need "
                  f"to configure SPECFEM2D manually.\n{e}")
            sys.exit(-1)

    def make_specfem2d_executables(self):
        """
        Run `$ make all` in SPECFEM2D to create binary executables
        """
        try:
            if not glob.glob("./bin/x*"):
                cmd = "make all"
                print(f"Making SPECFEM2D binaries with command: {cmd}")
                # Ignore the make outputs from SPECFEM
                subprocess.run(cmd, shell=True, check=True,
                               stdout=subprocess.DEVNULL)
            else:
                print("executables found in SPECFEM2D/bin directory, "
                      "skipping 'make'")
        except subprocess.CalledProcessError as e:
            print(f"SPECFEM 'make' step has failed, please check and "
                  f"retry. If this command repeatedly fails, you may need "
                  f"to compile SPECFEM2D manually.\n{e}")
            sys.exit(-1)

        # Symlink the finished repo into the working directory so that any
        # subsequent runs won't need to have the user re-type repo location
        if not os.path.exists(os.path.join(self.cwd, "specfem2d")):
            print("symlinking existing specfem2D repository to cwd")
            ln(self.sem2d_paths.repo, os.path.join(self.cwd, "specfem2d"))

    def create_specfem2d_working_directory(self):
        """
        Create the working directory where we will generate our initial and
        final models using one of the SPECFEM2D examples
        """
        assert(os.path.exists(self.sem2d_paths["example"])), (
            f"SPECFEM2D/EXAMPLE directory: '{self.sem2d['example']}' "
            f"does not exist, please check this path and try again."
        )

        # Incase this example has written before, remove dir. that were created
        rm(self.workdir_paths.workdir)
        mkdir(self.workdir_paths.workdir)

        # Copy the binary executables and DATA from the SPECFEM2D example
        cp(self.sem2d_paths.bin, self.workdir_paths.bin)
        cp(self.sem2d_paths.example_data, self.workdir_paths.data)

        # Make sure that SPECFEM2D can find the expected files in the DATA/ dir
        cd(self.workdir_paths.data)
        rm("SOURCE")
        ln("SOURCE_001", "SOURCE")
        rm("Par_file")
        ln("Par_file_Tape2007_onerec", "Par_file")

    def setup_specfem2d_for_model_init(self):
        """
        Make some adjustments to the original parameter file to.
        This function assumes it is running from inside the SPECFEM2D/DATA dir

        """
        cd(self.workdir_paths.data)
        assert(os.path.exists("Par_file")), f"I cannot find the Par_file!"

        print("> Setting the SPECFEM2D Par_file for SeisFlows compatiblility")

        self.sf.sempar("nproc", self.nproc)
        self.sf.sempar("setup_with_binary_database", 1)  # create .bin files
        self.sf.sempar("save_model", "binary")  # output model in .bin format
        self.sf.sempar("save_ASCII_kernels", ".false.")  # kernels also .bin

        rm(self.workdir_paths.output)
        mkdir(self.workdir_paths.output)

    def setup_specfem2d_for_model_true(self):
        """
        Make some adjustments to the parameter file to create the final velocity
        model. This function assumes it is running from inside the
        SPECFEM2D/DATA directory
        """
        cd(self.workdir_paths.data)
        assert(os.path.exists("Par_file")), f"I cannot find the Par_file!"

        print("> Updating initial homogeneous velocity model values")
        new_model = "1 1 2600.d0 5900.d0 3550.0d0 0 0 10.d0 10.d0 0 0 0 0 0 0"
        self.sf.sempar("velocity_model", new_model)

    def run_xspecfem2d_binaries(self):
        """
        Runs the xmeshfem2d and then xspecfem2d binaries using subprocess and then
        do some cleanup to get files in the correct locations. Assumes that we
        can run the binaries directly with './'
        """
        cd(self.workdir_paths.workdir)

        cmd_mesh = f"./bin/xmeshfem2D > OUTPUT_FILES/mesher.log.txt"
        cmd_spec = f"./bin/xspecfem2D > OUTPUT_FILES/solver.log.txt"

        for cmd in [cmd_mesh, cmd_spec]:
            print(f"Running SPECFEM2D with command: {cmd}")
            subprocess.run(cmd, shell=True, check=True,
                           stdout=subprocess.DEVNULL)

    def cleanup_xspecfem2d_run(self, choice=None):
        """
        Do some cleanup after running the SPECFEM2D binaries to make sure files
        are in the correct locations, and rename the OUTPUT_FILES directory so
        that it does not get overwritten by subsequent runs

        :type choice: str
        :param choice: Rename the OUTPUT_FILES directory with a suffix tag
            msut be 'INIT' or 'TRUE'. If None, will not rename but the
        """
        cd(self.workdir_paths.workdir)
        print("> Cleaning up after xspecfem2d, setting up for new run")

        # SPECFEM2D outputs its models in the DATA/ directory by default,
        # while SeisFlows expects this in the OUTPUT_FILES/ directory (which is
        # the default in SPECFEM3D)
        mv(glob.glob("DATA/*bin"), self.workdir_paths.output)

        if choice == "INIT":
            mv(self.workdir_paths.output, self.workdir_paths.model_init)
            # Create a new OUTPUT_FILES/ directory for TRUE run
            rm(self.workdir_paths.output)
            mkdir(self.workdir_paths.output)
        elif choice == "TRUE":
            mv(self.workdir_paths.output, self.workdir_paths.model_true)

    def setup_seisflows_working_directory(self):
        """
        Create and set the SeisFlows parameter file, making sure all required
        parameters are set correctly for this example problem
        """
        cd(self.cwd)

        self.sf.setup(force=True)  # Force will delete existing parameter file
        for key, val in self._modules.items():
            self.sf.par(key, val)

        self.sf.configure()

        # Adjust the parameters.yaml file
        for key, val in self._parameters.items():
            self.sf.par(key, val)

    def finalize_specfem2d_par_file(self):
        """
        Final changes to the SPECFEM2D Par_file before running SeisFlows.
        Par_file will be used to control all the child specfem2d directories.
        Need to tell them to read models from .bin files, and to use existing
        station files rather than create them from the Par_file
        """
        print("> Finalizing SPECFEM2D Par_file for SeisFlows example")

        cd(self.workdir_paths.data)
        self.sf.sempar("model", "gll")  # GLL so SPECFEM reads .bin files
        self.sf.sempar("use_existing_stations", ".true.")  # Use STATIONS file
        self.sf.sempar("nproc", self.nproc)

        # Assign STATIONS_checker file which has 132 stations
        rm("STATIONS")

        with open("STATIONS_checker", "r") as f:
            lines = f.readlines()

        print(f"> Using {self.nsta} stations in this SeisFlows example")
        with open("STATIONS", "w") as f:
            f.writelines(lines[:self.nsta])

    def run_sf_example(self):
        """
        Use subprocess to run the SeisFlows example we just set up
        """
        print(msg.cli("RUNNING SEISFLOWS EXAMPLE WORKFLOW", border="="))
        cd(self.cwd)
        try:
            subprocess.run("seisflows submit", check=True, shell=True)
        except subprocess.CalledProcessError as e:
            print(msg.cli("EXAMPLE FAILED", items=[str(e)], border="="))
            sys.exit(-1)
        print(msg.cli("EXAMPLE COMPLETED SUCCESFULLY", border="="))

    def main(self):
        """
        Setup the example and then optionally run the actual seisflows workflow
        """
        print(msg.cli("EXAMPLE SETUP", border="="))

        # Step 1: Download and configure SPECFEM2D, make binaries. Optional
        self.download_specfem2d()
        self.configure_specfem2d()
        self.make_specfem2d_executables()
        # Step 2: Create a working directory and generate initial/final models
        self.create_specfem2d_working_directory()
        # Step 2a: Generate MODEL_INIT, rearrange consequent directory structure
        print(msg.cli("GENERATING INITIAL MODEL", border="="))
        self.setup_specfem2d_for_model_init()
        self.run_xspecfem2d_binaries()
        self.cleanup_xspecfem2d_run(choice="INIT")
        # Step 2b: Generate MODEL_INIT, rearrange consequent directory structure
        print(msg.cli("GENERATING TRUE/TARGET MODEL", border="="))
        self.setup_specfem2d_for_model_true()
        self.run_xspecfem2d_binaries()
        self.cleanup_xspecfem2d_run(choice="TRUE")
        # Step 3: Prepare Par_file and directory for MODEL_TRUE generation
        self.setup_seisflows_working_directory()
        self.finalize_specfem2d_par_file()
        print(msg.cli("COMPLETE EXAMPLE SETUP", border="="))
        # Step 4: Run the workflwo
        if self.run_example:
            self.run_sf_example()

