#!/usr/bin/env python3
"""
A workflow similar to Forward but for ambient noise simulations
"""
import numpy as np
import os
from glob import glob

from seisflows import logger
from seisflows.tools import unix
from seisflows.tools.model import Model
from seisflows.workflow.forward import Forward


class ForwardFwani(Forward):
    """
    Forward Noise Workflow
    ----------------
    Dummy

    Parameters
    ----------
    :type dummy:
    :param dummy

    Paths
    -----
    :type path_scratch_local: str
    :param path_scratch_local: optional path to a directory where all solver
        simulations will be carried
    ***
    """
    __doc__ = Forward.__doc__ + __doc__

    def __init__(self, modules=None, path_scratch_local=None, **kwargs):
        """
        Instantiate ForwardNoise-specific parameters

        :type dummy:
        :param dummy:
        """
        super().__init__(**kwargs)

        self._modules = modules
        self.path_scratch_local = path_scratch_local

    def setup(self):
        super().setup()

        # traces must be exported so we can recover the observations when
        # `data_case: synthetic`
        assert(self.export_traces), ("Workflow `ForwardFwani` requires "
                                     "`export_traces: True`")

        self.solver.path.scratch_local = self.path_scratch_local
        self.solver.path.scratch_project = self.solver.path.scratch

        # Some solver directories can be lost when using a local scratch.
        # The structure of the missing directories is always initialied during
        # the workflow setup (on solver.setup). However, the model and
        # observations are not imported during setup. Here we import the
        # initial model and observations for each solver directory. This way we
        # make sure that lost solver directories have all the neccesary files
        # to resume the workflow. The inital model is overwritten later during
        # the workflow as neccesary (i.e., see solver.run_forward_simulations).

        # NOTE: `run_adjoint_simulation` does NOT import the model. It will
        # use whatever model is on the solver directory. Therefore, if the
        # workflow is resumed during `run_adjoint_simulations` the adjoint
        # simulations will use the initial model and the result will be wrong.
        # However, `InversionFwani` never resumes on `run_adjoint_simulations`
        # so there should be no problem on this workflow. Be careful.

        if self.path_scratch_local:
            for source_name in self.solver.source_names:
                source_state_file = os.path.join(self.path.scratch,
                                                 f"sfstate_{source_name}.txt")
                # If the source state file does not exists it means the
                # workflow is starting, thus, the initial model and
                # observations will be imported later not here
                if not os.path.exists(source_state_file):
                    continue

                cwd = os.path.join(self.solver.path.scratch, source_name)
                cwd_observations = os.path.join(cwd, "traces", "obs")
                cwd_databases = os.path.join(cwd, self.solver.model_databases)

                # import initial model
                cwd_model_files = glob(os.path.join(cwd_databases,
                                                    f"*{self.solver._ext}"))
                if cwd_model_files:
                    # the model already exists, no need to import it
                    continue

                model_files = glob(os.path.join(self.solver.path.model_init,
                                                f"*{self.solver._ext}"))
                unix.cp(src=model_files, dst=cwd_databases)

                # import observations
                cwd_obs = glob(os.path.join(cwd_observations, "*"))
                if cwd_obs:
                    # observations already exists, no need to import them
                    continue

                if self.data_case == "synthetic":
                    obs = glob(os.path.join(self.path.output,
                                            source_name, "obs", "*"))
                    if not obs:
                        # if there are no observations at this point it means
                        # that they will be computed with
                        # `prepare_data_for_solver`
                        continue
                elif self.data_case == "data":
                    obs = glob(os.path.join(self.path.data,
                                            source_name, "*"))
                unix.cp(src=obs, dst=cwd_observations)

    def prepare_data_for_solver(self, move_cwd=True, **kwargs):
        """
        Determines how to provide data to each of the solvers. Either by copying
        data in from a user-provided path, or generating synthetic 'data' using
        a target model.

        .. note ::
            Must be run by system.run() so that solvers are assigned individual
            task ids and working directories
        """
        source_state = self._read_source_state_file()
        if source_state["prepare_data_for_solver"] == "completed":
            return

        logger.info(f"preparing observation data for source "
                    f"{self.solver.source_name}")

        if move_cwd and self.solver.path.scratch_local: 
            self.move_solver_cwd(dst="local")

        if self.data_case == "data":
            logger.info(f"copying data from `path_data`")
            src = glob(os.path.join(self.path.data, self.solver.source_name,
                                    "*"))
            dst = os.path.join(self.solver.cwd, "traces", "obs", "")
            unix.cp(src, dst)
        elif self.data_case == "synthetic":
            # Figure out where to export waveform files to, if requested
            if self.export_traces:
                export_traces = os.path.join(self.path.output,
                                             self.solver.source_name, "obs")
            else:
                export_traces = False

            # Run the forward solver with target model
            logger.info(f"running forward noise simulation 1 with target model "
                        f"for {self.solver.source_name}")
            self.solver.import_model(path_model=self.path.model_true)
            self.solver.forward_simulation(save_traces=False,
                                           export_traces=False,
                                           noise_simulation="1")
            # Delete forward simulation 1 traces
            unix.rm(glob(os.path.join(self.solver.cwd, "OUTPUT_FILES",
                                      self.solver.data_wildcard())))

            # Determine solver executable
            if self.solver.__class__.__name__ == "Specfem2D":
                executables = ["bin/xspecfem2D"]
            elif self.solver.__class__.__name__ in ["Specfem3D", "Specfem3DGlobe"]:
                executables = ["bin/xspecfem3D"]

            # Run forward simulation 2 and save traces to 'obs'
            obs_path = os.path.join(self.solver.cwd, "traces", "obs")
            logger.info(f"running forward noise simulation 2 with target model "
                        f"for {self.solver.source_name}")
            self.solver.forward_simulation(
                executables=executables,
                save_traces=obs_path,
                export_traces=export_traces,
                noise_simulation="2"
            )

            unix.rm(glob(os.path.join(self.solver.cwd, "OUTPUT_FILES",
                                      "noise_*.bin")))

            unix.rm(glob(os.path.join(self.solver.cwd, "OUTPUT_FILES",
                                      "lastframe_*.bin")))

            unix.rm(glob(os.path.join(self.solver.cwd, "OUTPUT_FILES",
                                      "pml_interface_*.bin")))

        if move_cwd and self.solver.path.scratch_local:
            self.move_solver_cwd(dst="project")
            unix.cd(self.solver.cwd)

        source_state["prepare_data_for_solver"] = "completed"
        self.checkpoint_source(source_state)

    def run_forward_simulations(self, path_model, move_cwd=True,
                                delete_wavefield_arrays=True, **kwargs):
        """
        Performs two forward simulations to compute noise correlations
        for a master receiver according to Tromp et al. (2010). The first
        simulation computes the `generating wavefield` and the second one
        the `correlation wavefield`.
        """
        source_state = self._read_source_state_file()
        if source_state["run_forward_simulations"] == "completed":
            return

        assert(os.path.exists(path_model)), \
            f"Model path for objective function does not exist"

        logger.info(f"evaluating objective function for source "
                    f"{self.solver.source_name}")
        logger.debug(f"running forward noise simulation 1 with "
                     f"'{self.solver.__class__.__name__}'")

        if move_cwd and self.solver.path.scratch_local:
            self.move_solver_cwd(dst="local")

        # Run forward simulation 1
        self.solver.import_model(path_model=path_model)
        self.solver.forward_simulation(save_traces=False,
                                       export_traces=False,
                                       noise_simulation="1")
        # Delete forward simulation 1 traces
        unix.rm(glob(os.path.join(self.solver.cwd, "OUTPUT_FILES",
                                  self.solver.data_wildcard())))

        # Figure out where to export waveform files to, if requested
        # path will look like: 'output/solver/001/syn/NN.SSS.BXY.semd'
        if self.export_traces:
            export_traces = os.path.join(self.path.output, "solver",
                                         self.solver.source_name, "syn")
        else:
            export_traces = False

        # Determine solver executable
        if self.solver.__class__.__name__ == "Specfem2D":
            executables = ["bin/xspecfem2D"]
        elif self.solver.__class__.__name__ in ["Specfem3D", "Specfem3DGlobe"]:
            executables = ["bin/xspecfem3D"]

        # Run forward simulation 2
        logger.debug(f"running forward noise simulation 2 with "
                     f"'{self.solver.__class__.__name__}'")

        self.solver.forward_simulation(
            executables=executables,
            save_traces=os.path.join(self.solver.cwd, "traces", "syn"),
            export_traces=export_traces,
            noise_simulation="2"
        )

        # Delete large binary files related to wavefield snapshots
        if delete_wavefield_arrays:
            unix.rm(glob(os.path.join(self.solver.cwd, "OUTPUT_FILES",
                                      "noise_*.bin")))

            unix.rm(glob(os.path.join(self.solver.cwd, "OUTPUT_FILES",
                                      "lastframe_*.bin")))

            unix.rm(glob(os.path.join(self.solver.cwd, "OUTPUT_FILES",
                                      "pml_interface_*.bin")))

        if move_cwd and self.solver.path.scratch_local:
            self.move_solver_cwd(dst="project")
            unix.cd(self.solver.cwd)

        source_state["run_forward_simulations"] = "completed"
        self.checkpoint_source(source_state)

    def move_solver_cwd(self, dst):
        project_solver_cwd = os.path.join(self.solver.path.scratch_project,
                                          self.solver.source_name)

        if dst == "local":
            self.solver.path.scratch = self.solver.path.scratch_local
            unix.rm(self.solver.cwd)  # make sure destiny does not exist
            unix.mv(src=project_solver_cwd, dst=self.solver.cwd)
        elif dst == "project":
            unix.rm(project_solver_cwd) # make sure destiny does not exist
            unix.mv(src=self.solver.cwd, dst=project_solver_cwd)
            self.solver.path.scratch = self.solver.path.scratch_project
