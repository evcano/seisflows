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
    :type dummy:
    :param dummy:
    ***
    """
    __doc__ = Forward.__doc__ + __doc__

    def __init__(self, modules=None, **kwargs):
        """
        Instantiate ForwardNoise-specific parameters

        :type dummy:
        :param dummy:
        """

        super().__init__(**kwargs)

        self._modules = modules

    def prepare_data_for_solver(self, **kwargs):
        """
        Determines how to provide data to each of the solvers. Either by copying
        data in from a user-provided path, or generating synthetic 'data' using
        a target model.

        .. note ::
            Must be run by system.run() so that solvers are assigned individual
            task ids and working directories
        """
        logger.info(f"preparing observation data for source "
                    f"{self.solver.source_name}")

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

    def run_forward_simulations(self, path_model, **kwargs):
        """
        Performs two forward simulations to compute noise correlations
        for a master receiver according to Tromp et al. (2010). The first
        simulation computes the `generating wavefield` and the second one
        the `correlation wavefield`.
        """
        assert(os.path.exists(path_model)), \
            f"Model path for objective function does not exist"

        logger.info(f"evaluating objective function for source "
                    f"{self.solver.source_name}")
        logger.debug(f"running forward noise simulation 1 with "
                     f"'{self.solver.__class__.__name__}'")

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

        # TODO: Can we delete the generating wavefield files if we invert
        # only the structure?
