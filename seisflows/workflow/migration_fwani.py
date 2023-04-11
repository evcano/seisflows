#!/usr/bin/env python3
"""
A workflow similar to Migration but for ambient noise simulations
"""
import os
import sys
from glob import glob
from seisflows import logger
from seisflows.tools import msg,unix
from seisflows.workflow.forward_fwani import ForwardFwani
from seisflows.workflow.migration import Migration


class MigrationFwani(Migration, ForwardFwani):
    """
    Migration Noise Workflow
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
    __doc__ = Migration.__doc__ + __doc__

    def __init__(self, modules=None, **kwargs):
        """
        Instantiate MigrationNoise-specific parameters

        :type dummy:
        :param dummy:
        """
        super().__init__(**kwargs)

        self._modules = modules

    def run_adjoint_simulations(self):
        """
        Performs adjoint simulations for a single given event. File manipulation
        to ensure kernels are discoverable by other modules
        """
        logger.info(msg.mnr("EVALUATING EVENT KERNELS W/ ADJOINT NOISE "\
                            "SIMULATIONS"))
        self.system.run([self.run_adjoint_simulation])

    def run_adjoint_simulation(self, move_cwd=True):
        """Adjoint noise simulation function to be run by system.run()"""
        if self.export_kernels:
            export_kernels = os.path.join(self.path.output, "kernels",
                                          self.solver.source_name)
        else:
            export_kernels = False

        logger.info(f"running adjoint noise simulation for source "
                    f"{self.solver.source_name}")

        if move_cwd and self.solver.path.scratch_local:
            self.move_solver_cwd(dst="local")

        # Run adjoint simulations on system. Make kernels discoverable in
        # path `eval_grad`. Optionally export those kernels
        self.solver.adjoint_simulation(
            save_kernels=os.path.join(self.path.eval_grad, "kernels",
                                      self.solver.source_name, ""),
            export_kernels=export_kernels,
            noise_simulation=True
        )

        # Delete generating wavefield
        unix.rm(glob(os.path.join(self.solver.cwd, "OUTPUT_FILES",
                                  "noise_*.bin")))

        if move_cwd and self.solver.path.scratch_local:
            self.move_solver_cwd(dst="project")
