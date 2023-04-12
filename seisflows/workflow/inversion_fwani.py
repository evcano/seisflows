#!/usr/bin/env python3
"""
A workflow similar to Inversion but for ambient noise simulations
"""
import os
import numpy as np
from glob import glob

from seisflows import logger
from seisflows.workflow.inversion import Inversion
from seisflows.workflow.migration_fwani import MigrationFwani
from seisflows.tools import msg, unix
from seisflows.tools.model import Model


class InversionFwani(Inversion, MigrationFwani):
    """
    Inversion Noise Workflow
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
    __doc__ = Inversion.__doc__ + __doc__

    def __init__(self, modules=None, **kwargs):
        """
        Instantiate InversionNoise-specific parameters

        :type dummy:
        :param dummy:
        """
        super().__init__(**kwargs)

        self._modules = modules

    def compute_kernels(self):
        """
        Does evaluate_initial_misfit and run_adjoint_simulations
        """
        # start: inversion/evaluate_initial_misfit
        if self.iteration == 1:
            # start: forward/evaluate_initial_misfit
            logger.info(msg.mnr("COMPUTING KERNELS FOR INITIAL MODEL"))

            # Load in the initial model and check its poissons ratio
            if self.path.model_init:
                logger.info("checking initial model parameters")
                _model = Model(os.path.join(self.path.model_init),
                               parameters=self.solver._parameters,
                               regions=self.solver._regions  # 3DGLOBE only
                               )
                _model.check()

            # Load in the true model and check its poissons ratio
            if self.path.model_true:
                logger.info("checking true/target model parameters")
                _model = Model(os.path.join(self.path.model_true),
                               parameters=self.solver._parameters,
                               regions=self.solver._regions  # 3DGLOBE only
                               )
                _model.check()

            # Run steps
            run_list = [self.prepare_data_for_solver,
                        self.run_compute_kernels]

            self.system.run(run_list, path_model=self.path.model_init,
                            save_residuals=os.path.join(self.path.eval_grad,
                                                        "residuals_{src}_{it}.txt")
                            )
            # end: forward/evaluate_initial_misfit

            # Expose the initial model to the optimization library
            model = Model(self.path.model_init,
                          parameters=self.solver._parameters,
                          regions=self.solver._regions  # 3DGLOBE only
                          )
            self.optimize.save_vector(name="m_new", m=model)
        else:
            logger.info(msg.mnr("COMPUTING KERNELS FOR MODEL `m_new`"))
            # Previous line search will have saved `m_new` as the initial
            # model, export in SPECFEM format to a path discoverable by all
            # solvers
            path_model = os.path.join(self.path.eval_grad, "model")
            m_new = self.optimize.load_vector("m_new")
            m_new.write(path=path_model)

            # Run steps
            run_list = [self.run_compute_kernels]

            self.system.run(run_list, path_model=path_model,
                save_residuals=os.path.join(self.path.eval_grad,
                                            "residuals_{src}_{it}.txt")
            )

        # Rename exported synthetic traces so they are not overwritten by
        # future forward simulations
        if self.export_traces:
            unix.mv(src=os.path.join(self.path.output, "solver"),
                    dst=os.path.join(self.path.output,
                                     f"solver_{self.iteration:0>2}"))

        # Override function to sum residuals into the optimization library
        residuals_files = glob(os.path.join(self.path.eval_grad,
                            f"residuals_*_{self.iteration}.txt"))

        residuals = self.preprocess.read_residuals(residuals_files)
        total_misfit = self.preprocess.sum_residuals(residuals)
        self.optimize.save_vector(name="f_new", m=total_misfit)
        # end: inversion/evaluate_initial_misfit

    def run_compute_kernels(self, path_model, save_residuals):
        """
        """
        # move solver cwd to local scratch
        if self.solver.path.scratch_local:
            self.move_solver_cwd(dst="local")
            self.preprocess.path.solver = self.solver.path.scratch_local

        self.run_forward_simulations(path_model=path_model, move_cwd=False,
                                     keep_generating_wavefield=True)
        self.evaluate_objective_function(save_residuals=save_residuals)
        self.run_adjoint_simulation(move_cwd=False)

	# move solver cwd back to project scratch
        if self.solver.path.scratch_local:
            self.move_solver_cwd(dst="project")
            self.preprocess.path.solver = self.solver.path.scratch_project
            unix.cd(self.solver.cwd)

    @property
    def task_list(self):
        return [self.compute_kernels,
                self.postprocess_event_kernels,
                self.evaluate_gradient_from_kernels,
                self.initialize_line_search,
                self.perform_line_search,
                self.finalize_iteration
               ]
