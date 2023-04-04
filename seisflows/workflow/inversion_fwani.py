#!/usr/bin/env python3
"""
A workflow similar to Inversion but for ambient noise simulations
"""

from seisflows.workflow.inversion import Inversion
from seisflows.workflow.migration_fwani import MigrationFwani


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
