"""
SANS Model Fitter - A flexible template for fitting SANS data with SasModels
"""

__version__ = '0.6.2'

from . import examples, modelfree, structure
from . import inversion as pr_inversion
from .console import adopt_logger, set_verbosity
from .data import ops as data_ops
from .fitter import IQuana, get_all_models, get_structure_factors
from .inversion import InsufficientDataError, PrEstimationError, PrResult
from .modelfree import (
    GuinierResult,
    InvariantResult,
    KratkyCurve,
    ModelFreeError,
    ModelFreeFit,
    ModelFreeResult,
    PorodResult,
    PowerLawResult,
)
from .modeling.constraints import ConstraintError, ExpressionError
from .modeling.parameters import ParameterManager
from .modeling.polydispersity import PD_DEFAULTS, PD_DISTRIBUTION_TYPES
from .multi_results import MultiFitReport, MultiFitResult
from .multifit import DatasetHandle, MultiFitter
from .results import FitResultContract, PosteriorSummary
from .series_results import SeriesResult, SeriesRow
from .structure import GenericScatteringCalculator

adopt_logger('sinccity')

__all__ = [
    'IQuana',
    'MultiFitter',
    'DatasetHandle',
    'MultiFitResult',
    'MultiFitReport',
    'SeriesResult',
    'SeriesRow',
    'ConstraintError',
    'ExpressionError',
    'ParameterManager',
    'PD_DEFAULTS',
    'PD_DISTRIBUTION_TYPES',
    'get_all_models',
    'set_verbosity',
    'get_structure_factors',
    'FitResultContract',
    'data_ops',
    'examples',
    'PosteriorSummary',
    'pr_inversion',
    'PrResult',
    'InsufficientDataError',
    'PrEstimationError',
    'modelfree',
    'ModelFreeResult',
    'ModelFreeFit',
    'GuinierResult',
    'PowerLawResult',
    'PorodResult',
    'InvariantResult',
    'KratkyCurve',
    'ModelFreeError',
    'GenericScatteringCalculator',
    'structure',
]
