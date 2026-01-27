"""
Analyses module for whaling empirical research.

Implements R1-R17 regression specifications for estimating value creation,
organizational intermediation, risk pass-through, strategy choice, and
labor market dynamics in 19th-century offshore whaling.

Without using lay (contract share) data.
"""

from . import config
from . import data_loader
from . import baseline_production
from . import portability
from . import search_theory
from . import complementarity
from . import strategy
from . import risk_matching
from . import tfp_analysis
from . import output_generator
from . import counterfactual_simulations
from . import counterfactual_robustness
from . import counterfactual_suite
