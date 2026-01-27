"""
Analyses module for whaling empirical research.

Implements R1-R17 regression specifications for estimating value creation,
organizational intermediation, risk pass-through, strategy choice, and
labor market dynamics in 19th-century offshore whaling.

Without using lay (contract share) data.
"""

from . import config
from . import data_loader
from . import connected_set
from . import baseline_production
from . import portability
from . import event_study
from . import complementarity
from . import shock_analysis
from . import strategy
from . import labor_market
from . import extensions
from . import output_generator
from . import tfp_analysis
