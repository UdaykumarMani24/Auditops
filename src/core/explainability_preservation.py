"""
Explainability Preservation Engine
Ensures ongoing model interpretability throughout ML lifecycle
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import hashlib
import json
from collections import deque

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Some explainability features will be limited.")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available. Some explainability features will be limited.")


class ExplanationMethod(Enum):
    """Supported explanation methods"""
    SHAP = "shap"
    LIME = "lime"
    FEATURE_IMPORTANCE = "feature_importance"
    COUNTERFACTUAL = "counterfactual"
    ANCHOR = "anchor"
    INTEGRATED_GRADIENTS = "integrated_gradients"


@
