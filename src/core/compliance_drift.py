"""
Compliance Drift Detection Mechanism for AuditOps Framework
Implements real-time statistical monitoring for regulatory compliance deviations
Author: Udayakumar Mani, Senthilkumar Rathinasamy
Organization: SASTRA Deemed University
License: MIT
"""

import numpy as np
import warnings
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum, auto
from collections import deque, defaultdict
from datetime import datetime, timedelta
from scipy import stats
import pandas as pd
from pathlib import Path


class DriftSeverity(Enum):
    """Severity levels for detected compliance drift."""
    NONE = (0, "NONE", "No drift detected")
    INFO = (1, "INFO", "Informational drift within acceptable bounds")
    WARNING = (2, "WARNING", "Minor deviation requiring monitoring")
    MODERATE = (3, "MODERATE", "Significant deviation requiring investigation")
    CRITICAL = (4, "CRITICAL", "Severe deviation requiring immediate action")
    VIOLATION = (5, "VIOLATION", "Regulatory violation detected - immediate remediation required")
    
    def __init__(self, level: int, code: str, description: str):
        self.level = level
        self.code = code
        self.description = description
    
    def __lt__(self, other):
        return self.level < other.level
    
    def __le__(self, other):
        return self.level <= other.level
    
    def __gt__(self, other):
        return self.level > other.level
    
    def __ge__(self, other):
        return self.level >= other.level


class AlertCode(Enum):
    """Comprehensive alert codes for different drift types."""
    # Performance drift codes (0xP series)
    PERFORMANCE_DECAY = (0xP100, "Performance metric below threshold")
    ACCURACY_DROP = (0xP101, "Model accuracy degradation")
    RECALL_DECLINE = (0xP102, "Model recall/sensitivity degradation")
    PRECISION_DECLINE = (0xP103, "Model precision degradation")
    AUC_DECAY = (0xP104, "AUC-ROC performance degradation")
    F1_SCORE_DECAY = (0xP105, "F1-score performance degradation")
    LATENCY_INCREASE = (0xP106, "Inference latency increase")
    THROUGHPUT_DECLINE = (0xP107, "Throughput/system capacity decline")
    
    # Fairness drift codes (0xF series)
    FAIRNESS_DRIFT = (0xF200, "Fairness metric deviation")
    DISPARATE_IMPACT_INCREASE = (0xF201, "Disparate impact ratio increase")
    EQUAL_OPPORTUNITY_DRIFT = (0xF202, "Equal opportunity difference increase")
    DEMOGRAPHIC_PARITY_DRIFT = (0xF203, "Demographic parity difference increase")
    PREDICTIVE_PARITY_DRIFT = (0xF204, "Predictive parity difference increase")
    TREATMENT_EQUALITY_DRIFT = (0xF205, "Treatment equality difference increase")
    
    # Data drift codes (0xD series)
    DATA_DISTRIBUTION_SHIFT = (0xD300, "Input data distribution shift")
    COVARIATE_SHIFT = (0xD301, "Covariate shift detected")
    CONCEPT_DRIFT = (0xD302, "Concept drift detected")
    DATA_QUALITY_DECAY = (0xD303, "Data quality metrics degradation")
    FEATURE_DISTRIBUTION_CHANGE = (0xD304, "Feature distribution change")
    OUTLIER_INCREASE = (0xD305, "Unusual increase in outliers")
    MISSING_VALUE_INCREASE = (0xD306, "Increase in missing values")
    
    # Regulatory drift codes (0xR series)
    REGULATORY_THRESHOLD_VIOLATION = (0xR400, "Regulatory performance threshold violation")
    EXPLAINABILITY_DECAY = (0xR401, "Model explainability degradation")
    EXPLANATION_STABILITY_LOSS = (0xR402, "Explanation stability below threshold")
    AUDIT_TRAIL_GAP = (0xR403, "Audit trail completeness gap")
    COMPLIANCE_CHECK_FAILURE = (0xR404, "Automated compliance check failure")
    DOCUMENTATION_INCOMPLETE = (0xR405, "Required documentation incomplete")
    HUMAN_OVERSIGHT_MISSING = (0xR406, "Required human oversight missing")
    
    # System drift codes (0xS series)
    RESOURCE_USAGE_SPIKE = (0xS500, "Resource usage spike detected")
    DEPENDENCY_VERSION_DRIFT = (0xS501, "Dependency version drift")
    ENVIRONMENT_CONFIG_DRIFT = (0xS502, "Environment configuration drift")
    SECURITY_COMPLIANCE_DRIFT = (0xS503, "Security compliance deviation")
    PRIVACY_COMPLIANCE_DRIFT = (0xS504, "Privacy compliance deviation")
    
    def __init__(self, code: int, description: str):
        self.code = code
        self.description = description
    
    def __str__(self):
        return f"{self.name} (0x{self.code:X})"


@dataclass
class DriftAlert:
    """
    Comprehensive container for compliance drift alerts.
    
    Attributes:
        alert_id (str): Unique identifier for the alert
        code (AlertCode): Type of drift detected
        severity (DriftSeverity): Severity level
        message (str): Human-readable description
        timestamp (datetime): When the alert was generated
        metric (str): Metric that triggered the alert
        current_value (float): Current metric value
        expected_value (float): Expected/baseline value
        threshold (float): Regulatory or operational threshold
        confidence (float): Statistical confidence in detection (0-1)
        metadata (Dict[str, Any]): Additional context information
        recommendations (List[str]): Actionable recommendations
        acknowledged (bool): Whether alert has been acknowledged
        auto_remediated (bool): Whether auto-remediation was attempted
    """
    alert_id: str
    code: AlertCode
    severity: DriftSeverity
    message: str
    timestamp: datetime
    metric: str
    current_value: float
    expected_value: float
    threshold: float
    confidence: float = 0.95
    metadata: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    acknowledged: bool = False
    auto_remediated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "code": str(self.code),
            "severity": self.severity.code,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metric": self.metric,
            "current_value": self.current_value,
            "expected_value": self.expected_value,
            "threshold": self.threshold,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "recommendations": self.recommendations,
            "acknowledged": self.acknowledged,
            "auto_remediated": self.auto_remediated
        }
    
    def __str__(self) -> str:
        return (f"[{self.severity.code}] {self.metric}: {self.current_value:.3f} "
                f"(expected: {self.expected_value:.3f}, threshold: {self.threshold:.3f})")


@dataclass
class RegulatoryThreshold:
    """
    Definition of regulatory threshold for compliance monitoring.
    
    Attributes:
        metric_name (str): Name of the metric to monitor
        requirement_id (str): Regulatory requirement identifier
        min_value (Optional[float]): Minimum acceptable value
        max_value (Optional[float]): Maximum acceptable value
        target_value (Optional[float]): Target/optimal value
        window_size (int): Samples to consider for statistical detection
        severity_mapping (Dict[float, DriftSeverity]): Deviation to severity mapping
        weight (float): Importance weight for compliance scoring
        detection_method (str): Statistical method for drift detection
        grace_period_hours (int): Grace period before enforcement
    """
    metric_name: str
    requirement_id: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    target_value: Optional[float] = None
    window_size: int = 100
    severity_mapping: Dict[float, DriftSeverity] = field(default_factory=dict)
    weight: float = 1.0
    detection_method: str = "page_hinkley"
    grace_period_hours: int = 24
    
    def __post_init__(self):
        """Validate threshold configuration."""
        if self.min_value is None and self.max_value is None and self.target_value is None:
            raise ValueError("At least one of min_value, max_value, or target_value must be specified")
        
        if self.window_size < 10:
            warnings.warn(f"Window size {self.window_size} is small for reliable drift detection")
        
        # Set default severity mapping if not provided
        if not self.severity_mapping:
            if self.min_value is not None:
                deviations = [0.05, 0.10, 0.15, 0.20]
                for dev in deviations:
                    self.severity_mapping[self.min_value - dev] = {
                        0.05: DriftSeverity.WARNING,
                        0.10: DriftSeverity.MODERATE,
                        0.15: DriftSeverity.CRITICAL,
                        0.20: DriftSeverity.VIOLATION
                    }[dev]
            elif self.max_value is not None:
                deviations = [0.05, 0.10, 0.15, 0.20]
                for dev in deviations:
                    self.severity_mapping[self.max_value + dev] = {
                        0.05: DriftSeverity.WARNING,
                        0.10: DriftSeverity.MODERATE,
                        0.15: DriftSeverity.CRITICAL,
                        0.20: DriftSeverity.VIOLATION
                    }[dev]


class PageHinkleyDetector:
    """
    Page-Hinkley test for detecting changes in running processes.
    
    Implementation based on: Page, E. S. (1954). Continuous inspection schemes.
    Biometrika, 41(1/2), 100-115.
    
    Attributes:
        delta (float): Magnitude of changes to detect
        threshold (float): Detection threshold
        alpha (float): Forgetting factor (0 < alpha <= 1)
        cumulative_sum (float): Running cumulative sum
        min_cumulative_sum (float): Minimum cumulative sum observed
        sample_count (int): Number of samples processed
        mean (float): Exponential weighted mean
        drift_detected (bool): Whether drift was detected in last update
    """
    
    def __init__(self, 
                 delta: float = 0.005, 
                 threshold: float = 50.0, 
                 alpha: float = 1.0,
                 min_samples: int = 30):
        """
        Initialize Page-Hinkley detector.
        
        Args:
            delta: Magnitude of changes to detect (smaller = more sensitive)
            threshold: Detection threshold (higher = less sensitive)
            alpha: Forgetting factor (0 < alpha <= 1, 1 = no forgetting)
            min_samples: Minimum samples before detection is reliable
        """
        if delta <= 0:
            raise ValueError("delta must be positive")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.min_samples = min_samples
        self.reset()
    
    def reset(self):
        """Reset detector state."""
        self.cumulative_sum = 0.0
        self.min_cumulative_sum = float('inf')
        self.sample_count = 0
        self.mean = 0.0
        self.drift_detected = False
        self.drift_magnitude = 0.0
        self.ph_statistic = 0.0
        self.history = deque(maxlen=1000)
    
    def update(self, value: float) -> Tuple[bool, float, float]:
        """
        Update detector with new observation.
        
        Args:
            value: New observation value
            
        Returns:
            Tuple containing:
                - drift_detected (bool): Whether drift was detected
                - ph_statistic (float): Page-Hinkley test statistic
                - drift_magnitude (float): Magnitude of drift if detected
        """
        self.sample_count += 1
        self.history.append((datetime.now(), value))
        
        # Update exponential weighted mean
        if self.sample_count == 1:
            self.mean = value
        else:
            self.mean = self.alpha * value + (1 - self.alpha) * self.mean
        
        # Update cumulative sum
        deviation = value - self.mean - self.delta
        self.cumulative_sum += deviation
        
        # Update minimum cumulative sum
        if self.cumulative_sum < self.min_cumulative_sum:
            self.min_cumulative_sum = self.cumulative_sum
        
        # Calculate test statistic
        self.ph_statistic = self.cumulative_sum - self.min_cumulative_sum
        
        # Check for drift (only if we have enough samples)
        self.drift_detected = False
        self.drift_magnitude = 0.0
        
        if self.sample_count >= self.min_samples:
            self.drift_detected = self.ph_statistic > self.threshold
            
            if self.drift_detected:
                # Estimate drift magnitude
                recent_values = [v for _, v in list(self.history)[-self.min_samples:]]
                if len(recent_values) >= 10:
                    first_half = recent_values[:len(recent_values)//2]
                    second_half = recent_values[len(recent_values)//2:]
                    if first_half and second_half:
                        self.drift_magnitude = np.mean(second_half) - np.mean(first_half)
        
        return self.drift_detected, self.ph_statistic, self.drift_magnitude
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current detector statistics."""
        return {
            "sample_count": self.sample_count,
            "current_mean": self.mean,
            "cumulative_sum": self.cumulative_sum,
            "ph_statistic": self.ph_statistic,
            "drift_detected": self.drift_detected,
            "drift_magnitude": self.drift_magnitude,
            "detection_threshold": self.threshold,
            "sensitivity_delta": self.delta,
            "forgetting_factor": self.alpha
        }


class EWMAChangeDetector:
    """
    Exponentially Weighted Moving Average change detector.
    
    Detects changes based on deviations from EWMA with adaptive thresholds.
    """
    
    def __init__(self, 
                 alpha: float = 0.3,
                 threshold_sigma: float = 3.0,
                 min_samples: int = 30):
        """
        Initialize EWMA change detector.
        
        Args:
            alpha: Smoothing factor (0 < alpha < 1)
            threshold_sigma: Number of standard deviations for threshold
            min_samples: Minimum samples before detection is reliable
        """
        self.alpha = alpha
        self.threshold_sigma = threshold_sigma
        self.min_samples = min_samples
        self.reset()
    
    def reset(self):
        """Reset detector state."""
        self.ewma = None
        self.ewmv = None  # Exponentially weighted moving variance
        self.sample_count = 0
        self.history = deque(maxlen=1000)
        self.change_detected = False
        self.change_magnitude = 0.0
    
    def update(self, value: float) -> Tuple[bool, float]:
        """
        Update detector with new observation.
        
        Args:
            value: New observation value
            
        Returns:
            Tuple containing:
                - change_detected (bool): Whether change was detected
                - z_score (float): Standardized deviation from EWMA
        """
        self.sample_count += 1
        self.history.append((datetime.now(), value))
        
        # Initialize if first sample
        if self.ewma is None:
            self.ewma = value
            self.ewmv = 0.0
            return False, 0.0
        
        # Calculate deviation
        deviation = value - self.ewma
        z_score = 0.0
        
        if self.ewmv > 0:
            z_score = deviation / np.sqrt(self.ewmv)
        
        # Check for change
        self.change_detected = False
        self.change_magnitude = 0.0
        
        if self.sample_count >= self.min_samples and abs(z_score) > self.threshold_sigma:
            self.change_detected = True
            self.change_magnitude = deviation
        
        # Update EWMA and EWMV
        self.ewma = self.alpha * value + (1 - self.alpha) * self.ewma
        squared_deviation = deviation ** 2
        self.ewmv = self.alpha * squared_deviation + (1 - self.alpha) * self.ewmv
        
        return self.change_detected, z_score
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current detector statistics."""
        return {
            "sample_count": self.sample_count,
            "ewma": self.ewma,
            "ewmv": self.ewmv,
            "change_detected": self.change_detected,
            "change_magnitude": self.change_magnitude,
            "threshold_sigma": self.threshold_sigma,
            "smoothing_factor": self.alpha
        }


class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) control chart for change detection.
    
    Implementation based on: Hawkins, D. M., & Olwell, D. H. (1998).
    Cumulative sum charts and charting for quality improvement.
    """
    
    def __init__(self, 
                 k: float = 0.5,
                 h: float = 5.0,
                 min_samples: int = 30):
        """
        Initialize CUSUM detector.
        
        Args:
            k: Reference value (allowable deviation)
            h: Decision interval (detection threshold)
            min_samples: Minimum samples before detection is reliable
        """
        self.k = k
        self.h = h
        self.min_samples = min_samples
        self.reset()
    
    def reset(self):
        """Reset detector state."""
        self.cusum_positive = 0.0
        self.cusum_negative = 0.0
        self.sample_count = 0
        self.mean = 0.0
        self.std = 1.0
        self.history = deque(maxlen=1000)
        self.change_detected = False
        self.change_direction = 0  # -1: decrease, 0: none, 1: increase
    
    def update(self, value: float) -> Tuple[bool, int]:
        """
        Update detector with new observation.
        
        Args:
            value: New observation value
            
        Returns:
            Tuple containing:
                - change_detected (bool): Whether change was detected
                - change_direction (int): Direction of change (-1, 0, 1)
        """
        self.sample_count += 1
        self.history.append((datetime.now(), value))
        
        # Update statistics (simple running stats)
        if self.sample_count == 1:
            self.mean = value
            self.std = 0.0
        else:
            old_mean = self.mean
            self.mean = old_mean + (value - old_mean) / self.sample_count
            self.std = self.std + (value - old_mean) * (value - self.mean)
            if self.sample_count > 1:
                self.std = np.sqrt(self.std / (self.sample_count - 1))
        
        # Standardize value if we have enough samples
        if self.sample_count >= 10 and self.std > 0:
            standardized = (value - self.mean) / self.std
        else:
            standardized = value - self.mean
        
        # Update CUSUM statistics
        self.cusum_positive = max(0, self.cusum_positive + standardized - self.k)
        self.cusum_negative = max(0, self.cusum_negative - standardized - self.k)
        
        # Check for change
        self.change_detected = False
        self.change_direction = 0
        
        if self.sample_count >= self.min_samples:
            if self.cusum_positive > self.h:
                self.change_detected = True
                self.change_direction = 1  # Increase
            elif self.cusum_negative > self.h:
                self.change_detected = True
                self.change_direction = -1  # Decrease
        
        return self.change_detected, self.change_direction
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current detector statistics."""
        return {
            "sample_count": self.sample_count,
            "mean": self.mean,
            "std": self.std,
            "cusum_positive": self.cusum_positive,
            "cusum_negative": self.cusum_negative,
            "change_detected": self.change_detected,
            "change_direction": self.change_direction,
            "reference_value_k": self.k,
            "decision_interval_h": self.h
        }


class StatisticalProcessControl:
    """
    Statistical Process Control (SPC) for compliance monitoring.
    
    Implements control charts and statistical tests for process stability.
    """
    
    def __init__(self, 
                 control_limit_sigma: float = 3.0,
                 rules: List[str] = None):
        """
        Initialize SPC monitor.
        
        Args:
            control_limit_sigma: Sigma multiplier for control limits
            rules: List of Western Electric rules to apply
        """
        self.control_limit_sigma = control_limit_sigma
        self.rules = rules or ["1", "2", "3", "4", "5", "6", "7", "8"]
        self.reset()
    
    def reset(self):
        """Reset SPC state."""
        self.sample_count = 0
        self.values = deque(maxlen=1000)
        self.mean = 0.0
        self.std = 0.0
        self.ucl = None  # Upper Control Limit
        self.lcl = None  # Lower Control Limit
        self.violations = []
        self.history = deque(maxlen=1000)
    
    def update(self, value: float) -> List[Dict[str, Any]]:
        """
        Update SPC with new observation and check for violations.
        
        Args:
            value: New observation value
            
        Returns:
            List of rule violations detected
        """
        self.sample_count += 1
        self.values.append(value)
        self.history.append((datetime.now(), value))
        
        # Update statistics
        if self.sample_count == 1:
            self.mean = value
            self.std = 0.0
        else:
            old_mean = self.mean
            self.mean = old_mean + (value - old_mean) / self.sample_count
            self.std = self.std + (value - old_mean) * (value - self.mean)
            if self.sample_count > 1:
                self.std = np.sqrt(self.std / (self.sample_count - 1))
        
        # Calculate control limits if we have enough samples
        if self.sample_count >= 20 and self.std > 0:
            self.ucl = self.mean + self.control_limit_sigma * self.std
            self.lcl = self.mean - self.control_limit_sigma * self.std
            
            # Check for rule violations
            self.violations = self._check_western_electric_rules(value)
        else:
            self.violations = []
        
        return self.violations
    
    def _check_western_electric_rules(self, value: float) -> List[Dict[str, Any]]:
        """Check for Western Electric rule violations."""
        violations = []
        
        if self.ucl is None or self.lcl is None:
            return violations
        
        # Rule 1: Point outside 3-sigma control limits
        if value > self.ucl or value < self.lcl:
            violations.append({
                "rule": "1",
                "description": "Point outside 3-sigma control limits",
                "value": value,
                "limit": self.ucl if value > self.ucl else self.lcl,
                "severity": DriftSeverity.CRITICAL
            })
        
        # Rule 2: 2 of 3 consecutive points beyond 2-sigma
        if len(self.values) >= 3:
            recent = list(self.values)[-3:]
            sigma_2_upper = self.mean + 2 * self.std
            sigma_2_lower = self.mean - 2 * self.std
            
            beyond_2_sigma = sum(1 for v in recent if v > sigma_2_upper or v < sigma_2_lower)
            if beyond_2_sigma >= 2:
                violations.append({
                    "rule": "2",
                    "description": "2 of 3 consecutive points beyond 2-sigma limits",
                    "values": recent,
                    "severity": DriftSeverity.MODERATE
                })
        
        # Rule 3: 4 of 5 consecutive points beyond 1-sigma
        if len(self.values) >= 5:
            recent = list(self.values)[-5:]
            sigma_1_upper = self.mean + 1 * self.std
            sigma_1_lower = self.mean - 1 * self.std
            
            beyond_1_sigma = sum(1 for v in recent if v > sigma_1_upper or v < sigma_1_lower)
            if beyond_1_sigma >= 4:
                violations.append({
                    "rule": "3",
                    "description": "4 of 5 consecutive points beyond 1-sigma limits",
                    "values": recent,
                    "severity": DriftSeverity.WARNING
                })
        
        # Rule 4: 8 consecutive points on same side of center line
        if len(self.values) >= 8:
            recent = list(self.values)[-8:]
            if all(v > self.mean for v in recent) or all(v < self.mean for v in recent):
                violations.append({
                    "rule": "4",
                    "description": "8 consecutive points on same side of center line",
                    "values": recent,
                    "severity": DriftSeverity.WARNING
                })
        
        return violations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current SPC statistics."""
        return {
            "sample_count": self.sample_count,
            "mean": self.mean,
            "std": self.std,
            "ucl": self.ucl,
            "lcl": self.lcl,
            "current_violations": len(self.violations),
            "control_limit_sigma": self.control_limit_sigma,
            "active_rules": self.rules
        }


class ComplianceDriftDetector:
    """
    Main compliance drift detection system for AuditOps framework.
    
    Implements comprehensive real-time monitoring for regulatory compliance
    deviations using multiple statistical detection methods.
    
    Attributes:
        regulatory_context (str): Regulatory framework being monitored
        thresholds (Dict[str, RegulatoryThreshold]): Configured thresholds
        detectors (Dict[str, Any]): Statistical detectors per metric
        alerts (deque): Recent alerts storage
        metrics_history (Dict[str, deque]): Historical metric values
        performance_stats (Dict[str, Any]): Detection performance statistics
        alert_callbacks (List[Callable]): Functions to call when alerts generated
    """
    
    def __init__(self, 
                 regulatory_context: str = "EU_AI_ACT_HIGH_RISK",
                 config_path: Optional[Union[str, Path]] = None,
                 alert_callback: Optional[Callable[[DriftAlert], None]] = None,
                 storage_path: Optional[Union[str, Path]] = None,
                 max_alerts: int = 1000):
        """
        Initialize compliance drift detector.
        
        Args:
            regulatory_context: Regulatory framework to monitor
            config_path: Path to configuration file (optional)
            alert_callback: Function to call when alert is generated
            storage_path: Path for alert storage (optional)
            max_alerts: Maximum number of alerts to keep in memory
            
        Raises:
            ValueError: If regulatory context not supported
        """
        self.regulatory_context = regulatory_context
        self.max_alerts = max_alerts
        
        # Load regulatory thresholds
        self.thresholds = self._load_regulatory_thresholds(regulatory_context, config_path)
        
        # Initialize detectors
        self.detectors = self._initialize_detectors()
        
        # Initialize alert system
        self.alerts = deque(maxlen=max_alerts)
        self.alert_callbacks = []
        if alert_callback:
            self.alert_callbacks.append(alert_callback)
        
        # Initialize metrics history
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Initialize performance statistics
        self.performance_stats = {
            "total_checks": 0,
            "drifts_detected": 0,
            "false_positives": 0,
            "true_positives": 0,
            "false_negatives": 0,
            "response_time_ms": deque(maxlen=100),
            "start_time": datetime.now().isoformat(),
            "last_check": None,
            "regulatory_context": regulatory_context
        }
        
        # Initialize error handling
        self.error_codes = {
            0xE100: ("Non-critical", "Explanation partial failure", DriftSeverity.INFO),
            0xW200: ("Warning", "Drift alert, degradation mode", DriftSeverity.WARNING),
            0xC300: ("Critical", "Mandatory constraint violation", DriftSeverity.CRITICAL)
        }
        
        # Initialize storage if provided
        self.storage_path = Path(storage_path) if storage_path else None
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
        
        print(f"ComplianceDriftDetector initialized for {regulatory_context}")
        print(f"Monitoring {len(self.thresholds)} regulatory thresholds")
    
    def _load_regulatory_thresholds(self, 
                                   context: str, 
                                   config_path: Optional[Union[str, Path]] = None) -> Dict[str, RegulatoryThreshold]:
        """
        Load regulatory thresholds from configuration.
        
        Args:
            context: Regulatory context name
            config_path: Optional custom config path
            
        Returns:
            Dictionary of threshold configurations
            
        Raises:
            ValueError: If context not supported and no config provided
        """
        thresholds = {}
        
        # Try to load from config file first
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                return self._parse_thresholds_from_config(config)
            else:
                warnings.warn(f"Config file not found: {config_path}")
        
        # Use built-in configurations
        builtin_configs = {
            "EU_AI_ACT_HIGH_RISK": self._get_eu_ai_act_thresholds(),
            "FDA_21_CFR_820": self._get_fda_thresholds(),
            "REGULATION_B": self._get_regulation_b_thresholds(),
            "HIPAA_COMPLIANCE": self._get_hipaa_thresholds(),
            "GDPR_COMPLIANCE": self._get_gdpr_thresholds()
        }
        
        if context in builtin_configs:
            return builtin_configs[context]
        else:
            raise ValueError(f"Unsupported regulatory context: {context}. "
                           f"Supported: {list(builtin_configs.keys())}")
    
    def _get_eu_ai_act_thresholds(self) -> Dict[str, RegulatoryThreshold]:
        """Get EU AI Act thresholds for high-risk systems."""
        return {
            "accuracy": RegulatoryThreshold(
                metric_name="accuracy",
                requirement_id="EU_AI_ACT_ART_15",
                min_value=0.70,
                window_size=100,
                severity_mapping={
                    0.65: DriftSeverity.WARNING,
                    0.60: DriftSeverity.MODERATE,
                    0.55: DriftSeverity.CRITICAL,
                    0.50: DriftSeverity.VIOLATION
                },
                weight=1.0,
                detection_method="page_hinkley",
                grace_period_hours=24
            ),
            "explainability_stability": RegulatoryThreshold(
                metric_name="explainability_stability",
                requirement_id="EU_AI_ACT_ART_13",
                min_value=0.80,
                window_size=50,
                severity_mapping={
                    0.75: DriftSeverity.WARNING,
                    0.70: DriftSeverity.MODERATE,
                    0.65: DriftSeverity.CRITICAL,
                    0.60: DriftSeverity.VIOLATION
                },
                weight=0.9,
                detection_method="ewma",
                grace_period_hours=48
            ),
            "fairness_disparity": RegulatoryThreshold(
                metric_name="fairness_disparity",
                requirement_id="EU_AI_ACT_ART_10",
                max_value=0.20,
                window_size=200,
                severity_mapping={
                    0.25: DriftSeverity.WARNING,
                    0.30: DriftSeverity.MODERATE,
                    0.35: DriftSeverity.CRITICAL,
                    0.40: DriftSeverity.VIOLATION
                },
                weight=1.0,
                detection_method="cusum",
                grace_period_hours=72
            ),
            "inference_latency": RegulatoryThreshold(
                metric_name="inference_latency_ms",
                requirement_id="EU_AI_ACT_ART_15_PERF",
                max_value=1000.0,
                window_size=50,
                severity_mapping={
                    1500.0: DriftSeverity.WARNING,
                    2000.0: DriftSeverity.MODERATE,
                    3000.0: DriftSeverity.CRITICAL,
                    5000.0: DriftSeverity.VIOLATION
                },
                weight=0.7,
                detection_method="page_hinkley",
                grace_period_hours=12
            ),
            "data_quality_score": RegulatoryThreshold(
                metric_name="data_quality_score",
                requirement_id="EU_AI_ACT_ART_10_DATA",
                min_value=0.85,
                window_size=100,
                severity_mapping={
                    0.80: DriftSeverity.WARNING,
                    0.75: DriftSeverity.MODERATE,
                    0.70: DriftSeverity.CRITICAL,
                    0.65: DriftSeverity.VIOLATION
                },
                weight=0.8,
                detection_method="spc",
                grace_period_hours=96
            )
        }
    
    def _get_fda_thresholds(self) -> Dict[str, RegulatoryThreshold]:
        """Get FDA 21 CFR Part 820 thresholds."""
        return {
            "sensitivity": RegulatoryThreshold(
                metric_name="sensitivity",
                requirement_id="FDA_820_75_A",
                min_value=0.80,
                window_size=100,
                severity_mapping={
                    0.75: DriftSeverity.WARNING,
                    0.70: DriftSeverity.MODERATE,
                    0.65: DriftSeverity.CRITICAL,
                    0.60: DriftSeverity.VIOLATION
                },
                weight=1.0,
                detection_method="page_hinkley"
            ),
            "specificity": RegulatoryThreshold(
                metric_name="specificity",
                requirement_id="FDA_820_75_B",
                min_value=0.80,
                window_size=100,
                severity_mapping={
                    0.75: DriftSeverity.WARNING,
                    0.70: DriftSeverity.MODERATE,
                    0.65: DriftSeverity.CRITICAL,
                    0.60: DriftSeverity.VIOLATION
                },
                weight=1.0,
                detection_method="page_hinkley"
            ),
            "precision": RegulatoryThreshold(
                metric_name="precision",
                requirement_id="FDA_820_75_C",
                min_value=0.75,
                window_size=100,
                weight=0.9,
                detection_method="page_hinkley"
            ),
            "predictive_value_positive": RegulatoryThreshold(
                metric_name="predictive_value_positive",
                requirement_id="FDA_820_75_D",
                min_value=0.70,
                window_size=100,
                weight=0.8,
                detection_method="ewma"
            )
        }
    
    def _get_regulation_b_thresholds(self) -> Dict[str, RegulatoryThreshold]:
        """Get Regulation B (ECOA) thresholds for fair lending."""
        return {
            "disparate_impact_ratio": RegulatoryThreshold(
                metric_name="disparate_impact_ratio",
                requirement_id="REG_B_FAIR_LENDING",
                min_value=0.80,
                max_value=1.25,
                window_size=200,
                severity_mapping={
                    0.75: DriftSeverity.WARNING,
                    0.70: DriftSeverity.MODERATE,
                    0.65: DriftSeverity.CRITICAL,
                    0.60: DriftSeverity.VIOLATION
                },
                weight=1.0,
                detection_method="cusum"
            ),
            "equal_opportunity_difference": RegulatoryThreshold(
                metric_name="equal_opportunity_difference",
                requirement_id="REG_B_EQUAL_OPPORTUNITY",
                max_value=0.10,
                window_size=200,
                severity_mapping={
                    0.15: DriftSeverity.WARNING,
                    0.20: DriftSeverity.MODERATE,
                    0.25: DriftSeverity.CRITICAL,
                    0.30: DriftSeverity.VIOLATION
                },
                weight=0.9,
                detection_method="spc"
            ),
            "demographic_parity_difference": RegulatoryThreshold(
                metric_name="demographic_parity_difference",
                requirement_id="REG_B_DEMOGRAPHIC_PARITY",
                max_value=0.10,
                window_size=200,
                weight=0.8,
                detection_method="ewma"
            )
        }
    
    def _get_hipaa_thresholds(self) -> Dict[str, RegulatoryThreshold]:
        """Get HIPAA compliance thresholds."""
        return {
            "privacy_violation_rate": RegulatoryThreshold(
                metric_name="privacy_violation_rate",
                requirement_id="HIPAA_164_312",
                max_value=0.001,  # 0.1% maximum violation rate
                window_size=1000,
                severity_mapping={
                    0.005: DriftSeverity.WARNING,
                    0.01: DriftSeverity.MODERATE,
                    0.02: DriftSeverity.CRITICAL,
                    0.05: DriftSeverity.VIOLATION
                },
                weight=1.0,
                detection_method="page_hinkley"
            ),
            "access_control_violations": RegulatoryThreshold(
                metric_name="access_control_violations",
                requirement_id="HIPAA_164_308",
                max_value=0.0,  # Zero tolerance
                window_size=100,
                weight=1.0,
                detection_method="spc"
            )
        }
    
    def _get_gdpr_thresholds(self) -> Dict[str, RegulatoryThreshold]:
        """Get GDPR compliance thresholds."""
        return {
            "data_breach_incidents": RegulatoryThreshold(
                metric_name="data_breach_incidents",
                requirement_id="GDPR_ART_33",
                max_value=0.0,  # Zero tolerance
                window_size=100,
                weight=1.0,
                detection_method="spc"
            ),
            "right_to_erasure_compliance": RegulatoryThreshold(
                metric_name="right_to_erasure_compliance",
                requirement_id="GDPR_ART_17",
                min_value=0.95,  # 95% compliance rate
                window_size=50,
                weight=0.9,
                detection_method="ewma"
            )
        }
    
    def _parse_thresholds_from_config(self, config: Dict[str, Any]) -> Dict[str, RegulatoryThreshold]:
        """Parse thresholds from configuration dictionary."""
        thresholds = {}
        
        for metric_name, threshold_config in config.get("thresholds", {}).items():
            # Map string severity to DriftSeverity enum
            severity_mapping = {}
            if "severity_mapping" in threshold_config:
                for deviation_str, severity_str in threshold_config["severity_mapping"].items():
                    deviation = float(deviation_str)
                    severity = DriftSeverity[severity_str]
                    severity_mapping[deviation] = severity
            
            threshold = RegulatoryThreshold(
                metric_name=metric_name,
                requirement_id=threshold_config.get("requirement_id", "UNKNOWN"),
                min_value=threshold_config.get("min_value"),
                max_value=threshold_config.get("max_value"),
                target_value=threshold_config.get("target_value"),
                window_size=threshold_config.get("window_size", 100),
                severity_mapping=severity_mapping,
                weight=threshold_config.get("weight", 1.0),
                detection_method=threshold_config.get("detection_method", "page_hinkley"),
                grace_period_hours=threshold_config.get("grace_period_hours", 24)
            )
            
            thresholds[metric_name] = threshold
        
        return thresholds
    
    def _initialize_detectors(self) -> Dict[str, Any]:
        """Initialize statistical detectors for each metric."""
        detectors = {}
        
        for metric_name, threshold in self.thresholds.items():
            detector_config = {
                "window_size": threshold.window_size,
                "method": threshold.detection_method
            }
            
            if threshold.detection_method == "page_hinkley":
                detectors[metric_name] = {
                    "detector": PageHinkleyDetector(
                        delta=0.005,
                        threshold=50.0,
                        alpha=0.99,
                        min_samples=max(20, threshold.window_size // 5)
                    ),
                    "config": detector_config
                }
            elif threshold.detection_method == "ewma":
                detectors[metric_name] = {
                    "detector": EWMAChangeDetector(
                        alpha=0.3,
                        threshold_sigma=3.0,
                        min_samples=max(20, threshold.window_size // 5)
                    ),
                    "config": detector_config
                }
            elif threshold.detection_method == "cusum":
                detectors[metric_name] = {
                    "detector": CUSUMDetector(
                        k=0.5,
                        h=5.0,
                        min_samples=max(20, threshold.window_size // 5)
                    ),
                    "config": detector_config
                }
            elif threshold.detection_method == "spc":
                detectors[metric_name] = {
                    "detector": StatisticalProcessControl(
                        control_limit_sigma=3.0,
                        rules=["1", "2", "3", "4"]
                    ),
                    "config": detector_config
                }
            else:
                warnings.warn(f"Unknown detection method for {metric_name}: {threshold.detection_method}")
                # Default to Page-Hinkley
                detectors[metric_name] = {
                    "detector": PageHinkleyDetector(
                        delta=0.005,
                        threshold=50.0,
                        alpha=0.99,
                        min_samples=max(20, threshold.window_size // 5)
                    ),
                    "config": detector_config
                }
        
        return detectors
    
    def add_alert_callback(self, callback: Callable[[DriftAlert], None]):
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def monitor(self, 
                metrics: Dict[str, float],
                metadata: Optional[Dict[str, Any]] = None) -> List[DriftAlert]:
        """
        Monitor metrics for compliance drift.
        
        Args:
            metrics: Dictionary of metric names and values
            metadata: Additional context for monitoring
            
        Returns:
            List of drift alerts (empty if no drift detected)
            
        Example:
            >>> detector = ComplianceDriftDetector("EU_AI_ACT_HIGH_RISK")
            >>> alerts = detector.monitor({
            ...     "accuracy": 0.85,
            ...     "latency_ms": 120.0
            ... }, metadata={"batch_id": 123})
        """
        import time
        start_time = time.time()
        
        if metadata is None:
            metadata = {}
        
        alerts = []
        self.performance_stats["total_checks"] += 1
        self.performance_stats["last_check"] = datetime.now().isoformat()
        
        # Add timestamp to metadata if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
        
        for metric_name, value in metrics.items():
            # Skip if metric not configured
            if metric_name not in self.thresholds:
                continue
            
            threshold_config = self.thresholds[metric_name]
            
            # Update metric history
            self.metrics_history[metric_name].append({
                "timestamp": datetime.now().isoformat(),
                "value": value,
                "metadata": metadata
            })
            
            # Check threshold violation
            threshold_violation, violation_severity = self._check_threshold_violation(
                metric_name, value, threshold_config
            )
            
            if threshold_violation:
                # Create threshold violation alert
                alert = self._create_threshold_alert(
                    metric_name=metric_name,
                    value=value,
                    threshold_config=threshold_config,
                    severity=violation_severity,
                    metadata=metadata,
                    violation_type="threshold"
                )
                alerts.append(alert)
                self._trigger_alert(alert)
            
            # Check statistical drift (even if within thresholds)
            if metric_name in self.detectors:
                detector_info = self.detectors[metric_name]
                detector = detector_info["detector"]
                detection_method = threshold_config.detection_method
                
                statistical_drift = False
                drift_details = {}
                
                if detection_method == "page_hinkley":
                    drift_detected, ph_stat, drift_mag = detector.update(value)
                    statistical_drift = drift_detected
                    drift_details = {
                        "ph_statistic": ph_stat,
                        "drift_magnitude": drift_mag,
                        "threshold": detector.threshold
                    }
                elif detection_method == "ewma":
                    change_detected, z_score = detector.update(value)
                    statistical_drift = change_detected
                    drift_details = {
                        "z_score": z_score,
                        "threshold_sigma": detector.threshold_sigma
                    }
                elif detection_method == "cusum":
                    change_detected, direction = detector.update(value)
                    statistical_drift = change_detected
                    drift_details = {
                        "change_direction": direction,
                        "cusum_positive": detector.cusum_positive,
                        "cusum_negative": detector.cusum_negative
                    }
                elif detection_method == "spc":
                    violations = detector.update(value)
                    statistical_drift = len(violations) > 0
                    drift_details = {
                        "violations": violations,
                        "mean": detector.mean,
                        "std": detector.std
                    }
                
                if statistical_drift and not threshold_violation:
                    # Statistical drift without threshold violation
                    alert = self._create_statistical_drift_alert(
                        metric_name=metric_name,
                        value=value,
                        drift_details=drift_details,
                        threshold_config=threshold_config,
                        metadata=metadata,
                        detection_method=detection_method
                    )
                    alerts.append(alert)
                    self._trigger_alert(alert)
        
        # Check for cross-metric anomalies
        cross_alerts = self._check_cross_metric_anomalies(metrics, metadata)
        alerts.extend(cross_alerts)
        
        # Update performance statistics
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        self.performance_stats["response_time_ms"].append(response_time)
        
        if alerts:
            self.performance_stats["drifts_detected"] += len(alerts)
        
        return alerts
    
    def _check_threshold_violation(self,
                                  metric_name: str,
                                  value: float,
                                  threshold: RegulatoryThreshold) -> Tuple[bool, DriftSeverity]:
        """
        Check if value violates regulatory threshold.
        
        Returns:
            Tuple of (is_violation, severity)
        """
        # Check minimum threshold
        if threshold.min_value is not None and value < threshold.min_value:
            deviation = threshold.min_value - value
            severity = self._map_deviation_to_severity(deviation, threshold.severity_mapping)
            return True, severity
        
        # Check maximum threshold
        if threshold.max_value is not None and value > threshold.max_value:
            deviation = value - threshold.max_value
            severity = self._map_deviation_to_severity(deviation, threshold.severity_mapping)
            return True, severity
        
        # Check target value (if specified)
        if threshold.target_value is not None:
            deviation = abs(value - threshold.target_value)
            if deviation > 0.1:  # Arbitrary threshold for demonstration
                # Map deviation to severity
                if deviation > 0.2:
                    severity = DriftSeverity.CRITICAL
                elif deviation > 0.15:
                    severity = DriftSeverity.MODERATE
                elif deviation > 0.1:
                    severity = DriftSeverity.WARNING
                else:
                    severity = DriftSeverity.INFO
                return True, severity
        
        return False, DriftSeverity.NONE
    
    def _map_deviation_to_severity(self,
                                  deviation: float,
                                  severity_mapping: Dict[float, DriftSeverity]) -> DriftSeverity:
        """Map deviation magnitude to severity level."""
        if not severity_mapping:
            # Default mapping if not specified
            if deviation >= 0.20:
                return DriftSeverity.VIOLATION
            elif deviation >= 0.15:
                return DriftSeverity.CRITICAL
            elif deviation >= 0.10:
                return DriftSeverity.MODERATE
            elif deviation >= 0.05:
                return DriftSeverity.WARNING
            else:
                return DriftSeverity.INFO
        
        # Use custom mapping
        sorted_thresholds = sorted(severity_mapping.items(), key=lambda x: x[0])
        
        # Find appropriate severity level
        for threshold, severity in sorted_thresholds:
            if deviation >= threshold:
                return severity
        
        return DriftSeverity.INFO
    
    def _create_threshold_alert(self,
                               metric_name: str,
                               value: float,
                               threshold_config: RegulatoryThreshold,
                               severity: DriftSeverity,
                               metadata: Dict[str, Any],
                               violation_type: str = "threshold") -> DriftAlert:
        """Create alert for threshold violation."""
        alert_id = f"alert_{int(datetime.now().timestamp())}_{metric_name}_{hashlib.md5(str(value).encode()).hexdigest()[:8]}"
        
        # Determine expected value and direction
        if threshold_config.min_value is not None and value < threshold_config.min_value:
            expected_value = threshold_config.min_value
            direction = "below minimum"
            alert_code = AlertCode.REGULATORY_THRESHOLD_VIOLATION
        elif threshold_config.max_value is not None and value > threshold_config.max_value:
            expected_value = threshold_config.max_value
            direction = "above maximum"
            alert_code = AlertCode.REGULATORY_THRESHOLD_VIOLATION
        elif threshold_config.target_value is not None:
            expected_value = threshold_config.target_value
            direction = "deviated from target"
            alert_code = AlertCode.PERFORMANCE_DECAY
        else:
            expected_value = 0.0
            direction = "unknown deviation"
            alert_code = AlertCode.PERFORMANCE_DECAY
        
        # Create message
        message = (f"Regulatory threshold violation detected: {metric_name} = {value:.3f} "
                  f"({direction} required {expected_value:.3f}). "
                  f"Requirement: {threshold_config.requirement_id}")
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            metric_name=metric_name,
            severity=severity,
            violation_type=violation_type,
            current_value=value,
            expected_value=expected_value,
            threshold_config=threshold_config
        )
        
        alert = DriftAlert(
            alert_id=alert_id,
            code=alert_code,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            metric=metric_name,
            current_value=value,
            expected_value=expected_value,
            threshold=expected_value,
            confidence=0.95,
            metadata={
                "requirement_id": threshold_config.requirement_id,
                "weight": threshold_config.weight,
                "window_size": threshold_config.window_size,
                "detection_method": threshold_config.detection_method,
                "grace_period_hours": threshold_config.grace_period_hours,
                "violation_type": violation_type,
                "direction": direction,
                **metadata
            },
            recommendations=recommendations
        )
        
        return alert
    
    def _create_statistical_drift_alert(self,
                                       metric_name: str,
                                       value: float,
                                       drift_details: Dict[str, Any],
                                       threshold_config: RegulatoryThreshold,
                                       metadata: Dict[str, Any],
                                       detection_method: str) -> DriftAlert:
        """Create alert for statistical drift detection."""
        alert_id = f"stat_alert_{int(datetime.now().timestamp())}_{metric_name}_{hashlib.md5(str(value).encode()).hexdigest()[:8]}"
        
        # Calculate baseline from history
        history = list(self.metrics_history.get(metric_name, []))[:-1]  # Exclude current
        if len(history) > 0:
            baseline_mean = np.mean([h["value"] for h in history])
        else:
            baseline_mean = value
        
        # Create message based on detection method
        if detection_method == "page_hinkley":
            message = (f"Statistical drift detected in {metric_name} using Page-Hinkley test: "
                      f"PH-statistic = {drift_details.get('ph_statistic', 0):.2f}, "
                      f"drift magnitude = {drift_details.get('drift_magnitude', 0):.3f}")
            alert_code = AlertCode.DATA_DISTRIBUTION_SHIFT
        elif detection_method == "ewma":
            message = (f"Change point detected in {metric_name} using EWMA: "
                      f"z-score = {drift_details.get('z_score', 0):.2f}")
            alert_code = AlertCode.COVARIATE_SHIFT
        elif detection_method == "cusum":
            direction = "increase" if drift_details.get('change_direction', 0) > 0 else "decrease"
            message = (f"CUSUM change detection in {metric_name}: "
                      f"direction = {direction}, "
                      f"CUSUM+ = {drift_details.get('cusum_positive', 0):.2f}, "
                      f"CUSUM- = {drift_details.get('cusum_negative', 0):.2f}")
            alert_code = AlertCode.CONCEPT_DRIFT
        elif detection_method == "spc":
            violations = drift_details.get('violations', [])
            message = (f"SPC rule violation(s) detected in {metric_name}: "
                      f"{len(violations)} violation(s), "
                      f"current mean = {drift_details.get('mean', 0):.3f}")
            alert_code = AlertCode.DATA_QUALITY_DECAY
        else:
            message = f"Statistical anomaly detected in {metric_name}"
            alert_code = AlertCode.DATA_DISTRIBUTION_SHIFT
        
        alert = DriftAlert(
            alert_id=alert_id,
            code=alert_code,
            severity=DriftSeverity.WARNING,  # Statistical drift is typically warning level
            message=message,
            timestamp=datetime.now(),
            metric=metric_name,
            current_value=value,
            expected_value=baseline_mean,
            threshold=0.0,  # Statistical threshold already encoded in detection
            confidence=0.90,
            metadata={
                "detection_method": detection_method,
                "detection_details": drift_details,
                "baseline_samples": len(history),
                "requirement_id": threshold_config.requirement_id,
                **metadata
            },
            recommendations=[
                "Monitor metric for continued degradation",
                "Check for data distribution changes",
                "Consider model retraining if drift persists",
                "Review recent data inputs for anomalies"
            ]
        )
        
        return alert
    
    def _check_cross_metric_anomalies(self,
                                     metrics: Dict[str, float],
                                     metadata: Dict[str, Any]) -> List[DriftAlert]:
        """Check for anomalies across multiple metrics."""
        alerts = []
        
        # Example: Check if accuracy and fairness are both degrading
        if "accuracy" in metrics and "fairness_disparity" in metrics:
            accuracy = metrics["accuracy"]
            fairness = metrics["fairness_disparity"]
            
            # Both metrics degrading is particularly concerning
            if accuracy < 0.65 and fairness > 0.25:
                alert_id = f"cross_alert_{int(datetime.now().timestamp())}_acc_fair"
                
                alert = DriftAlert(
                    alert_id=alert_id,
                    code=AlertCode.FAIRNESS_DRIFT,
                    severity=DriftSeverity.CRITICAL,
                    message=(f"Critical cross-metric anomaly: "
                            f"Accuracy ({accuracy:.3f}) and fairness ({fairness:.3f}) "
                            f"both outside acceptable ranges simultaneously."),
                    timestamp=datetime.now(),
                    metric="accuracy_fairness_composite",
                    current_value=accuracy - fairness,  # Composite metric
                    expected_value=0.5,
                    threshold=0.3,
                    metadata={
                        "anomaly_type": "cross_metric_degradation",
                        "accuracy": accuracy,
                        "fairness_disparity": fairness,
                        **metadata
                    },
                    recommendations=[
                        "Immediate model review required",
                        "Check training data for biases",
                        "Consider fairness-aware retraining",
                        "Review recent inference patterns"
                    ]
                )
                
                alerts.append(alert)
                self._trigger_alert(alert)
        
        # Check for performance-latency trade-off anomalies
        if "accuracy" in metrics and "inference_latency_ms" in metrics:
            accuracy = metrics["accuracy"]
            latency = metrics["inference_latency_ms"]
            
            # High latency with low accuracy is problematic
            if latency > 2000 and accuracy < 0.70:
                alert_id = f"cross_alert_{int(datetime.now().timestamp())}_acc_latency"
                
                alert = DriftAlert(
                    alert_id=alert_id,
                    code=AlertCode.LATENCY_INCREASE,
                    severity=DriftSeverity.MODERATE,
                    message=(f"Performance-latency trade-off anomaly: "
                            f"High latency ({latency:.0f} ms) with low accuracy ({accuracy:.3f})"),
                    timestamp=datetime.now(),
                    metric="accuracy_latency_ratio",
                    current_value=accuracy / (latency / 1000),  # Accuracy per second
                    expected_value=0.5,
                    threshold=0.3,
                    metadata={
                        "anomaly_type": "performance_latency_tradeoff",
                        "accuracy": accuracy,
                        "latency_ms": latency,
                        **metadata
                    },
                    recommendations=[
                        "Check system resource utilization",
                        "Review model optimization settings",
                        "Consider model quantization or pruning",
                        "Check for infrastructure bottlenecks"
                    ]
                )
                
                alerts.append(alert)
                self._trigger_alert(alert)
        
        return alerts
    
    def _generate_recommendations(self,
                                 metric_name: str,
                                 severity: DriftSeverity,
                                 violation_type: str,
                                 current_value: float,
                                 expected_value: float,
                                 threshold_config: RegulatoryThreshold) -> List[str]:
        """Generate actionable recommendations based on drift."""
        recommendations = []
        
        # Base recommendations by severity
        base_recommendations = {
            DriftSeverity.INFO: [
                "Document deviation in audit trail",
                "Monitor metric more frequently for trends",
                "Review next scheduled maintenance window"
            ],
            DriftSeverity.WARNING: [
                "Increase monitoring frequency for this metric",
                "Investigate potential root causes",
                "Prepare contingency plan if trend continues"
            ],
            DriftSeverity.MODERATE: [
                "Schedule immediate diagnostic review",
                "Check data pipeline for issues",
                "Prepare retraining pipeline",
                "Notify compliance officer"
            ],
            DriftSeverity.CRITICAL: [
                "Initiate immediate root cause analysis",
                "Consider temporary model rollback",
                "Escalate to senior management",
                "Prepare incident report"
            ],
            DriftSeverity.VIOLATION: [
                "STOP: Regulatory violation detected",
                "Immediate escalation to compliance officer",
                "Initiate incident response protocol",
                "Prepare regulatory disclosure if required",
                "Consider system shutdown if safety-critical"
            ]
        }
        
        # Add severity-based recommendations
        if severity in base_recommendations:
            recommendations.extend(base_recommendations[severity])
        
        # Add metric-specific recommendations
        metric_specific = {
            "accuracy": [
                "Validate recent training data quality",
                "Check for concept drift in production data",
                "Review feature engineering pipeline",
                "Consider ensemble methods for stability"
            ],
            "fairness_disparity": [
                "Audit training data for representation bias",
                "Check preprocessing for disparate impact",
                "Consider fairness constraints in next training",
                "Review protected attribute handling"
            ],
            "explainability_stability": [
                "Verify explanation generation stability",
                "Check feature importance calculation",
                "Review SHAP/LIME configuration",
                "Consider alternative explanation methods"
            ],
            "inference_latency_ms": [
                "Check infrastructure health and load",
                "Review model optimization settings",
                "Consider model quantization or pruning",
                "Evaluate hardware acceleration options"
            ],
            "data_quality_score": [
                "Validate data source integrity",
                "Check data preprocessing pipeline",
                "Review data validation rules",
                "Implement additional quality checks"
            ]
        }
        
        if metric_name in metric_specific:
            recommendations.extend(metric_specific[metric_name])
        
        # Add regulatory-specific recommendations
        if "EU_AI_ACT" in threshold_config.requirement_id:
            recommendations.extend([
                "Review EU AI Act Article compliance documentation",
                "Ensure human oversight mechanisms are functional",
                "Verify risk management system is active"
            ])
        elif "FDA" in threshold_config.requirement_id:
            recommendations.extend([
                "Review FDA 21 CFR Part 820 compliance",
                "Check design control documentation",
                "Verify validation protocols are up-to-date"
            ])
        elif "REG_B" in threshold_config.requirement_id:
            recommendations.extend([
                "Conduct disparate impact analysis",
                "Review fair lending compliance documentation",
                "Verify model does not use prohibited bases"
            ])
        
        # Deduplicate recommendations
        return list(dict.fromkeys(recommendations))
    
    def _trigger_alert(self, alert: DriftAlert):
        """Trigger alert through callbacks and internal storage."""
        # Store alert
        self.alerts.append(alert)
        
        # Save to file if storage path configured
        if self.storage_path:
            self._save_alert_to_file(alert)
        
        # Call external callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                warnings.warn(f"Alert callback failed: {e}")
        
        # Log based on severity
        if alert.severity >= DriftSeverity.CRITICAL:
            print(f"🚨 {alert.severity.code} ALERT: {alert.message}")
        elif alert.severity >= DriftSeverity.MODERATE:
            print(f"⚠️ {alert.severity.code} ALERT: {alert.message}")
        elif alert.severity >= DriftSeverity.WARNING:
            print(f"⚠️ {alert.severity.code}: {alert.message}")
    
    def _save_alert_to_file(self, alert: DriftAlert):
        """Save alert to JSON file."""
        alert_file = self.storage_path / f"alert_{alert.alert_id}.json"
        with open(alert_file, 'w') as f:
            json.dump(alert.to_dict(), f, indent=2, default=str)
    
    def calculate_rddr(self, 
                      validation_period_days: int = 7,
                      ground_truth: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Calculate Regulatory Drift Detection Rate (RDDR).
        
        Args:
            validation_period_days: Days to consider for calculation
            ground_truth: Ground truth data for validation (optional)
            
        Returns:
            Dictionary with RDDR and related statistics
            
        Example:
            >>> rddr_stats = detector.calculate_rddr(validation_period_days=30)
            >>> print(f"RDDR: {rddr_stats['RDDR']:.3f}")
        """
        cutoff_time = datetime.now() - timedelta(days=validation_period_days)
        
        # Filter recent alerts
        recent_alerts = [
            alert for alert in self.alerts
            if alert.timestamp >= cutoff_time
        ]
        
        if ground_truth is None:
            # Estimate ground truth based on alert severity and patterns
            # This is a simplified estimation - in production, would use actual ground truth
            
            # Count true positives (alerts with moderate or higher severity)
            true_positives = len([
                alert for alert in recent_alerts
                if alert.severity >= DriftSeverity.MODERATE
            ])
            
            # Estimate false negatives based on performance stats
            total_checks = self.performance_stats.get("total_checks", 0)
            checks_in_period = total_checks * validation_period_days / 30  # Estimate
            
            # Assume 1% of checks should have generated alerts but didn't
            estimated_false_negatives = max(1, int(checks_in_period * 0.01))
        else:
            # Use provided ground truth
            true_positives = 0
            false_negatives = 0
            
            for truth in ground_truth:
                # Match ground truth with alerts
                matched = any(
                    alert.metric == truth.get("metric") and
                    abs(alert.timestamp - datetime.fromisoformat(truth.get("timestamp"))) < timedelta(hours=1)
                    for alert in recent_alerts
                )
                
                if matched and truth.get("is_drift", False):
                    true_positives += 1
                elif not matched and truth.get("is_drift", False):
                    false_negatives += 1
        
        # Calculate RDDR
        if true_positives + estimated_false_negatives > 0:
            rddr = true_positives / (true_positives + estimated_false_negatives)
        else:
            rddr = 0.0
        
        return {
            "RDDR": rddr,
            "true_positives": true_positives,
            "estimated_false_negatives": estimated_false_negatives,
            "total_alerts_analyzed": len(recent_alerts),
            "validation_period_days": validation_period_days,
            "detection_latency_avg_ms": np.mean(self.performance_stats["response_time_ms"]) 
            if self.performance_stats["response_time_ms"] else 0.0,
            "detection_latency_p95_ms": np.percentile(list(self.performance_stats["response_time_ms"]), 95) 
            if self.performance_stats["response_time_ms"] else 0.0,
            "interpretation": self._interpret_rddr(rddr)
        }
    
    def _interpret_rddr(self, rddr: float) -> str:
        """Interpret RDDR value."""
        if rddr >= 0.90:
            return "Excellent detection capability"
        elif rddr >= 0.80:
            return "Good detection capability"
        elif rddr >= 0.70:
            return "Adequate detection capability"
        elif rddr >= 0.60:
            return "Moderate detection capability"
        else:
            return "Detection capability needs improvement"
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics."""
        stats = {
            **self.performance_stats,
            "active_thresholds": len(self.thresholds),
            "active_detectors": len(self.detectors),
            "recent_alerts": len(self.alerts),
            "regulatory_context": self.regulatory_context,
            "current_time": datetime.now().isoformat()
        }
        
        # Calculate response time statistics
        response_times = list(self.performance_stats["response_time_ms"])
        if response_times:
            stats.update({
                "mean_response_time_ms": np.mean(response_times),
                "median_response_time_ms": np.median(response_times),
                "p95_response_time_ms": np.percentile(response_times, 95),
                "min_response_time_ms": np.min(response_times),
                "max_response_time_ms": np.max(response_times)
            })
        
        # Add detector-specific statistics
        detector_stats = {}
        for metric_name, detector_info in self.detectors.items():
            detector = detector_info["detector"]
            if hasattr(detector, 'get_statistics'):
                detector_stats[metric_name] = detector.get_statistics()
        
        stats["detector_statistics"] = detector_stats
        
        return stats
    
    def get_alert_summary(self, 
                         severity_filter: Optional[DriftSeverity] = None,
                         time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Get summary of alerts with filtering options."""
        filtered_alerts = list(self.alerts)
        
        # Apply severity filter
        if severity_filter:
            filtered_alerts = [a for a in filtered_alerts if a.severity >= severity_filter]
        
        # Apply time range filter
        if time_range:
            start_time, end_time = time_range
            filtered_alerts = [
                a for a in filtered_alerts
                if start_time <= a.timestamp <= end_time
            ]
        
        # Calculate statistics
        severity_counts = {}
        metric_counts = {}
        
        for alert in filtered_alerts:
            severity_counts[alert.severity.code] = severity_counts.get(alert.severity.code, 0) + 1
            metric_counts[alert.metric] = metric_counts.get(alert.metric, 0) + 1
        
        return {
            "total_alerts": len(filtered_alerts),
            "severity_distribution": severity_counts,
            "metric_distribution": metric_counts,
            "time_range": {
                "start": time_range[0].isoformat() if time_range else None,
                "end": time_range[1].isoformat() if time_range else None
            } if time_range else None,
            "alerts": [alert.to_dict() for alert in filtered_alerts[:10]]  # First 10 alerts
        }
    
    def graceful_degradation(self, error_code: int) -> Dict[str, Any]:
        """
        Implement graceful degradation based on error codes.
        
        Args:
            error_code: Error code from error handling system
            
        Returns:
            Degradation mode configuration
        """
        degradation_modes = {
            0xE100: {  # Non-critical explanation failure
                "action": "continue",
                "degradation_level": "partial",
                "degradation_description": "Partial explanation functionality",
                "log_level": "WARNING",
                "alert_required": False,
                "monitoring_frequency": "normal",
                "compliance_impact": "low"
            },
            0xW200: {  # Warning drift alert
                "action": "degraded_mode",
                "degradation_level": "moderate",
                "degradation_description": "Increased monitoring with reduced functionality",
                "log_level": "WARNING",
                "alert_required": True,
                "monitoring_frequency": "5min",  # Increase from 30min
                "compliance_impact": "medium",
                "fallback_mechanisms": ["human_review_queue", "simplified_explanations"]
            },
            0xC300: {  # Critical constraint violation
                "action": "halt_pipeline",
                "degradation_level": "severe",
                "degradation_description": "System halted due to regulatory violation",
                "log_level": "CRITICAL",
                "alert_required": True,
                "escalation": "immediate",
                "compliance_impact": "high",
                "fallback_mechanisms": ["human_operator_intervention", "legacy_system_fallback"],
                "recovery_procedure": "manual_validation_required"
            }
        }
        
        if error_code in degradation_modes:
            mode = degradation_modes[error_code].copy()
            error_info = self.error_codes.get(error_code, ("Unknown", "Unknown error", DriftSeverity.INFO))
            mode.update({
                "error_code": hex(error_code),
                "error_level": error_info[0],
                "error_description": error_info[1],
                "error_severity": error_info[2].code
            })
            return mode
        else:
            return {
                "action": "continue",
                "degradation_level": "none",
                "log_level": "INFO",
                "alert_required": False,
                "error": f"Unknown error code: {hex(error_code)}"
            }
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current configuration for documentation or replication."""
        config = {
            "regulatory_context": self.regulatory_context,
            "thresholds": {},
            "detectors": {},
            "export_timestamp": datetime.now().isoformat(),
            "auditops_version": "1.0.0"
        }
        
        # Export thresholds
        for metric_name, threshold in self.thresholds.items():
            config["thresholds"][metric_name] = {
                "requirement_id": threshold.requirement_id,
                "min_value": threshold.min_value,
                "max_value": threshold.max_value,
                "target_value": threshold.target_value,
                "window_size": threshold.window_size,
                "weight": threshold.weight,
                "detection_method": threshold.detection_method,
                "grace_period_hours": threshold.grace_period_hours,
                "severity_mapping": {
                    str(dev): severity.code 
                    for dev, severity in threshold.severity_mapping.items()
                }
            }
        
        # Export detector configurations
        for metric_name, detector_info in self.detectors.items():
            detector = detector_info["detector"]
            config["detectors"][metric_name] = detector_info["config"]
            
            if hasattr(detector, 'get_statistics'):
                config["detectors"][metric_name]["current_stats"] = detector.get_statistics()
        
        return config
    
    def reset_detector(self, metric_name: str):
        """Reset detector for a specific metric."""
        if metric_name in self.detectors:
            detector_info = self.detectors[metric_name]
            detector_info["detector"].reset()
            print(f"Reset detector for metric: {metric_name}")
        else:
            warnings.warn(f"No detector found for metric: {metric_name}")
    
    def reset_all_detectors(self):
        """Reset all detectors."""
        for metric_name, detector_info in self.detectors.items():
            detector_info["detector"].reset()
        print("Reset all detectors")


# Example usage and demonstration
if __name__ == "__main__":
    print("Demonstrating ComplianceDriftDetector functionality...")
    
    # Initialize detector
    detector = ComplianceDriftDetector(
        regulatory_context="EU_AI_ACT_HIGH_RISK",
        storage_path="./compliance_alerts"
    )
    
    # Add alert callback
    def alert_handler(alert: DriftAlert):
        print(f"ALERT RECEIVED: {alert}")
    
    detector.add_alert_callback(alert_handler)
    
    # Simulate monitoring over time
    print("\nSimulating 100 days of compliance monitoring...")
    
    # Baseline metrics (good performance)
    baseline_metrics = {
        "accuracy": 0.88,
        "explainability_stability": 0.85,
        "fairness_disparity": 0.15,
        "inference_latency_ms": 450.0,
        "data_quality_score": 0.92
    }
    
    all_alerts = []
    
    for day in range(100):
        # Simulate metric values with some noise and potential drift
        metrics = {}
        for metric, baseline in baseline_metrics.items():
            # Add random noise
            noise = np.random.normal(0, 0.02)
            
            # Simulate gradual drift starting at day 50
            if day >= 50:
                if metric == "accuracy":
                    drift = -0.005 * (day - 50)  # Gradual accuracy decay
                elif metric == "fairness_disparity":
                    drift = 0.002 * (day - 50)  # Gradual fairness degradation
                elif metric == "inference_latency_ms":
                    # Random latency spikes
                    drift = np.random.exponential(50) if np.random.random() < 0.1 else 0
                else:
                    drift = 0
            else:
                drift = 0
            
            # Apply bounds
            if metric == "accuracy":
                metrics[metric] = max(0.0, min(1.0, baseline + noise + drift))
            elif metric == "explainability_stability":
                metrics[metric] = max(0.0, min(1.0, baseline + noise + drift))
            elif metric == "fairness_disparity":
                metrics[metric] = max(0.0, min(1.0, baseline + noise + drift))
            elif metric == "inference_latency_ms":
                metrics[metric] = max(0.0, baseline + noise + drift)
            elif metric == "data_quality_score":
                metrics[metric] = max(0.0, min(1.0, baseline + noise + drift))
        
        # Add some random anomalies
        if day in [25, 75]:
            metrics["accuracy"] = 0.55  # Severe accuracy drop
        if day == 60:
            metrics["fairness_disparity"] = 0.35  # Fairness violation
        
        # Monitor metrics
        alerts = detector.monitor(
            metrics,
            metadata={
                "day": day,
                "simulation": True,
                "batch_id": f"batch_{day:03d}"
            }
        )
        
        if alerts:
            all_alerts.extend(alerts)
            print(f"Day {day:3d}: {len(alerts)} alert(s)")
    
    # Calculate and display statistics
    print("\n" + "="*80)
    print("COMPLIANCE DRIFT DETECTION STATISTICS")
    print("="*80)
    
    stats = detector.get_detection_stats()
    for key, value in stats.items():
        if key != "detector_statistics":  # Skip detailed detector stats
            if isinstance(value, float):
                print(f"{key:30}: {value:.4f}")
            else:
                print(f"{key:30}: {value}")
    
    # Calculate RDDR
    rddr_stats = detector.calculate_rddr(validation_period_days=30)
    print(f"\nRDDR (30-day window): {rddr_stats['RDDR']:.3f}")
    print(f"True Positives: {rddr_stats['true_positives']}")
    print(f"Estimated False Negatives: {rddr_stats['estimated_false_negatives']}")
    print(f"Detection Latency (avg): {rddr_stats['detection_latency_avg_ms']:.2f} ms")
    print(f"Interpretation: {rddr_stats['interpretation']}")
    
    # Get alert summary
    alert_summary = detector.get_alert_summary(
        severity_filter=DriftSeverity.WARNING,
        time_range=(datetime.now() - timedelta(days=30), datetime.now())
    )
    
    print(f"\nAlert Summary (last 30 days, WARNING+):")
    print(f"Total Alerts: {alert_summary['total_alerts']}")
    print(f"Severity Distribution: {alert_summary['severity_distribution']}")
    print(f"Metric Distribution: {alert_summary['metric_distribution']}")
    
    # Test graceful degradation
    print("\nGraceful Degradation Tests:")
    for error_code in [0xE100, 0xW200, 0xC300]:
        mode = detector.graceful_degradation(error_code)
        print(f"Error {hex(error_code)}: {mode['action']} - {mode.get('error_description', '')}")
    
    # Export configuration
    config = detector.export_configuration()
    with open("compliance_detector_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    
    print("\nConfiguration exported to 'compliance_detector_config.json'")
    print("\nComplianceDriftDetector demonstration completed successfully!")
