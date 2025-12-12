"""
Compliance Drift Detection Mechanism for AuditOps Framework
Implements real-time monitoring for regulatory deviations
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
from collections import deque
import time
from scipy import stats


class DriftSeverity(Enum):
    """Severity levels for detected drift"""
    NONE = 0
    WARNING = 1  # Minor deviation, monitor
    MODERATE = 2  # Significant deviation, investigate
    CRITICAL = 3  # Severe deviation, immediate action required
    VIOLATION = 4  # Regulatory violation detected


class AlertCode(Enum):
    """Alert codes for different drift types"""
    # Performance drift codes (0xP series)
    PERFORMANCE_DECAY = 0xP100
    ACCURACY_DROP = 0xP101
    FAIRNESS_DRIFT = 0xP102
    LATENCY_INCREASE = 0xP103
    
    # Data drift codes (0xD series)
    DATA_DISTRIBUTION_SHIFT = 0xD200
    COVARIATE_SHIFT = 0xD201
    CONCEPT_DRIFT = 0xD202
    DATA_QUALITY_DECAY = 0xD203
    
    # Regulatory drift codes (0xR series)
    THRESHOLD_VIOLATION = 0xR300
    EXPLAINABILITY_DECAY = 0xR301
    AUDIT_TRAIL_GAP = 0xR302
    COMPLIANCE_CHECK_FAILURE = 0xR303
    
    # System drift codes (0xS series)
    RESOURCE_USAGE_SPIKE = 0xS400
    DEPENDENCY_VERSION_DRIFT = 0xS401
    ENVIRONMENT_CONFIG_DRIFT = 0xS402


@dataclass
class DriftAlert:
    """Container for drift alerts"""
    alert_id: str
    code: AlertCode
    severity: DriftSeverity
    message: str
    timestamp: float
    metric: str
    current_value: float
    expected_value: float
    threshold: float
    confidence: float = 0.95
    metadata: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    acknowledged: bool = False


@dataclass
class RegulatoryThreshold:
    """Definition of a regulatory threshold"""
    metric_name: str
    requirement_id: str  # e.g., "EU_AI_ACT_ART_13", "FDA_PART_820_75"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    target_value: Optional[float] = None
    window_size: int = 100  # Samples to consider for detection
    severity_mapping: Dict[float, DriftSeverity] = field(default_factory=dict)
    weight: float = 1.0  # Importance weight for CCS calculation


class PageHinkleyDetector:
    """
    Page-Hinkley test for detecting changes in running processes
    """
    
    def __init__(self, delta: float = 0.005, threshold: float = 50.0, alpha: float = 1.0):
        """
        Initialize Page-Hinkley detector
        
        Args:
            delta: Magnitude of changes to detect (smaller = more sensitive)
            threshold: Detection threshold
            alpha: Forgetting factor (0 < alpha <= 1)
        """
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.reset()
    
    def reset(self):
        """Reset detector state"""
        self.cumulative_sum = 0.0
        self.min_cumulative_sum = float('inf')
        self.sample_count = 0
        self.mean = 0.0
        self.drift_detected = False
    
    def update(self, value: float) -> Tuple[bool, float]:
        """
        Update detector with new value
        
        Args:
            value: New observation
            
        Returns:
            (drift_detected, test_statistic)
        """
        self.sample_count += 1
        
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
        ph_statistic = self.cumulative_sum - self.min_cumulative_sum
        
        # Check for drift
        self.drift_detected = ph_statistic > self.threshold
        
        return self.drift_detected, ph_statistic


class ComplianceDriftDetector:
    """
    Main class for detecting compliance drift in ML systems
    """
    
    def __init__(self, 
                 regulatory_context: str = "EU_AI_ACT_HIGH_RISK",
                 alert_callback: Optional[Callable] = None):
        """
        Initialize compliance drift detector
        
        Args:
            regulatory_context: Regulatory framework to monitor
            alert_callback: Function to call when alert is generated
        """
        self.regulatory_context = regulatory_context
        self.alert_callback = alert_callback
        self.thresholds = self._load_regulatory_thresholds(regulatory_context)
        self.detectors = self._initialize_detectors()
        self.alerts = deque(maxlen=1000)  # Store recent alerts
        self.metrics_history = {}
        self.error_codes = {
            0xE100: "Non-critical: Explanation partial failure",
            0xW200: "Warning: Drift alert, degradation mode",
            0xC300: "Critical: Mandatory constraint violation, pipeline halt"
        }
        
        # Performance metrics
        self.detection_stats = {
            "total_checks": 0,
            "drifts_detected": 0,
            "false_positives": 0,
            "response_time_ms": []
        }
    
    def _load_regulatory_thresholds(self, context: str) -> Dict[str, RegulatoryThreshold]:
        """
        Load regulatory thresholds based on context
        
        Args:
            context: Regulatory context identifier
            
        Returns:
            Dictionary of threshold configurations
        """
        thresholds = {}
        
        if context == "EU_AI_ACT_HIGH_RISK":
            thresholds = {
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
                    weight=1.0
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
                    weight=0.9
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
                    weight=1.0
                ),
                "inference_latency": RegulatoryThreshold(
                    metric_name="inference_latency_ms",
                    requirement_id="FDA_CLASS_II_PERF",
                    max_value=1000.0,  # 1 second
                    window_size=50,
                    severity_mapping={
                        1500.0: DriftSeverity.WARNING,
                        2000.0: DriftSeverity.MODERATE,
                        3000.0: DriftSeverity.CRITICAL,
                        5000.0: DriftSeverity.VIOLATION
                    },
                    weight=0.7
                )
            }
        elif context == "FDA_21_CFR_820":
            # FDA-specific thresholds
            thresholds = {
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
                    weight=1.0
                ),
                "specificity": RegulatoryThreshold(
                    metric_name="specificity",
                    requirement_id="FDA_820_75_B",
                    min_value=0.80,
                    window_size=100,
                    weight=1.0
                )
            }
        
        return thresholds
    
    def _initialize_detectors(self) -> Dict[str, Any]:
        """
        Initialize drift detectors for each metric
        
        Returns:
            Dictionary of initialized detectors
        """
        detectors = {}
        
        for metric_name in self.thresholds.keys():
            # Use Page-Hinkley for performance metrics
            if metric_name in ["accuracy", "sensitivity", "specificity", "auc"]:
                detectors[metric_name] = PageHinkleyDetector(
                    delta=0.005,
                    threshold=50.0,
                    alpha=0.99
                )
            # Use different parameters for latency
            elif "latency" in metric_name:
                detectors[metric_name] = PageHinkleyDetector(
                    delta=0.01,
                    threshold=30.0,
                    alpha=0.95
                )
            # Default detector
            else:
                detectors[metric_name] = PageHinkleyDetector(
                    delta=0.005,
                    threshold=50.0,
                    alpha=0.99
                )
        
        return detectors
    
    def monitor(self, 
                metrics: Dict[str, float],
                metadata: Dict[str, Any] = None) -> List[DriftAlert]:
        """
        Monitor metrics for compliance drift
        
        Args:
            metrics: Dictionary of metric names and values
            metadata: Additional context for monitoring
            
        Returns:
            List of drift alerts (empty if no drift)
        """
        if metadata is None:
            metadata = {}
        
        start_time = time.time()
        self.detection_stats["total_checks"] += 1
        
        alerts = []
        
        for metric_name, value in metrics.items():
            if metric_name not in self.thresholds:
                continue
            
            threshold_config = self.thresholds[metric_name]
            
            # Update metric history
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = deque(
                    maxlen=threshold_config.window_size
                )
            self.metrics_history[metric_name].append(value)
            
            # Check threshold violation
            threshold_violation, severity = self._check_threshold_violation(
                metric_name, value, threshold_config
            )
            
            if threshold_violation:
                alert = self._create_threshold_alert(
                    metric_name, value, threshold_config, severity, metadata
                )
                alerts.append(alert)
                self._trigger_alert(alert)
            
            # Check statistical drift (even if within thresholds)
            if metric_name in self.detectors:
                drift_detected, ph_statistic = self.detectors[metric_name].update(value)
                
                if drift_detected and not threshold_violation:
                    # Statistical drift without threshold violation
                    alert = self._create_statistical_drift_alert(
                        metric_name, value, ph_statistic, threshold_config, metadata
                    )
                    alerts.append(alert)
                    self._trigger_alert(alert)
        
        # Check for cross-metric anomalies
        cross_alerts = self._check_cross_metric_anomalies(metrics, metadata)
        alerts.extend(cross_alerts)
        
        # Update detection statistics
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        self.detection_stats["response_time_ms"].append(response_time)
        self.detection_stats["response_time_ms"] = self.detection_stats[
            "response_time_ms"
        ][-100:]  # Keep last 100 measurements
        
        if alerts:
            self.detection_stats["drifts_detected"] += 1
        
        return alerts
    
    def _check_threshold_violation(self,
                                  metric_name: str,
                                  value: float,
                                  threshold: RegulatoryThreshold) -> Tuple[bool, DriftSeverity]:
        """
        Check if value violates regulatory threshold
        
        Returns:
            (is_violation, severity)
        """
        severity = DriftSeverity.NONE
        
        # Check min threshold
        if threshold.min_value is not None and value < threshold.min_value:
            deviation = threshold.min_value - value
            severity = self._map_deviation_to_severity(
                deviation, threshold.severity_mapping
            )
            return True, severity
        
        # Check max threshold
        if threshold.max_value is not None and value > threshold.max_value:
            deviation = value - threshold.max_value
            severity = self._map_deviation_to_severity(
                deviation, threshold.severity_mapping
            )
            return True, severity
        
        return False, severity
    
    def _map_deviation_to_severity(self,
                                  deviation: float,
                                  severity_mapping: Dict[float, DriftSeverity]) -> DriftSeverity:
        """
        Map deviation magnitude to severity level
        """
        if not severity_mapping:
            # Default mapping if not specified
            if deviation < 0.05:
                return DriftSeverity.WARNING
            elif deviation < 0.10:
                return DriftSeverity.MODERATE
            elif deviation < 0.20:
                return DriftSeverity.CRITICAL
            else:
                return DriftSeverity.VIOLATION
        
        # Use custom mapping
        sorted_thresholds = sorted(severity_mapping.items(), key=lambda x: x[0])
        
        for threshold, mapped_severity in sorted_thresholds:
            if deviation >= threshold:
                severity = mapped_severity
            else:
                break
        
        return severity if severity != DriftSeverity.NONE else DriftSeverity.WARNING
    
    def _create_threshold_alert(self,
                               metric_name: str,
                               value: float,
                               threshold: RegulatoryThreshold,
                               severity: DriftSeverity,
                               metadata: Dict[str, Any]) -> DriftAlert:
        """
        Create alert for threshold violation
        """
        alert_id = f"alert_{int(time.time())}_{metric_name}"
        
        # Determine expected value
        if threshold.min_value is not None and value < threshold.min_value:
            expected_value = threshold.min_value
            direction = "below"
        elif threshold.max_value is not None and value > threshold.max_value:
            expected_value = threshold.max_value
            direction = "above"
        else:
            expected_value = threshold.target_value or 0.0
            direction = "deviated from"
        
        # Create message
        message = (
            f"Regulatory threshold violation detected: {metric_name} = {value:.3f} "
            f"({direction} required {expected_value:.3f}). "
            f"Requirement: {threshold.requirement_id}"
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            metric_name, severity, value, expected_value
        )
        
        # Determine alert code
        if severity == DriftSeverity.VIOLATION:
            alert_code = AlertCode.THRESHOLD_VIOLATION
        else:
            alert_code = AlertCode.PERFORMANCE_DECAY
        
        alert = DriftAlert(
            alert_id=alert_id,
            code=alert_code,
            severity=severity,
            message=message,
            timestamp=time.time(),
            metric=metric_name,
            current_value=value,
            expected_value=expected_value,
            threshold=expected_value,
            metadata={
                "requirement_id": threshold.requirement_id,
                "weight": threshold.weight,
                "window_size": threshold.window_size,
                **metadata
            },
            recommendations=recommendations
        )
        
        return alert
    
    def _create_statistical_drift_alert(self,
                                       metric_name: str,
                                       value: float,
                                       ph_statistic: float,
                                       threshold: RegulatoryThreshold,
                                       metadata: Dict[str, Any]) -> DriftAlert:
        """
        Create alert for statistical drift (Page-Hinkley detection)
        """
        alert_id = f"stat_alert_{int(time.time())}_{metric_name}"
        
        # Calculate baseline from history
        history = list(self.metrics_history.get(metric_name, []))[:-1]  # Exclude current
        if len(history) > 0:
            baseline_mean = np.mean(history)
        else:
            baseline_mean = value
        
        message = (
            f"Statistical drift detected in {metric_name}: "
            f"current={value:.3f}, baseline≈{baseline_mean:.3f}, "
            f"PH-statistic={ph_statistic:.2f}"
        )
        
        alert = DriftAlert(
            alert_id=alert_id,
            code=AlertCode.PERFORMANCE_DECAY,
            severity=DriftSeverity.WARNING,  # Statistical drift is typically warning level
            message=message,
            timestamp=time.time(),
            metric=metric_name,
            current_value=value,
            expected_value=baseline_mean,
            threshold=ph_statistic,
            confidence=0.90,
            metadata={
                "detection_method": "Page-Hinkley",
                "ph_statistic": ph_statistic,
                "baseline_samples": len(history),
                "requirement_id": threshold.requirement_id,
                **metadata
            },
            recommendations=[
                "Monitor metric for continued degradation",
                "Check for data distribution changes",
                "Consider model retraining if drift persists"
            ]
        )
        
        return alert
    
    def _check_cross_metric_anomalies(self,
                                     metrics: Dict[str, float],
                                     metadata: Dict[str, Any]) -> List[DriftAlert]:
        """
        Check for anomalies across multiple metrics
        """
        alerts = []
        
        # Example: Check if accuracy and fairness are both degrading
        if "accuracy" in metrics and "fairness_disparity" in metrics:
            accuracy = metrics["accuracy"]
            fairness = metrics["fairness_disparity"]
            
            # Both metrics degrading is particularly concerning
            if (accuracy < 0.65 and fairness > 0.25):
                alert_id = f"cross_alert_{int(time.time())}_acc_fair"
                
                alert = DriftAlert(
                    alert_id=alert_id,
                    code=AlertCode.FAIRNESS_DRIFT,
                    severity=DriftSeverity.CRITICAL,
                    message=(
                        "Critical cross-metric anomaly: "
                        f"Accuracy ({accuracy:.3f}) and fairness ({fairness:.3f}) "
                        "both outside acceptable ranges simultaneously."
                    ),
                    timestamp=time.time(),
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
                        "Consider fairness-aware retraining"
                    ]
                )
                
                alerts.append(alert)
                self._trigger_alert(alert)
        
        return alerts
    
    def _generate_recommendations(self,
                                 metric_name: str,
                                 severity: DriftSeverity,
                                 current_value: float,
                                 expected_value: float) -> List[str]:
        """
        Generate actionable recommendations based on drift
        """
        recommendations = []
        
        base_recommendations = {
            DriftSeverity.WARNING: [
                "Increase monitoring frequency",
                "Document deviation in audit trail",
                "Review next scheduled retraining"
            ],
            DriftSeverity.MODERATE: [
                "Schedule model diagnostic review",
                "Check data pipeline for issues",
                "Prepare retraining pipeline"
            ],
            DriftSeverity.CRITICAL: [
                "Pause model in production (if applicable)",
                "Immediate root cause analysis",
                "Initiate emergency retraining"
            ],
            DriftSeverity.VIOLATION: [
                "STOP: Regulatory violation detected",
                "Escalate to compliance officer",
                "Initiate incident response protocol",
                "Prepare regulatory disclosure if required"
            ]
        }
        
        # Add metric-specific recommendations
        if "accuracy" in metric_name or "auc" in metric_name:
            recommendations.extend([
                "Validate recent training data quality",
                "Check for concept drift in production data",
                "Review feature engineering pipeline"
            ])
        elif "fairness" in metric_name:
            recommendations.extend([
                "Audit training data for representation bias",
                "Check preprocessing for disparate impact",
                "Consider fairness constraints in next training"
            ])
        elif "latency" in metric_name:
            recommendations.extend([
                "Check infrastructure health and load",
                "Review model optimization settings",
                "Consider model quantization or pruning"
            ])
        elif "explainability" in metric_name:
            recommendations.extend([
                "Verify explanation generation stability",
                "Check feature importance calculation",
                "Review SHAP/LIME configuration"
            ])
        
        # Add severity-based recommendations
        if severity in base_recommendations:
            recommendations = base_recommendations[severity] + recommendations
        
        return recommendations
    
    def _trigger_alert(self, alert: DriftAlert):
        """
        Trigger alert through callback and internal storage
        """
        # Store alert
        self.alerts.append(alert)
        
        # Call external callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                warnings.warn(f"Alert callback failed: {e}")
        
        # Log based on severity
        if alert.severity == DriftSeverity.VIOLATION:
            print(f"🚨 VIOLATION ALERT: {alert.message}")
            # In production, this would trigger incident management system
        elif alert.severity == DriftSeverity.CRITICAL:
            print(f"⚠️ CRITICAL ALERT: {alert.message}")
        elif alert.severity == DriftSeverity.MODERATE:
            print(f"⚠️ MODERATE ALERT: {alert.message}")
    
    def calculate_rddr(self, validation_period: int = 30) -> Dict[str, float]:
        """
        Calculate Regulatory Drift Detection Rate (RDDR)
        
        Args:
            validation_period: Days to consider for calculation
            
        Returns:
            Dictionary with RDDR and related statistics
        """
        now = time.time()
        period_seconds = validation_period * 24 * 3600
        
        # Filter alerts from validation period
        recent_alerts = [
            alert for alert in self.alerts
            if now - alert.timestamp <= period_seconds
        ]
        
        # Count true positives and false negatives
        # In real implementation, this would compare against ground truth
        # For simulation, we assume all critical/vi alerts are TP
        true_positives = len([
            alert for alert in recent_alerts
            if alert.severity in [DriftSeverity.CRITICAL, DriftSeverity.VIOLATION]
        ])
        
        # Estimate false negatives (this would require ground truth)
        # For this example, we estimate based on total checks
        total_checks = self.detection_stats["total_checks"]
        estimated_false_negatives = max(0, total_checks // 100)  # Example: 1% FN rate
        
        if true_positives + estimated_false_negatives == 0:
            rddr = 0.0
        else:
            rddr = true_positives / (true_positives + estimated_false_negatives)
        
        return {
            "RDDR": rddr,
            "true_positives": true_positives,
            "estimated_false_negatives": estimated_false_negatives,
            "total_alerts": len(recent_alerts),
            "validation_period_days": validation_period,
            "detection_latency_avg_ms": np.mean(self.detection_stats["response_time_ms"]) 
            if self.detection_stats["response_time_ms"] else 0.0
        }
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive detection statistics
        
        Returns:
            Dictionary of detection statistics
        """
        return {
            **self.detection_stats,
            "active_thresholds": len(self.thresholds),
            "recent_alerts": len(self.alerts),
            "regulatory_context": self.regulatory_context,
            "mean_response_time_ms": np.mean(self.detection_stats["response_time_ms"]) 
            if self.detection_stats["response_time_ms"] else 0.0,
            "p95_response_time_ms": np.percentile(self.detection_stats["response_time_ms"], 95) 
            if self.detection_stats["response_time_ms"] else 0.0
        }
    
    def graceful_degradation(self, error_code: int) -> Dict[str, Any]:
        """
        Implement graceful degradation based on error codes
        
        Args:
            error_code: Error code from error handling system
            
        Returns:
            Degradation mode configuration
        """
        degradation_modes = {
            0xE100: {  # Non-critical explanation failure
                "action": "continue",
                "degradation": "partial_explanation",
                "log_level": "WARNING",
                "alert": False
            },
            0xW200: {  # Warning drift alert
                "action": "degraded_mode",
                "degradation": "increased_monitoring",
                "log_level": "WARNING",
                "alert": True,
                "monitoring_frequency": "5min"  # Increase from 30min
            },
            0xC300: {  # Critical constraint violation
                "action": "halt_pipeline",
                "degradation": "full_stop",
                "log_level": "CRITICAL",
                "alert": True,
                "escalation": "immediate",
                "fallback": "human_review_queue"
            }
        }
        
        if error_code in degradation_modes:
            mode = degradation_modes[error_code]
            mode["error_description"] = self.error_codes.get(error_code, "Unknown error")
            return mode
        else:
            return {
                "action": "continue",
                "degradation": "none",
                "log_level": "INFO",
                "alert": False,
                "error": f"Unknown error code: {hex(error_code)}"
            }


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = ComplianceDriftDetector(
        regulatory_context="EU_AI_ACT_HIGH_RISK",
        alert_callback=lambda alert: print(f"Alert callback: {alert.message}")
    )
    
    # Simulate monitoring over time
    print("Simulating compliance monitoring over 50 iterations...")
    
    # Baseline metrics (good performance)
    baseline_metrics = {
        "accuracy": 0.88,
        "explainability_stability": 0.85,
        "fairness_disparity": 0.15,
        "inference_latency_ms": 450.0
    }
    
    alerts_history = []
    
    for i in range(50):
        # Simulate metric values with some noise and potential drift
        metrics = {}
        for metric, baseline in baseline_metrics.items():
            # Add random noise
            noise = np.random.normal(0, 0.02)
            
            # Simulate drift starting at iteration 30
            if i >= 30:
                if metric == "accuracy":
                    drift = -0.01 * (i - 30)  # Gradual accuracy decay
                elif metric == "fairness_disparity":
                    drift = 0.005 * (i - 30)  # Gradual fairness degradation
                else:
                    drift = 0
            else:
                drift = 0
            
            metrics[metric] = max(0.0, min(1.0, baseline + noise + drift))
        
        # Add some random latency spikes
        if i in [15, 35]:
            metrics["inference_latency_ms"] = 1200.0
        
        # Monitor metrics
        alerts = detector.monitor(
            metrics,
            metadata={
                "iteration": i,
                "model_version": "1.2.0",
                "environment": "production"
            }
        )
        
        if alerts:
            alerts_history.extend(alerts)
            for alert in alerts:
                print(f"Iteration {i}: {alert.severity.name} - {alert.message[:80]}...")
    
    # Calculate and display statistics
    print("\n" + "="*60)
    print("Compliance Drift Detection Statistics")
    print("="*60)
    
    stats = detector.get_detection_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:30}: {value:.4f}")
        else:
            print(f"{key:30}: {value}")
    
    # Calculate RDDR
    rddr_stats = detector.calculate_rddr(validation_period=7)  # 7-day window
    print(f"\nRDDR (7-day window): {rddr_stats['RDDR']:.3f}")
    print(f"True Positives: {rddr_stats['true_positives']}")
    print(f"Detection Latency: {rddr_stats['detection_latency_avg_ms']:.2f} ms")
    
    # Test graceful degradation
    print("\nGraceful Degradation Tests:")
    for error_code in [0xE100, 0xW200, 0xC300]:
        mode = detector.graceful_degradation(error_code)
        print(f"Error {hex(error_code)}: {mode['action']} - {mode.get('error_description', '')}")
