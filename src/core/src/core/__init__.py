"""
AuditOps: Continuous Compliance Framework for MLOps
Core modules for regulatory compliance integration
"""

__version__ = "1.0.0"
__author__ = "Udayakumar Mani, Senthilkumar Rathinasamy"
__email__ = "uthay@bioinfo.sastra.edu"

from .audit_trail_generator import AuditTrailGenerator, AuditEventType
from .compliance_drift import (
    ComplianceDriftDetector, 
    DriftSeverity, 
    AlertCode,
    PageHinkleyDetector
)
from .regulatory_embedding import RegulatoryEmbeddingLayer
from .explainability_preservation import ExplainabilityPreservationEngine

# Main framework class
class AuditOpsFramework:
    """
    Main AuditOps framework orchestrator
    """
    def __init__(self, regulatory_context: str = "EU_AI_ACT_HIGH_RISK"):
        """
        Initialize AuditOps framework with regulatory context
        
        Args:
            regulatory_context: Regulatory framework to enforce
        """
        self.regulatory_context = regulatory_context
        self.reg_layer = RegulatoryEmbeddingLayer(regulatory_context)
        self.exp_engine = ExplainabilityPreservationEngine()
        self.audit_system = AuditTrailGenerator()
        self.drift_detector = ComplianceDriftDetector(regulatory_context)
    
    def process_pipeline(self, ml_pipeline):
        """
        Transform standard ML pipeline to compliance-aware pipeline
        
        Args:
            ml_pipeline: Standard ML pipeline object
            
        Returns:
            tuple: (enhanced_pipeline, audit_trail_system)
        """
        # Step 1: Inject regulatory constraints
        constrained_pipeline = self.reg_layer.inject_constraints(ml_pipeline)
        
        # Step 2: Apply explainability preservation
        explainable_pipeline = self.exp_engine.apply(constrained_pipeline)
        
        # Step 3: Generate audit trails
        audit_trails = self.audit_system.generate(explainable_pipeline)
        
        # Step 4: Monitor for compliance drift
        self.drift_detector.monitor(explainable_pipeline, audit_trails)
        
        return explainable_pipeline, audit_trails
    
    def get_compliance_metrics(self):
        """
        Calculate comprehensive compliance metrics
        
        Returns:
            dict: Compliance metrics including CCS, EPI, ATCM, RDDR
        """
        # Calculate ATCM from audit system
        atcm_result = self.audit_system.get_completeness_metric(
            required_fields=self.reg_layer.get_required_audit_fields()
        )
        
        # Calculate RDDR from drift detector
        rddr_result = self.drift_detector.calculate_rddr()
        
        # Calculate EPI from explainability engine
        epi_result = self.exp_engine.calculate_epi()
        
        # Calculate CCS from regulatory layer
        ccs_result = self.reg_layer.calculate_coverage_score()
        
        return {
            "CCS": ccs_result,
            "EPI": epi_result,
            "ATCM": atcm_result["ATCM"],
            "RDDR": rddr_result["RDDR"],
            "details": {
                "audit_completeness": atcm_result,
                "drift_detection": rddr_result,
                "explainability_stability": epi_result,
                "regulatory_coverage": ccs_result
            }
        }

__all__ = [
    "AuditOpsFramework",
    "AuditTrailGenerator",
    "ComplianceDriftDetector",
    "RegulatoryEmbeddingLayer",
    "ExplainabilityPreservationEngine",
    "AuditEventType",
    "DriftSeverity",
    "AlertCode"
]
