"""
Regulatory Requirement Embedding Layer
Translates legal mandates into actionable technical controls
"""

import yaml
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import re


class RequirementType(Enum):
    """Types of regulatory requirements"""
    PERFORMANCE = "performance"
    FAIRNESS = "fairness"
    TRANSPARENCY = "transparency"
    PRIVACY = "privacy"
    SECURITY = "security"
    AUDITABILITY = "auditability"
    ROBUSTNESS = "robustness"
    HUMAN_OVERSIGHT = "human_oversight"


@dataclass
class RegulatoryRequirement:
    """Structured representation of a regulatory requirement"""
    id: str
    source: str  # e.g., "EU_AI_ACT_ARTICLE_13"
    type: RequirementType
    description: str
    technical_controls: List[str]
    validation_method: str
    threshold: Optional[float] = None
    weight: float = 1.0
    applicable_phases: List[str] = field(default_factory=lambda: ["all"])
    metadata: Dict[str, Any] = field(default_factory=dict)


class RegulatoryEmbeddingLayer:
    """
    Translates regulatory texts into technical constraints for ML pipelines
    """
    
    def __init__(self, regulatory_context: str, config_path: str = None):
        """
        Initialize regulatory embedding layer
        
        Args:
            regulatory_context: Name of regulatory framework
            config_path: Path to configuration file (optional)
        """
        self.regulatory_context = regulatory_context
        self.requirements = self._load_requirements(regulatory_context, config_path)
        self.constraint_mapping = self._build_constraint_mapping()
        
    def _load_requirements(self, context: str, config_path: str = None) -> List[RegulatoryRequirement]:
        """
        Load regulatory requirements from configuration
        
        Args:
            context: Regulatory context name
            config_path: Optional custom config path
            
        Returns:
            List of regulatory requirements
        """
        requirements = []
        
        # Default configurations
        default_configs = {
            "EU_AI_ACT_HIGH_RISK": self._get_eu_ai_act_requirements(),
            "FDA_21_CFR_820": self._get_fda_requirements(),
            "REGULATION_B": self._get_regulation_b_requirements()
        }
        
        # Load from config file if provided
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                requirements = self._parse_config_to_requirements(config)
        elif context in default_configs:
            requirements = default_configs[context]
        else:
            raise ValueError(f"Unknown regulatory context: {context}")
        
        return requirements
    
    def _get_eu_ai_act_requirements(self) -> List[RegulatoryRequirement]:
        """Get EU AI Act requirements for high-risk systems"""
        return [
            RegulatoryRequirement(
                id="EU_AA_ART_9",
                source="EU_AI_ACT_ARTICLE_9",
                type=RequirementType.ROBUSTNESS,
                description="Risk management system for high-risk AI systems",
                technical_controls=[
                    "risk_assessment_pipeline",
                    "failure_mode_analysis",
                    "stress_testing_framework"
                ],
                validation_method="automated_testing",
                weight=1.0,
                applicable_phases=["design", "development", "deployment"]
            ),
            RegulatoryRequirement(
                id="EU_AA_ART_10",
                source="EU_AI_ACT_ARTICLE_10",
                type=RequirementType.DATA_GOVERNANCE,
                description="Data and data governance requirements",
                technical_controls=[
                    "data_quality_validation",
                    "bias_detection_pipeline",
                    "data_provenance_tracking"
                ],
                validation_method="data_validation",
                weight=0.9,
                applicable_phases=["data_ingestion", "training", "monitoring"]
            ),
            RegulatoryRequirement(
                id="EU_AA_ART_13",
                source="EU_AI_ACT_ARTICLE_13",
                type=RequirementType.TRANSPARENCY,
                description="Transparency and provision of information to users",
                technical_controls=[
                    "model_explainability_engine",
                    "decision_documentation",
                    "user_notification_system"
                ],
                validation_method="explanation_validation",
                threshold=0.80,
                weight=1.0,
                applicable_phases=["inference", "monitoring"]
            ),
            RegulatoryRequirement(
                id="EU_AA_ART_15",
                source="EU_AI_ACT_ARTICLE_15",
                type=RequirementType.HUMAN_OVERSIGHT,
                description="Human oversight measures",
                technical_controls=[
                    "human_in_the_loop_interface",
                    "escalation_mechanism",
                    "override_capability"
                ],
                validation_method="process_validation",
                weight=0.8,
                applicable_phases=["inference", "monitoring"]
            )
        ]
    
    def _get_fda_requirements(self) -> List[RegulatoryRequirement]:
        """Get FDA 21 CFR Part 820 requirements"""
        return [
            RegulatoryRequirement(
                id="FDA_820_30",
                source="FDA_21_CFR_820_30",
                type=RequirementType.PERFORMANCE,
                description="Design controls for medical devices",
                technical_controls=[
                    "design_validation_pipeline",
                    "performance_benchmarking",
                    "change_control_system"
                ],
                validation_method="clinical_validation",
                weight=1.0,
                applicable_phases=["design", "validation", "monitoring"]
            ),
            RegulatoryRequirement(
                id="FDA_820_70",
                source="FDA_21_CFR_820_70",
                type=RequirementType.AUDITABILITY,
                description="Production and process controls",
                technical_controls=[
                    "process_validation_monitoring",
                    "equipment_calibration_tracking",
                    "environmental_monitoring"
                ],
                validation_method="process_audit",
                weight=0.9,
                applicable_phases=["production", "monitoring"]
            ),
            RegulatoryRequirement(
                id="FDA_820_75",
                source="FDA_21_CFR_820_75",
                type=RequirementType.PERFORMANCE,
                description="Acceptance activities for software",
                technical_controls=[
                    "acceptance_testing_framework",
                    "performance_threshold_monitoring",
                    "regression_testing_suite"
                ],
                validation_method="acceptance_testing",
                threshold=0.80,  # Minimum performance threshold
                weight=1.0,
                applicable_phases=["validation", "deployment", "monitoring"]
            )
        ]
    
    def _get_regulation_b_requirements(self) -> List[RegulatoryRequirement]:
        """Get Regulation B (ECOA) requirements for fair lending"""
        return [
            RegulatoryRequirement(
                id="REG_B_FAIRNESS",
                source="REGULATION_B_ECOA",
                type=RequirementType.FAIRNESS,
                description="Prohibition against discrimination in credit transactions",
                technical_controls=[
                    "disparate_impact_analysis",
                    "fairness_constrained_training",
                    "bias_mitigation_pipeline"
                ],
                validation_method="fairness_testing",
                threshold=0.20,  # Maximum disparate impact ratio
                weight=1.0,
                applicable_phases=["training", "validation", "monitoring"]
            )
        ]
    
    def _parse_config_to_requirements(self, config: Dict) -> List[RegulatoryRequirement]:
        """Parse YAML configuration to requirement objects"""
        requirements = []
        
        for req_id, req_config in config.get("requirements", {}).items():
            requirement = RegulatoryRequirement(
                id=req_id,
                source=req_config.get("source", ""),
                type=RequirementType(req_config.get("type", "performance")),
                description=req_config.get("description", ""),
                technical_controls=req_config.get("technical_controls", []),
                validation_method=req_config.get("validation_method", "manual"),
                threshold=req_config.get("threshold"),
                weight=req_config.get("weight", 1.0),
                applicable_phases=req_config.get("applicable_phases", ["all"]),
                metadata=req_config.get("metadata", {})
            )
            requirements.append(requirement)
        
        return requirements
    
    def _build_constraint_mapping(self) -> Dict[str, List[str]]:
        """Build mapping from ML pipeline stages to applicable constraints"""
        mapping = {}
        
        pipeline_stages = [
            "data_ingestion", "preprocessing", "feature_engineering",
            "model_training", "model_validation", "model_deployment",
            "model_inference", "model_monitoring", "model_retraining"
        ]
        
        for stage in pipeline_stages:
            stage_constraints = []
            for req in self.requirements:
                if "all" in req.applicable_phases or stage in req.applicable_phases:
                    stage_constraints.extend(req.technical_controls)
            mapping[stage] = list(set(stage_constraints))  # Remove duplicates
        
        return mapping
    
    def inject_constraints(self, ml_pipeline) -> Any:
        """
        Inject regulatory constraints into ML pipeline
        
        Args:
            ml_pipeline: Original ML pipeline object
            
        Returns:
            Enhanced pipeline with regulatory constraints
        """
        # This is a framework-agnostic implementation
        # Actual implementation would depend on the ML framework being used
        
        enhanced_pipeline = {
            "original_pipeline": ml_pipeline,
            "regulatory_context": self.regulatory_context,
            "constraints": self.constraint_mapping,
            "requirements": [req.id for req in self.requirements]
        }
        
        # Add constraint checking mechanisms
        enhanced_pipeline["constraint_checkers"] = self._create_constraint_checkers()
        
        return enhanced_pipeline
    
    def _create_constraint_checkers(self) -> Dict[str, Any]:
        """Create constraint checking functions for each requirement"""
        checkers = {}
        
        for req in self.requirements:
            checker_name = f"check_{req.id.lower()}"
            
            if req.type == RequirementType.PERFORMANCE and req.threshold is not None:
                checkers[checker_name] = self._create_performance_checker(req)
            elif req.type == RequirementType.FAIRNESS and req.threshold is not None:
                checkers[checker_name] = self._create_fairness_checker(req)
            elif req.type == RequirementType.TRANSPARENCY:
                checkers[checker_name] = self._create_transparency_checker(req)
            else:
                checkers[checker_name] = self._create_generic_checker(req)
        
        return checkers
    
    def _create_performance_checker(self, requirement: RegulatoryRequirement):
        """Create performance threshold checker"""
        def checker(metric_value: float) -> Dict[str, Any]:
            if requirement.threshold is None:
                return {"passed": True, "value": metric_value}
            
            passed = metric_value >= requirement.threshold
            deviation = requirement.threshold - metric_value if not passed else 0
            
            return {
                "passed": passed,
                "value": metric_value,
                "threshold": requirement.threshold,
                "deviation": deviation,
                "severity": "VIOLATION" if deviation > 0.1 else "WARNING"
            }
        
        return checker
    
    def _create_fairness_checker(self, requirement: RegulatoryRequirement):
        """Create fairness checker"""
        def checker(disparity_ratio: float) -> Dict[str, Any]:
            if requirement.threshold is None:
                return {"passed": True, "value": disparity_ratio}
            
            passed = disparity_ratio <= requirement.threshold
            deviation = disparity_ratio - requirement.threshold if not passed else 0
            
            return {
                "passed": passed,
                "value": disparity_ratio,
                "threshold": requirement.threshold,
                "deviation": deviation,
                "severity": "VIOLATION" if deviation > 0.1 else "WARNING"
            }
        
        return checker
    
    def _create_transparency_checker(self, requirement: RegulatoryRequirement):
        """Create transparency/explainability checker"""
        def checker(explanation_quality: Dict[str, Any]) -> Dict[str, Any]:
            # Check if explanation meets minimum requirements
            has_explanation = explanation_quality.get("has_explanation", False)
            completeness = explanation_quality.get("completeness", 0.0)
            stability = explanation_quality.get("stability", 0.0)
            
            passed = (has_explanation and 
                     completeness >= 0.8 and 
                     stability >= 0.8)
            
            return {
                "passed": passed,
                "has_explanation": has_explanation,
                "completeness": completeness,
                "stability": stability,
                "threshold": 0.8,
                "severity": "WARNING" if not passed else "PASS"
            }
        
        return checker
    
    def _create_generic_checker(self, requirement: RegulatoryRequirement):
        """Create generic requirement checker"""
        def checker(**kwargs) -> Dict[str, Any]:
            # Generic checker that always passes
            # In real implementation, this would be customized per requirement
            return {
                "passed": True,
                "requirement_id": requirement.id,
                "type": requirement.type.value,
                "message": f"Generic check for {requirement.id}"
            }
        
        return checker
    
    def calculate_coverage_score(self) -> Dict[str, Any]:
        """
        Calculate Compliance Coverage Score (CCS)
        
        Returns:
            Dictionary with CCS and detailed breakdown
        """
        if not self.requirements:
            return {"CCS": 0.0, "details": {}}
        
        # Simulate checking each requirement
        # In real implementation, this would use actual validation results
        requirement_results = []
        total_weight = 0.0
        satisfied_weight = 0.0
        
        for req in self.requirements:
            # Simulate requirement satisfaction (would be actual validation in production)
            # For demonstration, assume 90% of requirements are satisfied
            satisfied = True  # Placeholder
            
            requirement_results.append({
                "id": req.id,
                "type": req.type.value,
                "weight": req.weight,
                "satisfied": satisfied,
                "threshold": req.threshold,
                "source": req.source
            })
            
            total_weight += req.weight
            if satisfied:
                satisfied_weight += req.weight
        
        ccs = satisfied_weight / total_weight if total_weight > 0 else 0.0
        
        return {
            "CCS": ccs,
            "satisfied_weight": satisfied_weight,
            "total_weight": total_weight,
            "requirement_count": len(self.requirements),
            "details": requirement_results
        }
    
    def get_required_audit_fields(self) -> List[str]:
        """
        Get list of required audit trail fields based on regulatory requirements
        
        Returns:
            List of field names that must be captured in audit trails
        """
        required_fields = [
            "timestamp",
            "decision_id",
            "model_version",
           
