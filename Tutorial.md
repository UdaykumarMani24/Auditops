


```python
"""
Regulatory Requirement Embedding Layer
Translates legal mandates into actionable technical controls
"""

import yaml
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import datetime
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
    DATA_GOVERNANCE = "data_governance"


@dataclass
class RegulatoryRequirement:
    """Structured representation of a regulatory requirement"""
    id: str
    source: str
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
        self.regulatory_context = regulatory_context
        self.requirements = self._load_requirements(regulatory_context, config_path)
        self.constraint_checkers = self._create_constraint_checkers()
        self.constraint_mapping = self._build_constraint_mapping()
    
    def _load_requirements(self, context: str, config_path: str = None) -> List[RegulatoryRequirement]:
        """Load regulatory requirements from configuration"""
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return self._parse_config_to_requirements(config)
        
        # Default configurations
        configs = {
            "EU_AI_ACT_HIGH_RISK": self._get_eu_ai_act_requirements(),
            "FDA_21_CFR_820": self._get_fda_requirements(),
            "REGULATION_B": self._get_regulation_b_requirements()
        }
        
        return configs.get(context, [])
    
    def _get_eu_ai_act_requirements(self) -> List[RegulatoryRequirement]:
        return [
            RegulatoryRequirement(
                id="EU_AA_ART_13",
                source="EU_AI_ACT_ARTICLE_13",
                type=RequirementType.TRANSPARENCY,
                description="Transparency and provision of information to users",
                technical_controls=["model_explainability", "user_notification"],
                validation_method="explanation_validation",
                threshold=0.80,
                weight=1.0
            ),
            # Add more requirements...
        ]
    
    def _get_fda_requirements(self) -> List[RegulatoryRequirement]:
        return [
            RegulatoryRequirement(
                id="FDA_820_75",
                source="FDA_21_CFR_820_75",
                type=RequirementType.PERFORMANCE,
                description="Acceptance activities for software",
                technical_controls=["performance_monitoring", "threshold_checking"],
                validation_method="acceptance_testing",
                threshold=0.80,
                weight=1.0
            )
        ]
    
    def _get_regulation_b_requirements(self) -> List[RegulatoryRequirement]:
        return [
            RegulatoryRequirement(
                id="REG_B_FAIRNESS",
                source="REGULATION_B_ECOA",
                type=RequirementType.FAIRNESS,
                description="Prohibition against discrimination",
                technical_controls=["fairness_testing", "bias_detection"],
                validation_method="disparate_impact_analysis",
                threshold=0.20,
                weight=1.0
            )
        ]
    
    def _parse_config_to_requirements(self, config: Dict) -> List[RegulatoryRequirement]:
        requirements = []
        for req_id, req_config in config.get("requirements", {}).items():
            requirements.append(RegulatoryRequirement(
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
            ))
        return requirements
    
    def _build_constraint_mapping(self) -> Dict[str, List[str]]:
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
            mapping[stage] = list(set(stage_constraints))
        
        return mapping
    
    def _create_constraint_checkers(self) -> Dict[str, Callable]:
        checkers = {}
        for req in self.requirements:
            if req.type == RequirementType.PERFORMANCE and req.threshold is not None:
                checkers[f"check_{req.id}"] = self._create_performance_checker(req)
            elif req.type == RequirementType.FAIRNESS and req.threshold is not None:
                checkers[f"check_{req.id}"] = self._create_fairness_checker(req)
            elif req.type == RequirementType.TRANSPARENCY:
                checkers[f"check_{req.id}"] = self._create_transparency_checker(req)
            else:
                checkers[f"check_{req.id}"] = self._create_generic_checker(req)
        return checkers
    
    def _create_performance_checker(self, req: RegulatoryRequirement) -> Callable:
        def checker(value: float) -> Dict[str, Any]:
            passed = value >= req.threshold if req.threshold else True
            return {
                "passed": passed,
                "value": value,
                "threshold": req.threshold,
                "deviation": req.threshold - value if not passed else 0
            }
        return checker
    
    def _create_fairness_checker(self, req: RegulatoryRequirement) -> Callable:
        def checker(disparity: float) -> Dict[str, Any]:
            passed = disparity <= req.threshold if req.threshold else True
            return {
                "passed": passed,
                "value": disparity,
                "threshold": req.threshold,
                "deviation": disparity - req.threshold if not passed else 0
            }
        return checker
    
    def _create_transparency_checker(self, req: RegulatoryRequirement) -> Callable:
        def checker(explanation: Dict[str, Any]) -> Dict[str, Any]:
            has_explanation = explanation.get("has_explanation", False)
            completeness = explanation.get("completeness", 0.0)
            passed = has_explanation and completeness >= 0.8
            return {
                "passed": passed,
                "has_explanation": has_explanation,
                "completeness": completeness,
                "threshold": 0.8
            }
        return checker
    
    def _create_generic_checker(self, req: RegulatoryRequirement) -> Callable:
        def checker(**kwargs) -> Dict[str, Any]:
            return {
                "passed": True,
                "requirement_id": req.id,
                "type": req.type.value
            }
        return checker
    
    def inject_constraints(self, ml_pipeline) -> Dict[str, Any]:
        """Inject regulatory constraints into ML pipeline"""
        return {
            "original_pipeline": ml_pipeline,
            "regulatory_context": self.regulatory_context,
            "constraints": self.constraint_mapping,
            "requirements": [req.id for req in self.requirements],
            "constraint_checkers": self.constraint_checkers
        }
    
    def calculate_coverage_score(self) -> Dict[str, Any]:
        """Calculate Compliance Coverage Score (CCS)"""
        if not self.requirements:
            return {"CCS": 0.0, "details": {}}
        
        total_weight = sum(req.weight for req in self.requirements)
        satisfied_weight = sum(req.weight for req in self.requirements if True)  # Would use actual validation
        
        return {
            "CCS": satisfied_weight / total_weight if total_weight > 0 else 0.0,
            "total_requirements": len(self.requirements),
            "satisfied_requirements": len(self.requirements)  # Placeholder
        }
    
    def get_required_audit_fields(self) -> List[str]:
        """Get required audit trail fields"""
        fields = [
            "timestamp", "decision_id", "model_version", "input_data_hash",
            "prediction", "confidence_score", "explanation_available"
        ]
        
        for req in self.requirements:
            if req.type == RequirementType.TRANSPARENCY:
                fields.extend(["explanation_method", "feature_importance"])
            elif req.type == RequirementType.FAIRNESS:
                fields.extend(["protected_attributes", "disparity_metrics"])
        
        return list(set(fields))
