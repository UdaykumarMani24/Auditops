"""
Audit Trail Generator for AuditOps Framework
Implements W3C PROV-O compliant provenance tracking
"""

import uuid
import json
import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib


class AuditEventType(Enum):
    """Types of audit events"""
    DATA_INGESTION = "data_ingestion"
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    MODEL_VALIDATION = "model_validation"
    COMPLIANCE_CHECK = "compliance_check"
    HUMAN_REVIEW = "human_review"
    SYSTEM_ALERT = "system_alert"


@dataclass
class ProvenanceEntity:
    """W3C PROV-O Entity representation"""
    id: str
    type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())


@dataclass
class ProvenanceActivity:
    """W3C PROV-O Activity representation"""
    id: str
    type: str
    started_at: str
    ended_at: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvenanceAgent:
    """W3C PROV-O Agent representation"""
    id: str
    type: str  # 'system', 'user', 'automated_process'
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)


class AuditTrailGenerator:
    """
    Generates comprehensive audit trails for ML decisions
    """
    
    def __init__(self, 
                 storage_backend: str = "json",  # 'json', 'database', 's3'
                 min_log_level: str = "INFO"):
        """
        Initialize the audit trail generator
        
        Args:
            storage_backend: Where to store audit trails
            min_log_level: Minimum log level to capture
        """
        self.storage_backend = storage_backend
        self.min_log_level = min_log_level
        self.trail_id = self._generate_trail_id()
        self.entities = {}
        self.activities = {}
        self.agents = {}
        self.relations = []
        
        # Initialize with system agent
        self.system_agent = self._create_agent(
            agent_id="system:auditops",
            agent_type="system",
            name="AuditOps Framework",
            attributes={"version": "1.0.0"}
        )
    
    def _generate_trail_id(self) -> str:
        """Generate unique trail ID"""
        return f"trail_{uuid.uuid4().hex[:8]}_{int(datetime.datetime.utcnow().timestamp())}"
    
    def _create_entity(self, 
                      entity_type: str, 
                      attributes: Dict[str, Any]) -> ProvenanceEntity:
        """Create a provenance entity"""
        entity_id = f"entity:{entity_type}:{uuid.uuid4().hex[:8]}"
        entity = ProvenanceEntity(
            id=entity_id,
            type=entity_type,
            attributes=attributes,
            timestamp=datetime.datetime.utcnow().isoformat()
        )
        self.entities[entity_id] = entity
        return entity
    
    def _create_activity(self,
                        activity_type: str,
                        attributes: Dict[str, Any]) -> ProvenanceActivity:
        """Create a provenance activity"""
        activity_id = f"activity:{activity_type}:{uuid.uuid4().hex[:8]}"
        activity = ProvenanceActivity(
            id=activity_id,
            type=activity_type,
            started_at=datetime.datetime.utcnow().isoformat(),
            attributes=attributes
        )
        self.activities[activity_id] = activity
        return activity
    
    def _create_agent(self,
                     agent_id: str,
                     agent_type: str,
                     name: str,
                     attributes: Dict[str, Any]) -> ProvenanceAgent:
        """Create a provenance agent"""
        agent = ProvenanceAgent(
            id=agent_id,
            type=agent_type,
            name=name,
            attributes=attributes
        )
        self.agents[agent_id] = agent
        return agent
    
    def _record_relation(self,
                        relation_type: str,
                        source_id: str,
                        target_id: str,
                        attributes: Dict[str, Any] = None):
        """Record a provenance relation"""
        if attributes is None:
            attributes = {}
        
        relation = {
            "id": f"relation:{uuid.uuid4().hex[:8]}",
            "type": relation_type,
            "source": source_id,
            "target": target_id,
            "attributes": attributes,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        self.relations.append(relation)
    
    def log_data_ingestion(self,
                          dataset_id: str,
                          dataset_info: Dict[str, Any],
                          hash_value: str) -> str:
        """
        Log data ingestion event
        
        Args:
            dataset_id: Unique identifier for the dataset
            dataset_info: Metadata about the dataset
            hash_value: Hash of the dataset for integrity
            
        Returns:
            Entity ID for the dataset
        """
        # Create dataset entity
        dataset_entity = self._create_entity(
            entity_type="dataset",
            attributes={
                "dataset_id": dataset_id,
                "hash": hash_value,
                "record_count": dataset_info.get("record_count"),
                "features": dataset_info.get("features"),
                "source": dataset_info.get("source"),
                "ingestion_timestamp": datetime.datetime.utcnow().isoformat()
            }
        )
        
        # Create ingestion activity
        ingestion_activity = self._create_activity(
            activity_type="data_ingestion",
            attributes={
                "method": dataset_info.get("ingestion_method"),
                "validation_status": dataset_info.get("validation_status"),
                "compliance_checks_passed": dataset_info.get("compliance_checks", [])
            }
        )
        
        # Record relation: wasGeneratedBy
        self._record_relation(
            relation_type="wasGeneratedBy",
            source_id=dataset_entity.id,
            target_id=ingestion_activity.id,
            attributes={"role": "output"}
        )
        
        # Record relation: wasAssociatedWith
        self._record_relation(
            relation_type="wasAssociatedWith",
            source_id=ingestion_activity.id,
            target_id=self.system_agent.id,
            attributes={"role": "executor"}
        )
        
        return dataset_entity.id
    
    def log_model_training(self,
                          model_id: str,
                          model_info: Dict[str, Any],
                          input_entity_ids: List[str]) -> str:
        """
        Log model training event
        
        Args:
            model_id: Unique identifier for the model
            model_info: Metadata about the model
            input_entity_ids: IDs of input datasets/entities
            
        Returns:
            Entity ID for the trained model
        """
        # Create model entity
        model_entity = self._create_entity(
            entity_type="model",
            attributes={
                "model_id": model_id,
                "model_type": model_info.get("model_type"),
                "hyperparameters": model_info.get("hyperparameters"),
                "training_metrics": model_info.get("training_metrics"),
                "version": model_info.get("version", "1.0.0"),
                "hash": self._calculate_model_hash(model_info),
                "training_timestamp": datetime.datetime.utcnow().isoformat()
            }
        )
        
        # Create training activity
        training_activity = self._create_activity(
            activity_type="model_training",
            attributes={
                "algorithm": model_info.get("algorithm"),
                "training_duration": model_info.get("training_duration"),
                "hardware_used": model_info.get("hardware_info", {}),
                "compliance_checks": model_info.get("compliance_checks", [])
            }
        )
        
        # Record relations for inputs
        for input_id in input_entity_ids:
            self._record_relation(
                relation_type="used",
                source_id=training_activity.id,
                target_id=input_id,
                attributes={"role": "input"}
            )
        
        # Record relation for output
        self._record_relation(
            relation_type="wasGeneratedBy",
            source_id=model_entity.id,
            target_id=training_activity.id,
            attributes={"role": "output"}
        )
        
        # Record agent relation
        self._record_relation(
            relation_type="wasAssociatedWith",
            source_id=training_activity.id,
            target_id=self.system_agent.id,
            attributes={"role": "trainer"}
        )
        
        return model_entity.id
    
    def log_inference(self,
                     inference_id: str,
                     model_entity_id: str,
                     input_data: Dict[str, Any],
                     prediction: Any,
                     explanation: Dict[str, Any] = None) -> str:
        """
        Log model inference event
        
        Args:
            inference_id: Unique identifier for the inference
            model_entity_id: ID of the model entity used
            input_data: Input data for inference
            prediction: Model prediction/output
            explanation: Explanation of the prediction
            
        Returns:
            Entity ID for the inference result
        """
        # Create inference entity
        inference_entity = self._create_entity(
            entity_type="inference",
            attributes={
                "inference_id": inference_id,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "input_hash": hashlib.sha256(
                    str(input_data).encode()
                ).hexdigest()[:16],
                "prediction": prediction,
                "confidence": getattr(prediction, 'confidence', None) 
                if hasattr(prediction, 'confidence') else None,
                "explanation_summary": self._summarize_explanation(explanation),
                "environment": {
                    "hostname": "localhost",  # Would be actual hostname in prod
                    "python_version": "3.9.0",
                    "framework_versions": {
                        "scikit-learn": "1.0.0",
                        "auditops": "1.0.0"
                    }
                }
            }
        )
        
        # Create inference activity
        inference_activity = self._create_activity(
            activity_type="model_inference",
            attributes={
                "latency_ms": input_data.get("latency_ms", 0),
                "batch_size": len(input_data) if isinstance(input_data, list) else 1,
                "compliance_checks": input_data.get("compliance_checks", [])
            }
        )
        
        # Record relations
        self._record_relation(
            relation_type="used",
            source_id=inference_activity.id,
            target_id=model_entity_id,
            attributes={"role": "model"}
        )
        
        self._record_relation(
            relation_type="wasGeneratedBy",
            source_id=inference_entity.id,
            target_id=inference_activity.id,
            attributes={"role": "output"}
        )
        
        # Record agent relation
        self._record_relation(
            relation_type="wasAssociatedWith",
            source_id=inference_activity.id,
            target_id=self.system_agent.id,
            attributes={"role": "inference_engine"}
        )
        
        return inference_entity.id
    
    def log_compliance_check(self,
                           check_type: str,
                           check_result: Dict[str, Any],
                           related_entity_ids: List[str]) -> str:
        """
        Log compliance check event
        
        Args:
            check_type: Type of compliance check
            check_result: Result of the check
            related_entity_ids: IDs of related entities
            
        Returns:
            Activity ID for the compliance check
        """
        # Create compliance check activity
        compliance_activity = self._create_activity(
            activity_type="compliance_check",
            attributes={
                "check_type": check_type,
                "result": check_result.get("result"),
                "details": check_result.get("details"),
                "threshold": check_result.get("threshold"),
                "actual_value": check_result.get("actual_value"),
                "timestamp": datetime.datetime.utcnow().isoformat()
            }
        )
        
        # Record relations to related entities
        for entity_id in related_entity_ids:
            self._record_relation(
                relation_type="wasInformedBy",
                source_id=compliance_activity.id,
                target_id=entity_id,
                attributes={"informational_role": "checked_entity"}
            )
        
        # Record agent relation
        self._record_relation(
            relation_type="wasAssociatedWith",
            source_id=compliance_activity.id,
            target_id=self.system_agent.id,
            attributes={"role": "compliance_checker"}
        )
        
        # Create alert entity if check failed
        if check_result.get("result") == "FAILED":
            alert_entity = self._create_entity(
                entity_type="alert",
                attributes={
                    "severity": check_result.get("severity", "WARNING"),
                    "message": check_result.get("message", "Compliance check failed"),
                    "recommended_action": check_result.get("recommended_action"),
                    "acknowledged": False
                }
            )
            
            self._record_relation(
                relation_type="wasGeneratedBy",
                source_id=alert_entity.id,
                target_id=compliance_activity.id,
                attributes={"role": "output"}
            )
        
        return compliance_activity.id
    
    def _calculate_model_hash(self, model_info: Dict[str, Any]) -> str:
        """Calculate hash for model identification"""
        hash_input = f"{model_info.get('model_type', '')}" \
                    f"{model_info.get('hyperparameters', {})}" \
                    f"{model_info.get('training_metrics', {})}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:32]
    
    def _summarize_explanation(self, explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize explanation for audit trail"""
        if not explanation:
            return {"available": False}
        
        summary = {
            "available": True,
            "method": explanation.get("method", "unknown"),
            "top_features": explanation.get("top_features", []),
            "confidence": explanation.get("confidence"),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        
        # Limit feature details for storage efficiency
        if "feature_importances" in explanation:
            feature_importances = explanation["feature_importances"]
            if isinstance(feature_importances, dict) and len(feature_importances) > 10:
                summary["feature_count"] = len(feature_importances)
                # Keep only top 10 features in summary
                sorted_features = sorted(
                    feature_importances.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:10]
                summary["top_feature_importances"] = dict(sorted_features)
            else:
                summary["feature_importances"] = feature_importances
        
        return summary
    
    def export_trail(self, format: str = "json") -> Dict[str, Any]:
        """
        Export the complete audit trail
        
        Args:
            format: Export format ('json', 'prov-o')
            
        Returns:
            Serialized audit trail
        """
        if format == "prov-o":
            return self._export_prov_o()
        else:
            return self._export_json()
    
    def _export_json(self) -> Dict[str, Any]:
        """Export trail as JSON-serializable dict"""
        return {
            "trail_id": self.trail_id,
            "generated_at": datetime.datetime.utcnow().isoformat(),
            "entities": {k: asdict(v) for k, v in self.entities.items()},
            "activities": {k: asdict(v) for k, v in self.activities.items()},
            "agents": {k: asdict(v) for k, v in self.agents.items()},
            "relations": self.relations,
            "metadata": {
                "framework": "AuditOps",
                "version": "1.0.0",
                "export_format": "auditops_v1"
            }
        }
    
    def _export_prov_o(self) -> Dict[str, Any]:
        """Export trail as W3C PROV-O compliant JSON-LD"""
        prov_o = {
            "@context": {
                "prov": "http://www.w3.org/ns/prov#",
                "auditops": "https://auditops.org/ns#"
            },
            "@graph": []
        }
        
        # Convert entities to PROV-O
        for entity in self.entities.values():
            prov_o["@graph"].append({
                "@id": entity.id,
                "@type": ["prov:Entity", f"auditops:{entity.type}"],
                "prov:generatedAtTime": entity.timestamp,
                **entity.attributes
            })
        
        # Convert activities to PROV-O
        for activity in self.activities.values():
            activity_node = {
                "@id": activity.id,
                "@type": ["prov:Activity", f"auditops:{activity.type}"],
                "prov:startedAtTime": activity.started_at,
            }
            if activity.ended_at:
                activity_node["prov:endedAtTime"] = activity.ended_at
            activity_node.update(activity.attributes)
            prov_o["@graph"].append(activity_node)
        
        # Convert agents to PROV-O
        for agent in self.agents.values():
            prov_o["@graph"].append({
                "@id": agent.id,
                "@type": ["prov:Agent", f"auditops:{agent.type}"],
                "prov:label": agent.name,
                **agent.attributes
            })
        
        # Convert relations to PROV-O
        for relation in self.relations:
            prov_o["@graph"].append({
                "@id": relation["id"],
                "@type": f"prov:{relation['type']}",
                "prov:qualifiedAssociation": {
                    "@type": "prov:Association",
                    "prov:agent": {"@id": relation["target"]}
                    if relation["type"] == "wasAssociatedWith"
                    else {"@id": relation["source"]},
                    "prov:hadRole": relation["attributes"].get("role", "")
                },
                "prov:atTime": relation["timestamp"]
            })
        
        return prov_o
    
    def get_completeness_metric(self, required_fields: List[str]) -> Dict[str, Any]:
        """
        Calculate ATCM (Audit Trail Completeness Metric)
        
        Args:
            required_fields: List of field names that should be captured
            
        Returns:
            Dictionary with ATCM score and details
        """
        total_fields = len(required_fields)
        captured_fields = 0
        missing_fields = []
        
        # Check each inference entity for required fields
        for entity_id, entity in self.entities.items():
            if entity.type == "inference":
                for field in required_fields:
                    if field in entity.attributes:
                        captured_fields += 1
                    else:
                        missing_fields.append(field)
        
        if total_fields == 0:
            atcm = 1.0
        else:
            atcm = captured_fields / (len(self.entities.get("inference", [])) * total_fields 
                                     if "inference" in str(self.entities) else total_fields)
        
        return {
            "ATCM": atcm,
            "captured_fields": captured_fields,
            "total_expected": total_fields * len([e for e in self.entities.values() 
                                                 if e.type == "inference"]),
            "missing_fields": list(set(missing_fields)),
            "entity_count": len(self.entities),
            "activity_count": len(self.activities)
        }


# Example usage
if __name__ == "__main__":
    # Initialize audit trail generator
    audit = AuditTrailGenerator()
    
    # Log data ingestion
    dataset_info = {
        "record_count": 1000,
        "features": ["age", "sex", "cp", "trestbps"],
        "source": "UCI Heart Disease",
        "ingestion_method": "batch_csv",
        "validation_status": "validated",
        "compliance_checks": ["pii_removed", "format_validated"]
    }
    
    dataset_entity_id = audit.log_data_ingestion(
        dataset_id="heart_disease_v1",
        dataset_info=dataset_info,
        hash_value="abc123def456"
    )
    
    # Log model training
    model_info = {
        "model_type": "RandomForest",
        "hyperparameters": {"n_estimators": 100, "max_depth": 10},
        "training_metrics": {"accuracy": 0.85, "auc": 0.91},
        "algorithm": "RandomForestClassifier",
        "training_duration": 120.5,
        "hardware_info": {"cpus": 4, "memory_gb": 16},
        "compliance_checks": ["fairness_check", "bias_audit"]
    }
    
    model_entity_id = audit.log_model_training(
        model_id="rf_heart_disease_v1",
        model_info=model_info,
        input_entity_ids=[dataset_entity_id]
    )
    
    # Log inference
    inference_data = {
        "patient_id": "P001",
        "age": 55,
        "sex": 1,
        "cp": 3,
        "trestbps": 130,
        "latency_ms": 45,
        "compliance_checks": ["real_time_monitoring"]
    }
    
    inference_entity_id = audit.log_inference(
        inference_id="inf_001",
        model_entity_id=model_entity_id,
        input_data=inference_data,
        prediction={"class": 1, "confidence": 0.87, "risk_score": 0.72},
        explanation={
            "method": "SHAP",
            "feature_importances": {"age": 0.35, "cp": 0.28, "trestbps": 0.22, "sex": 0.15},
            "confidence": 0.92
        }
    )
    
    # Log compliance check
    check_result = {
        "result": "PASSED",
        "details": "All thresholds met",
        "threshold": 0.70,
        "actual_value": 0.87,
        "severity": "INFO"
    }
    
    audit.log_compliance_check(
        check_type="performance_threshold",
        check_result=check_result,
        related_entity_ids=[inference_entity_id]
    )
    
    # Export and analyze
    trail = audit.export_trail()
    
    # Calculate ATCM
    required_fields = ["timestamp", "prediction", "explanation_summary", 
                      "input_hash", "environment"]
    completeness = audit.get_completeness_metric(required_fields)
    
    print(f"Audit Trail ID: {audit.trail_id}")
    print(f"ATCM Score: {completeness['ATCM']:.3f}")
    print(f"Entities captured: {completeness['entity_count']}")
    print(f"Activities captured: {completeness['activity_count']}")
