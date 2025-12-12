"""
Audit Trail Generator for AuditOps Framework
Implements W3C PROV-O compliant provenance tracking for regulatory compliance
Author: Udayakumar Mani, Senthilkumar Rathinasamy
Organization: SASTRA Deemed University
License: MIT
"""

import uuid
import json
import datetime
import hashlib
import pickle
import warnings
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum, auto
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from pathlib import Path


class AuditEventType(Enum):
    """Types of audit events for classification and filtering."""
    DATA_INGESTION = "data_ingestion"
    DATA_PREPROCESSING = "data_preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    MODEL_DEPLOYMENT = "model_deployment"
    MODEL_INFERENCE = "model_inference"
    MODEL_MONITORING = "model_monitoring"
    MODEL_RETRAINING = "model_retraining"
    COMPLIANCE_CHECK = "compliance_check"
    EXPLANATION_GENERATION = "explanation_generation"
    HUMAN_REVIEW = "human_review"
    SYSTEM_ALERT = "system_alert"
    ERROR_EVENT = "error_event"
    CONFIGURATION_CHANGE = "configuration_change"


class ProvenanceRelationType(Enum):
    """W3C PROV-O relation types for provenance tracking."""
    WAS_GENERATED_BY = "wasGeneratedBy"
    USED = "used"
    WAS_INFORMED_BY = "wasInformedBy"
    WAS_STARTED_BY = "wasStartedBy"
    WAS_ENDED_BY = "wasEndedBy"
    WAS_INVALIDATED_BY = "wasInvalidatedBy"
    WAS_DERIVED_FROM = "wasDerivedFrom"
    WAS_ATTRIBUTED_TO = "wasAttributedTo"
    WAS_ASSOCIATED_WITH = "wasAssociatedWith"
    ACTED_ON_BEHALF_OF = "actedOnBehalfOf"


@dataclass
class ProvenanceEntity:
    """
    W3C PROV-O Entity representation for audit trail.
    
    Attributes:
        id (str): Unique identifier for the entity
        type (str): Type of entity (dataset, model, inference, etc.)
        attributes (Dict[str, Any]): Key-value pairs describing the entity
        timestamp (str): ISO format timestamp of entity creation
        version (str): Version identifier for entity evolution
        hash (str): Cryptographic hash for integrity verification
    """
    id: str
    type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    version: str = "1.0.0"
    hash: Optional[str] = None
    
    def __post_init__(self):
        """Calculate hash if not provided."""
        if self.hash is None:
            self.hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of entity representation."""
        content = f"{self.id}{self.type}{json.dumps(self.attributes, sort_keys=True)}{self.timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_prov_o(self) -> Dict[str, Any]:
        """Convert to W3C PROV-O JSON-LD format."""
        return {
            "@id": self.id,
            "@type": ["prov:Entity", f"auditops:{self.type}"],
            "prov:generatedAtTime": self.timestamp,
            "prov:hadPrimarySource": self.attributes.get("source"),
            "auditops:version": self.version,
            "auditops:hash": self.hash,
            **{f"auditops:{k}": v for k, v in self.attributes.items()}
        }


@dataclass
class ProvenanceActivity:
    """
    W3C PROV-O Activity representation for audit trail.
    
    Attributes:
        id (str): Unique identifier for the activity
        type (str): Type of activity (training, inference, validation, etc.)
        started_at (str): ISO format timestamp of activity start
        ended_at (Optional[str]): ISO format timestamp of activity end
        attributes (Dict[str, Any]): Key-value pairs describing the activity
        status (str): Activity status (running, completed, failed)
    """
    id: str
    type: str
    started_at: str
    ended_at: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    
    def to_prov_o(self) -> Dict[str, Any]:
        """Convert to W3C PROV-O JSON-LD format."""
        activity = {
            "@id": self.id,
            "@type": ["prov:Activity", f"auditops:{self.type}"],
            "prov:startedAtTime": self.started_at,
            "auditops:status": self.status,
            **{f"auditops:{k}": v for k, v in self.attributes.items()}
        }
        if self.ended_at:
            activity["prov:endedAtTime"] = self.ended_at
        return activity


@dataclass
class ProvenanceAgent:
    """
    W3C PROV-O Agent representation for audit trail.
    
    Attributes:
        id (str): Unique identifier for the agent
        type (str): Type of agent (system, user, automated_process, organization)
        name (str): Human-readable name of the agent
        attributes (Dict[str, Any]): Key-value pairs describing the agent
        roles (List[str]): Roles performed by the agent
    """
    id: str
    type: str
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    roles: List[str] = field(default_factory=list)
    
    def to_prov_o(self) -> Dict[str, Any]:
        """Convert to W3C PROV-O JSON-LD format."""
        return {
            "@id": self.id,
            "@type": ["prov:Agent", f"auditops:{self.type}"],
            "prov:label": self.name,
            "auditops:roles": self.roles,
            **{f"auditops:{k}": v for k, v in self.attributes.items()}
        }


@dataclass
class ProvenanceRelation:
    """
    W3C PROV-O Relation representation for audit trail.
    
    Attributes:
        id (str): Unique identifier for the relation
        type (ProvenanceRelationType): Type of relation
        source_id (str): ID of source entity/activity/agent
        target_id (str): ID of target entity/activity/agent
        attributes (Dict[str, Any]): Key-value pairs describing the relation
        timestamp (str): ISO format timestamp of relation creation
    """
    id: str
    type: ProvenanceRelationType
    source_id: str
    target_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    
    def to_prov_o(self) -> Dict[str, Any]:
        """Convert to W3C PROV-O JSON-LD format."""
        return {
            "@id": self.id,
            "@type": f"prov:{self.type.value}",
            "prov:qualifiedAssociation": {
                "@type": "prov:Association",
                "prov:agent": {"@id": self.target_id} if "agent" in self.type.value else {"@id": self.source_id},
                "prov:hadRole": self.attributes.get("role", "")
            },
            "prov:atTime": self.timestamp,
            **{f"auditops:{k}": v for k, v in self.attributes.items()}
        }


class AuditTrailStorageBackend:
    """Abstract base class for audit trail storage backends."""
    
    def save_entity(self, entity: ProvenanceEntity) -> bool:
        """Save entity to storage."""
        raise NotImplementedError
    
    def save_activity(self, activity: ProvenanceActivity) -> bool:
        """Save activity to storage."""
        raise NotImplementedError
    
    def save_agent(self, agent: ProvenanceAgent) -> bool:
        """Save agent to storage."""
        raise NotImplementedError
    
    def save_relation(self, relation: ProvenanceRelation) -> bool:
        """Save relation to storage."""
        raise NotImplementedError
    
    def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query audit trail data."""
        raise NotImplementedError
    
    def export(self, format: str = "json") -> Any:
        """Export audit trail data."""
        raise NotImplementedError


class JSONFileStorage(AuditTrailStorageBackend):
    """JSON file-based storage backend for audit trails."""
    
    def __init__(self, storage_path: Union[str, Path]):
        """
        Initialize JSON file storage.
        
        Args:
            storage_path: Path to storage directory
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage files
        self.entities_file = self.storage_path / "entities.json"
        self.activities_file = self.storage_path / "activities.json"
        self.agents_file = self.storage_path / "agents.json"
        self.relations_file = self.storage_path / "relations.json"
        
        # Initialize empty storage if files don't exist
        for file in [self.entities_file, self.activities_file, 
                    self.agents_file, self.relations_file]:
            if not file.exists():
                with open(file, 'w') as f:
                    json.dump([], f)
    
    def save_entity(self, entity: ProvenanceEntity) -> bool:
        """Save entity to JSON file."""
        try:
            with open(self.entities_file, 'r') as f:
                entities = json.load(f)
            
            # Check if entity already exists
            existing_idx = next((i for i, e in enumerate(entities) 
                               if e.get('id') == entity.id), None)
            
            entity_dict = asdict(entity)
            if existing_idx is not None:
                entities[existing_idx] = entity_dict
            else:
                entities.append(entity_dict)
            
            with open(self.entities_file, 'w') as f:
                json.dump(entities, f, indent=2, default=str)
            
            return True
        except Exception as e:
            warnings.warn(f"Failed to save entity: {e}")
            return False
    
    def save_activity(self, activity: ProvenanceActivity) -> bool:
        """Save activity to JSON file."""
        try:
            with open(self.activities_file, 'r') as f:
                activities = json.load(f)
            
            activity_dict = asdict(activity)
            activities.append(activity_dict)
            
            with open(self.activities_file, 'w') as f:
                json.dump(activities, f, indent=2, default=str)
            
            return True
        except Exception as e:
            warnings.warn(f"Failed to save activity: {e}")
            return False
    
    def save_agent(self, agent: ProvenanceAgent) -> bool:
        """Save agent to JSON file."""
        try:
            with open(self.agents_file, 'r') as f:
                agents = json.load(f)
            
            agent_dict = asdict(agent)
            
            # Check if agent already exists
            existing_idx = next((i for i, a in enumerate(agents) 
                               if a.get('id') == agent.id), None)
            
            if existing_idx is not None:
                agents[existing_idx] = agent_dict
            else:
                agents.append(agent_dict)
            
            with open(self.agents_file, 'w') as f:
                json.dump(agents, f, indent=2, default=str)
            
            return True
        except Exception as e:
            warnings.warn(f"Failed to save agent: {e}")
            return False
    
    def save_relation(self, relation: ProvenanceRelation) -> bool:
        """Save relation to JSON file."""
        try:
            with open(self.relations_file, 'r') as f:
                relations = json.load(f)
            
            relation_dict = asdict(relation)
            relation_dict['type'] = relation.type.value  # Convert enum to string
            
            relations.append(relation_dict)
            
            with open(self.relations_file, 'w') as f:
                json.dump(relations, f, indent=2, default=str)
            
            return True
        except Exception as e:
            warnings.warn(f"Failed to save relation: {e}")
            return False
    
    def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query audit trail data with flexible filtering."""
        results = []
        
        # Determine which collection to query
        collections = {
            'entity': self.entities_file,
            'activity': self.activities_file,
            'agent': self.agents_file,
            'relation': self.relations_file
        }
        
        target_collection = query_params.get('collection', 'entity')
        
        if target_collection not in collections:
            raise ValueError(f"Invalid collection: {target_collection}")
        
        try:
            with open(collections[target_collection], 'r') as f:
                items = json.load(f)
            
            # Apply filters
            for item in items:
                match = True
                for key, value in query_params.get('filters', {}).items():
                    if key not in item or item[key] != value:
                        match = False
                        break
                
                # Apply time range filter
                if 'start_time' in query_params and 'timestamp' in item:
                    item_time = datetime.datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                    start_time = datetime.datetime.fromisoformat(query_params['start_time'].replace('Z', '+00:00'))
                    if item_time < start_time:
                        match = False
                
                if 'end_time' in query_params and 'timestamp' in item:
                    item_time = datetime.datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                    end_time = datetime.datetime.fromisoformat(query_params['end_time'].replace('Z', '+00:00'))
                    if item_time > end_time:
                        match = False
                
                if match:
                    results.append(item)
            
            # Apply sorting
            sort_key = query_params.get('sort_by')
            if sort_key and results:
                results.sort(key=lambda x: x.get(sort_key, ''))
            
            # Apply limit
            limit = query_params.get('limit')
            if limit:
                results = results[:limit]
            
        except Exception as e:
            warnings.warn(f"Query failed: {e}")
        
        return results
    
    def export(self, format: str = "json") -> Dict[str, Any]:
        """Export complete audit trail."""
        try:
            with open(self.entities_file, 'r') as f:
                entities = json.load(f)
            
            with open(self.activities_file, 'r') as f:
                activities = json.load(f)
            
            with open(self.agents_file, 'r') as f:
                agents = json.load(f)
            
            with open(self.relations_file, 'r') as f:
                relations = json.load(f)
            
            export_data = {
                "metadata": {
                    "export_timestamp": datetime.datetime.utcnow().isoformat(),
                    "format": format,
                    "total_entities": len(entities),
                    "total_activities": len(activities),
                    "total_agents": len(agents),
                    "total_relations": len(relations),
                    "storage_backend": "JSONFileStorage",
                    "auditops_version": "1.0.0"
                },
                "entities": entities,
                "activities": activities,
                "agents": agents,
                "relations": relations
            }
            
            if format == "prov-o":
                return self._convert_to_prov_o(export_data)
            else:
                return export_data
                
        except Exception as e:
            warnings.warn(f"Export failed: {e}")
            return {}
    
    def _convert_to_prov_o(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert internal format to W3C PROV-O JSON-LD."""
        prov_o = {
            "@context": {
                "prov": "http://www.w3.org/ns/prov#",
                "auditops": "https://auditops.org/ns#",
                "xsd": "http://www.w3.org/2001/XMLSchema#"
            },
            "@graph": []
        }
        
        # Convert entities
        for entity in data["entities"]:
            prov_entity = {
                "@id": entity["id"],
                "@type": ["prov:Entity", f"auditops:{entity['type']}"],
                "prov:generatedAtTime": {
                    "@type": "xsd:dateTime",
                    "@value": entity["timestamp"]
                }
            }
            if entity.get("hash"):
                prov_entity["auditops:hash"] = entity["hash"]
            prov_o["@graph"].append(prov_entity)
        
        # Convert activities
        for activity in data["activities"]:
            prov_activity = {
                "@id": activity["id"],
                "@type": ["prov:Activity", f"auditops:{activity['type']}"],
                "prov:startedAtTime": {
                    "@type": "xsd:dateTime",
                    "@value": activity["started_at"]
                }
            }
            if activity.get("ended_at"):
                prov_activity["prov:endedAtTime"] = {
                    "@type": "xsd:dateTime",
                    "@value": activity["ended_at"]
                }
            prov_o["@graph"].append(prov_activity)
        
        # Convert relations
        for relation in data["relations"]:
            prov_relation = {
                "@id": relation["id"],
                f"prov:{relation['type']}": {
                    "prov:entity": {"@id": relation["source_id"]},
                    "prov:activity": {"@id": relation["target_id"]}
                } if relation["type"] == "wasGeneratedBy" else {
                    "prov:activity": {"@id": relation["source_id"]},
                    "prov:entity": {"@id": relation["target_id"]}
                }
            }
            prov_o["@graph"].append(prov_relation)
        
        return prov_o


class AuditTrailGenerator:
    """
    Main audit trail generator for AuditOps framework.
    
    This class implements comprehensive provenance tracking following W3C PROV-O
    standards, capturing all decision-influencing factors in ML systems for
    regulatory compliance.
    
    Attributes:
        trail_id (str): Unique identifier for this audit trail session
        storage_backend (AuditTrailStorageBackend): Storage implementation
        system_agent (ProvenanceAgent): System agent representing AuditOps
        required_fields (List[str]): Fields required for compliance
        metrics (Dict[str, Any]): Performance and completeness metrics
    """
    
    def __init__(self, 
                 storage_backend: Optional[AuditTrailStorageBackend] = None,
                 storage_path: Optional[Union[str, Path]] = None,
                 regulatory_context: str = "EU_AI_ACT_HIGH_RISK",
                 required_fields: Optional[List[str]] = None):
        """
        Initialize audit trail generator.
        
        Args:
            storage_backend: Custom storage backend (optional)
            storage_path: Path for file-based storage (used if backend not provided)
            regulatory_context: Regulatory framework for compliance
            required_fields: Fields required to capture for compliance
            
        Raises:
            ValueError: If neither storage_backend nor storage_path provided
        """
        self.trail_id = self._generate_trail_id()
        self.regulatory_context = regulatory_context
        self.required_fields = required_fields or self._get_default_required_fields(regulatory_context)
        
        # Initialize storage
        if storage_backend:
            self.storage = storage_backend
        elif storage_path:
            self.storage = JSONFileStorage(storage_path)
        else:
            raise ValueError("Must provide either storage_backend or storage_path")
        
        # Initialize system agent
        self.system_agent = self._create_system_agent()
        self.storage.save_agent(self.system_agent)
        
        # Initialize metrics tracking
        self.metrics = {
            "entities_created": 0,
            "activities_logged": 0,
            "relations_recorded": 0,
            "compliance_checks": 0,
            "start_time": datetime.datetime.utcnow().isoformat(),
            "last_activity": datetime.datetime.utcnow().isoformat()
        }
        
        # Cache for quick lookups
        self.entity_cache = {}
        self.activity_cache = {}
        
        print(f"AuditTrailGenerator initialized with trail_id: {self.trail_id}")
    
    def _generate_trail_id(self) -> str:
        """Generate unique trail ID."""
        timestamp = int(datetime.datetime.utcnow().timestamp())
        random_component = uuid.uuid4().hex[:8]
        return f"audit_trail_{timestamp}_{random_component}"
    
    def _get_default_required_fields(self, regulatory_context: str) -> List[str]:
        """Get default required fields based on regulatory context."""
        base_fields = [
            "timestamp",
            "entity_id",
            "entity_type",
            "activity_type",
            "agent_id",
            "input_hash",
            "output_hash",
            "model_version",
            "environment_info",
            "compliance_status"
        ]
        
        context_specific_fields = {
            "EU_AI_ACT_HIGH_RISK": [
                "explanation_method",
                "feature_importance",
                "fairness_metrics",
                "risk_assessment",
                "human_oversight_flag"
            ],
            "FDA_21_CFR_820": [
                "validation_protocol",
                "acceptance_criteria",
                "calibration_data",
                "quality_metrics",
                "change_control_id"
            ],
            "REGULATION_B": [
                "protected_attributes",
                "disparate_impact_ratio",
                "fairness_test_results",
                "bias_mitigation_applied",
                "equal_opportunity_score"
            ]
        }
        
        return base_fields + context_specific_fields.get(regulatory_context, [])
    
    def _create_system_agent(self) -> ProvenanceAgent:
        """Create system agent representing AuditOps framework."""
        return ProvenanceAgent(
            id="agent:system:auditops",
            type="system",
            name="AuditOps Compliance Framework",
            attributes={
                "version": "1.0.0",
                "regulatory_context": self.regulatory_context,
                "capabilities": ["provenance_tracking", "compliance_monitoring", "audit_trail_generation"],
                "framework": "AuditOps"
            },
            roles=["compliance_monitor", "audit_trail_generator", "provenance_recorder"]
        )
    
    def log_data_ingestion(self,
                          dataset_id: str,
                          dataset_info: Dict[str, Any],
                          data_hash: Optional[str] = None,
                          source_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Log data ingestion event with comprehensive provenance.
        
        Args:
            dataset_id: Unique identifier for the dataset
            dataset_info: Metadata about the dataset
            data_hash: Cryptographic hash of dataset (calculated if not provided)
            source_info: Information about data source
            
        Returns:
            str: Entity ID of the logged dataset
            
        Example:
            >>> audit = AuditTrailGenerator(storage_path="./audit_logs")
            >>> dataset_id = audit.log_data_ingestion(
            ...     dataset_id="heart_disease_v1",
            ...     dataset_info={
            ...         "record_count": 1000,
            ...         "features": ["age", "sex", "chol"],
            ...         "source": "UCI Repository"
            ...     }
            ... )
        """
        # Calculate hash if not provided
        if data_hash is None and 'data' in dataset_info:
            data_hash = hashlib.sha256(str(dataset_info['data']).encode()).hexdigest()
        
        # Create dataset entity
        dataset_entity = ProvenanceEntity(
            id=f"entity:dataset:{dataset_id}:{uuid.uuid4().hex[:8]}",
            type="dataset",
            attributes={
                "dataset_id": dataset_id,
                "data_hash": data_hash,
                "record_count": dataset_info.get("record_count"),
                "feature_count": dataset_info.get("feature_count"),
                "features": dataset_info.get("features", []),
                "source": dataset_info.get("source", "unknown"),
                "ingestion_method": dataset_info.get("ingestion_method"),
                "validation_status": dataset_info.get("validation_status", "pending"),
                "compliance_checks": dataset_info.get("compliance_checks", []),
                "pii_handling": dataset_info.get("pii_handling", "anonymized"),
                "license": dataset_info.get("license"),
                "retention_policy": dataset_info.get("retention_policy"),
                **({"source_info": source_info} if source_info else {})
            }
        )
        
        # Create ingestion activity
        ingestion_activity = ProvenanceActivity(
            id=f"activity:data_ingestion:{uuid.uuid4().hex[:8]}",
            type="data_ingestion",
            started_at=datetime.datetime.utcnow().isoformat(),
            ended_at=datetime.datetime.utcnow().isoformat(),
            attributes={
                "method": dataset_info.get("ingestion_method", "batch"),
                "duration_seconds": dataset_info.get("duration_seconds"),
                "throughput_records_per_second": dataset_info.get("throughput"),
                "validation_checks_passed": dataset_info.get("validation_checks_passed", True),
                "quality_metrics": dataset_info.get("quality_metrics", {}),
                "compliance_violations": dataset_info.get("compliance_violations", [])
            }
        )
        
        # Save to storage
        self.storage.save_entity(dataset_entity)
        self.storage.save_activity(ingestion_activity)
        
        # Create relations
        self._create_relation(
            relation_type=ProvenanceRelationType.WAS_GENERATED_BY,
            source_id=dataset_entity.id,
            target_id=ingestion_activity.id,
            attributes={"role": "output_dataset"}
        )
        
        self._create_relation(
            relation_type=ProvenanceRelationType.WAS_ASSOCIATED_WITH,
            source_id=ingestion_activity.id,
            target_id=self.system_agent.id,
            attributes={"role": "ingestion_engine"}
        )
        
        # Update metrics and cache
        self.metrics["entities_created"] += 1
        self.metrics["activities_logged"] += 1
        self.entity_cache[dataset_entity.id] = dataset_entity
        
        return dataset_entity.id
    
    def log_model_training(self,
                          model_id: str,
                          model_info: Dict[str, Any],
                          input_entity_ids: List[str],
                          training_data_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Log model training event with comprehensive provenance.
        
        Args:
            model_id: Unique identifier for the model
            model_info: Metadata about the model and training process
            input_entity_ids: IDs of input datasets/entities used for training
            training_data_info: Additional information about training data
            
        Returns:
            str: Entity ID of the trained model
            
        Example:
            >>> model_id = audit.log_model_training(
            ...     model_id="rf_classifier_v1",
            ...     model_info={
            ...         "model_type": "RandomForest",
            ...         "hyperparameters": {"n_estimators": 100},
            ...         "training_metrics": {"accuracy": 0.85}
            ...     },
            ...     input_entity_ids=[dataset_entity_id]
            ... )
        """
        # Calculate model hash
        model_hash = self._calculate_model_hash(model_info)
        
        # Create model entity
        model_entity = ProvenanceEntity(
            id=f"entity:model:{model_id}:{uuid.uuid4().hex[:8]}",
            type="model",
            attributes={
                "model_id": model_id,
                "model_hash": model_hash,
                "model_type": model_info.get("model_type"),
                "architecture": model_info.get("architecture"),
                "hyperparameters": model_info.get("hyperparameters", {}),
                "training_metrics": model_info.get("training_metrics", {}),
                "validation_metrics": model_info.get("validation_metrics", {}),
                "feature_names": model_info.get("feature_names", []),
                "target_names": model_info.get("target_names", []),
                "training_duration_seconds": model_info.get("training_duration"),
                "hardware_used": model_info.get("hardware_info", {}),
                "framework_version": model_info.get("framework_version"),
                "random_seed": model_info.get("random_seed"),
                "compliance_checks": model_info.get("compliance_checks", []),
                "fairness_metrics": model_info.get("fairness_metrics", {}),
                "explainability_support": model_info.get("explainability_support", True),
                **({"training_data_info": training_data_info} if training_data_info else {})
            },
            version=model_info.get("version", "1.0.0")
        )
        
        # Create training activity
        training_activity = ProvenanceActivity(
            id=f"activity:model_training:{uuid.uuid4().hex[:8]}",
            type="model_training",
            started_at=datetime.datetime.utcnow().isoformat(),
            ended_at=datetime.datetime.utcnow().isoformat(),
            attributes={
                "algorithm": model_info.get("algorithm"),
                "training_strategy": model_info.get("training_strategy", "standard"),
                "cross_validation_folds": model_info.get("cross_validation_folds"),
                "early_stopping": model_info.get("early_stopping", False),
                "regularization": model_info.get("regularization", {}),
                "optimization_method": model_info.get("optimization_method"),
                "loss_function": model_info.get("loss_function"),
                "compliance_constraints": model_info.get("compliance_constraints", []),
                "fairness_constraints": model_info.get("fairness_constraints", []),
                "privacy_parameters": model_info.get("privacy_parameters", {})
            }
        )
        
        # Save to storage
        self.storage.save_entity(model_entity)
        self.storage.save_activity(training_activity)
        
        # Create relations for inputs
        for input_id in input_entity_ids:
            self._create_relation(
                relation_type=ProvenanceRelationType.USED,
                source_id=training_activity.id,
                target_id=input_id,
                attributes={"role": "training_data"}
            )
        
        # Create relation for output
        self._create_relation(
            relation_type=ProvenanceRelationType.WAS_GENERATED_BY,
            source_id=model_entity.id,
            target_id=training_activity.id,
            attributes={"role": "trained_model"}
        )
        
        # Create agent relation
        self._create_relation(
            relation_type=ProvenanceRelationType.WAS_ASSOCIATED_WITH,
            source_id=training_activity.id,
            target_id=self.system_agent.id,
            attributes={"role": "training_engine"}
        )
        
        # Update metrics and cache
        self.metrics["entities_created"] += 1
        self.metrics["activities_logged"] += 1
        self.entity_cache[model_entity.id] = model_entity
        
        return model_entity.id
    
    def log_inference(self,
                     inference_id: str,
                     model_entity_id: str,
                     input_data: Union[Dict[str, Any], np.ndarray, pd.DataFrame],
                     prediction: Any,
                     explanation: Optional[Dict[str, Any]] = None,
                     context: Optional[Dict[str, Any]] = None) -> str:
        """
        Log model inference event with comprehensive provenance.
        
        Args:
            inference_id: Unique identifier for the inference
            model_entity_id: ID of the model entity used
            input_data: Input data for inference
            prediction: Model prediction/output
            explanation: Explanation of the prediction (optional)
            context: Additional context information (optional)
            
        Returns:
            str: Entity ID of the inference result
            
        Example:
            >>> inference_id = audit.log_inference(
            ...     inference_id="inf_001",
            ...     model_entity_id=model_entity_id,
            ...     input_data={"age": 55, "chol": 250},
            ...     prediction={"class": 1, "confidence": 0.87},
            ...     explanation={"method": "SHAP", "feature_importance": {...}}
            ... )
        """
        # Calculate input hash
        if isinstance(input_data, (np.ndarray, pd.DataFrame)):
            input_hash = hashlib.sha256(pickle.dumps(input_data)).hexdigest()
        else:
            input_hash = hashlib.sha256(json.dumps(input_data, sort_keys=True).encode()).hexdigest()
        
        # Prepare prediction info
        if hasattr(prediction, '__dict__'):
            prediction_info = prediction.__dict__
        elif isinstance(prediction, dict):
            prediction_info = prediction
        else:
            prediction_info = {"value": prediction}
        
        # Create inference entity
        inference_entity = ProvenanceEntity(
            id=f"entity:inference:{inference_id}:{uuid.uuid4().hex[:8]}",
            type="inference",
            attributes={
                "inference_id": inference_id,
                "input_hash": input_hash,
                "prediction": prediction_info,
                "confidence": prediction_info.get("confidence"),
                "prediction_class": prediction_info.get("class"),
                "prediction_probabilities": prediction_info.get("probabilities"),
                "explanation_available": explanation is not None,
                "explanation_method": explanation.get("method") if explanation else None,
                "explanation_confidence": explanation.get("confidence") if explanation else None,
                "feature_importance": explanation.get("feature_importance") if explanation else None,
                "latency_ms": context.get("latency_ms") if context else None,
                "batch_size": context.get("batch_size") if context else 1,
                "request_id": context.get("request_id") if context else None,
                "user_id": context.get("user_id") if context else None,
                "environment": {
                    "hostname": context.get("hostname", "localhost") if context else "localhost",
                    "python_version": context.get("python_version", sys.version),
                    "framework_versions": context.get("framework_versions", {}),
                    "timestamp": datetime.datetime.utcnow().isoformat()
                },
                "compliance_checks": context.get("compliance_checks", []) if context else [],
                "regulatory_flags": context.get("regulatory_flags", []) if context else []
            }
        )
        
        # Create inference activity
        inference_activity = ProvenanceActivity(
            id=f"activity:model_inference:{uuid.uuid4().hex[:8]}",
            type="model_inference",
            started_at=datetime.datetime.utcnow().isoformat(),
            ended_at=datetime.datetime.utcnow().isoformat(),
            attributes={
                "model_version": context.get("model_version") if context else "unknown",
                "inference_mode": context.get("inference_mode", "online") if context else "online",
                "hardware_accelerated": context.get("hardware_accelerated", False) if context else False,
                "quantization_applied": context.get("quantization_applied", False) if context else False,
                "privacy_preserving": context.get("privacy_preserving", False) if context else False,
                "explanation_generated": explanation is not None,
                "compliance_validated": context.get("compliance_validated", True) if context else True
            }
        )
        
        # Save to storage
        self.storage.save_entity(inference_entity)
        self.storage.save_activity(inference_activity)
        
        # Create relations
        self._create_relation(
            relation_type=ProvenanceRelationType.USED,
            source_id=inference_activity.id,
            target_id=model_entity_id,
            attributes={"role": "inference_model"}
        )
        
        self._create_relation(
            relation_type=ProvenanceRelationType.WAS_GENERATED_BY,
            source_id=inference_entity.id,
            target_id=inference_activity.id,
            attributes={"role": "inference_result"}
        )
        
        # Create agent relation
        self._create_relation(
            relation_type=ProvenanceRelationType.WAS_ASSOCIATED_WITH,
            source_id=inference_activity.id,
            target_id=self.system_agent.id,
            attributes={"role": "inference_engine"}
        )
        
        # If explanation provided, log it separately
        if explanation:
            self.log_explanation_generation(
                inference_entity_id=inference_entity.id,
                explanation_data=explanation,
                model_entity_id=model_entity_id
            )
        
        # Update metrics and cache
        self.metrics["entities_created"] += 1
        self.metrics["activities_logged"] += 1
        self.metrics["last_activity"] = datetime.datetime.utcnow().isoformat()
        self.entity_cache[inference_entity.id] = inference_entity
        
        return inference_entity.id
    
    def log_explanation_generation(self,
                                 inference_entity_id: str,
                                 explanation_data: Dict[str, Any],
                                 model_entity_id: Optional[str] = None) -> str:
        """
        Log explanation generation event.
        
        Args:
            inference_entity_id: ID of the inference being explained
            explanation_data: Explanation data and metadata
            model_entity_id: ID of the model (optional)
            
        Returns:
            str: Entity ID of the explanation
        """
        # Create explanation entity
        explanation_entity = ProvenanceEntity(
            id=f"entity:explanation:{uuid.uuid4().hex[:8]}",
            type="explanation",
            attributes={
                "method": explanation_data.get("method"),
                "feature_importance": explanation_data.get("feature_importance"),
                "local_explanations": explanation_data.get("local_explanations"),
                "global_explanations": explanation_data.get("global_explanations"),
                "confidence": explanation_data.get("confidence"),
                "completeness_score": explanation_data.get("completeness_score"),
                "stability_score": explanation_data.get("stability_score"),
                "regulatory_compliant": explanation_data.get("regulatory_compliant", True),
                "audience_level": explanation_data.get("audience_level", "expert")
            }
        )
        
        # Create explanation activity
        explanation_activity = ProvenanceActivity(
            id=f"activity:explanation_generation:{uuid.uuid4().hex[:8]}",
            type="explanation_generation",
            started_at=datetime.datetime.utcnow().isoformat(),
            ended_at=datetime.datetime.utcnow().isoformat(),
            attributes={
                "generation_time_ms": explanation_data.get("generation_time_ms"),
                "resource_usage": explanation_data.get("resource_usage", {}),
                "approximation_used": explanation_data.get("approximation_used", False),
                "black_box_compatible": explanation_data.get("black_box_compatible", True)
            }
        )
        
        # Save to storage
        self.storage.save_entity(explanation_entity)
        self.storage.save_activity(explanation_activity)
        
        # Create relations
        self._create_relation(
            relation_type=ProvenanceRelationType.WAS_GENERATED_BY,
            source_id=explanation_entity.id,
            target_id=explanation_activity.id,
            attributes={"role": "explanation_output"}
        )
        
        self._create_relation(
            relation_type=ProvenanceRelationType.WAS_DERIVED_FROM,
            source_id=explanation_entity.id,
            target_id=inference_entity_id,
            attributes={"role": "explained_inference"}
        )
        
        if model_entity_id:
            self._create_relation(
                relation_type=ProvenanceRelationType.USED,
                source_id=explanation_activity.id,
                target_id=model_entity_id,
                attributes={"role": "explained_model"}
            )
        
        # Update metrics
        self.metrics["entities_created"] += 1
        self.metrics["activities_logged"] += 1
        
        return explanation_entity.id
    
    def log_compliance_check(self,
                           check_type: str,
                           check_result: Dict[str, Any],
                           related_entity_ids: List[str],
                           regulatory_requirement: Optional[str] = None) -> str:
        """
        Log compliance check event.
        
        Args:
            check_type: Type of compliance check
            check_result: Result of the check
            related_entity_ids: IDs of related entities
            regulatory_requirement: Specific regulatory requirement ID
            
        Returns:
            str: Activity ID of the compliance check
        """
        # Create compliance check activity
        compliance_activity = ProvenanceActivity(
            id=f"activity:compliance_check:{uuid.uuid4().hex[:8]}",
            type="compliance_check",
            started_at=datetime.datetime.utcnow().isoformat(),
            ended_at=datetime.datetime.utcnow().isoformat(),
            attributes={
                "check_type": check_type,
                "result": check_result.get("result"),
                "details": check_result.get("details"),
                "threshold": check_result.get("threshold"),
                "actual_value": check_result.get("actual_value"),
                "deviation": check_result.get("deviation"),
                "severity": check_result.get("severity", "INFO"),
                "regulatory_requirement": regulatory_requirement,
                "automated": check_result.get("automated", True),
                "validation_method": check_result.get("validation_method", "automated")
            },
            status="completed" if check_result.get("result") == "PASSED" else "failed"
        )
        
        # Save to storage
        self.storage.save_activity(compliance_activity)
        
        # Create relations to related entities
        for entity_id in related_entity_ids:
            self._create_relation(
                relation_type=ProvenanceRelationType.WAS_INFORMED_BY,
                source_id=compliance_activity.id,
                target_id=entity_id,
                attributes={
                    "role": "checked_entity",
                    "check_applicability": "direct"
                }
            )
        
        # Create agent relation
        self._create_relation(
            relation_type=ProvenanceRelationType.WAS_ASSOCIATED_WITH,
            source_id=compliance_activity.id,
            target_id=self.system_agent.id,
            attributes={"role": "compliance_checker"}
        )
        
        # Create alert entity if check failed
        if check_result.get("result") == "FAILED":
            alert_entity = self._create_alert_entity(
                compliance_activity_id=compliance_activity.id,
                check_result=check_result,
                check_type=check_type,
                regulatory_requirement=regulatory_requirement
            )
            
            self._create_relation(
                relation_type=ProvenanceRelationType.WAS_GENERATED_BY,
                source_id=alert_entity.id,
                target_id=compliance_activity.id,
                attributes={"role": "compliance_alert"}
            )
        
        # Update metrics
        self.metrics["activities_logged"] += 1
        self.metrics["compliance_checks"] += 1
        
        return compliance_activity.id
    
    def _create_alert_entity(self,
                           compliance_activity_id: str,
                           check_result: Dict[str, Any],
                           check_type: str,
                           regulatory_requirement: Optional[str] = None) -> ProvenanceEntity:
        """Create alert entity for compliance violations."""
        alert_entity = ProvenanceEntity(
            id=f"entity:alert:{uuid.uuid4().hex[:8]}",
            type="alert",
            attributes={
                "severity": check_result.get("severity", "WARNING"),
                "message": check_result.get("message", "Compliance check failed"),
                "check_type": check_type,
                "regulatory_requirement": regulatory_requirement,
                "detected_at": datetime.datetime.utcnow().isoformat(),
                "recommended_action": check_result.get("recommended_action"),
                "escalation_level": check_result.get("escalation_level", "L1"),
                "acknowledged": False,
                "resolved": False,
                "time_to_acknowledge": None,
                "time_to_resolve": None
            }
        )
        
        self.storage.save_entity(alert_entity)
        self.metrics["entities_created"] += 1
        
        return alert_entity
    
    def _create_relation(self,
                        relation_type: ProvenanceRelationType,
                        source_id: str,
                        target_id: str,
                        attributes: Optional[Dict[str, Any]] = None) -> str:
        """Create and save a provenance relation."""
        if attributes is None:
            attributes = {}
        
        relation = ProvenanceRelation(
            id=f"relation:{uuid.uuid4().hex[:8]}",
            type=relation_type,
            source_id=source_id,
            target_id=target_id,
            attributes=attributes
        )
        
        self.storage.save_relation(relation)
        self.metrics["relations_recorded"] += 1
        
        return relation.id
    
    def _calculate_model_hash(self, model_info: Dict[str, Any]) -> str:
        """Calculate hash for model identification."""
        hash_components = [
            str(model_info.get('model_type', '')),
            str(model_info.get('hyperparameters', {})),
            str(model_info.get('architecture', '')),
            str(model_info.get('framework_version', ''))
        ]
        
        hash_input = ''.join(hash_components)
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def get_completeness_metric(self, 
                               entity_type: Optional[str] = None,
                               time_range: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        Calculate Audit Trail Completeness Metric (ATCM).
        
        Args:
            entity_type: Specific entity type to analyze (optional)
            time_range: Time range for analysis (start_time, end_time) in ISO format
            
        Returns:
            Dict containing ATCM and detailed analysis
            
        Example:
            >>> completeness = audit.get_completeness_metric(
            ...     entity_type="inference",
            ...     time_range=("2024-01-01T00:00:00Z", "2024-12-31T23:59:59Z")
            ... )
            >>> print(f"ATCM: {completeness['ATCM']:.3f}")
        """
        # Query entities based on filters
        query_params = {"collection": "entity"}
        
        if entity_type:
            query_params["filters"] = {"type": entity_type}
        
        if time_range:
            query_params["start_time"] = time_range[0]
            query_params["end_time"] = time_range[1]
        
        entities = self.storage.query(query_params)
        
        if not entities:
            return {
                "ATCM": 0.0,
                "total_entities": 0,
                "captured_fields": 0,
                "total_expected_fields": 0,
                "field_coverage": {},
                "missing_fields": []
            }
        
        # Calculate field coverage
        field_presence = defaultdict(int)
        total_fields = 0
        
        for entity in entities:
            entity_fields = set(entity.get("attributes", {}).keys())
            
            # Check required fields
            for field in self.required_fields:
                total_fields += 1
                if field in entity_fields:
                    field_presence[field] += 1
        
        # Calculate ATCM
        captured_fields = sum(field_presence.values())
        atcm = captured_fields / total_fields if total_fields > 0 else 0.0
        
        # Identify missing fields
        entity_count = len(entities)
        missing_fields = [
            field for field in self.required_fields
            if field_presence.get(field, 0) < entity_count * 0.8  # < 80% coverage
        ]
        
        # Calculate field coverage percentages
        field_coverage = {
            field: (field_presence.get(field, 0) / entity_count * 100)
            for field in self.required_fields
        }
        
        return {
            "ATCM": atcm,
            "total_entities": len(entities),
            "captured_fields": captured_fields,
            "total_expected_fields": total_fields,
            "field_coverage": field_coverage,
            "missing_fields": missing_fields,
            "compliance_threshold_met": atcm >= 0.95,  # 95% completeness target
            "regulatory_context": self.regulatory_context,
            "analysis_timestamp": datetime.datetime.utcnow().isoformat()
        }
    
    def export_trail(self, 
                    format: str = "json",
                    include_metrics: bool = True,
                    compress: bool = False) -> Union[Dict[str, Any], bytes]:
        """
        Export complete audit trail.
        
        Args:
            format: Export format ('json', 'prov-o', 'csv')
            include_metrics: Whether to include performance metrics
            compress: Whether to compress the output
            
        Returns:
            Exported audit trail data
            
        Raises:
            ValueError: If unsupported format requested
        """
        if format not in ["json", "prov-o", "csv"]:
            raise ValueError(f"Unsupported format: {format}")
        
        # Get data from storage
        export_data = self.storage.export(format=format)
        
        # Add trail metadata
        export_data["trail_metadata"] = {
            "trail_id": self.trail_id,
            "regulatory_context": self.regulatory_context,
            "created_at": self.metrics["start_time"],
            "last_updated": self.metrics["last_activity"],
            "required_fields": self.required_fields,
            "storage_backend": type(self.storage).__name__
        }
        
        # Add metrics if requested
        if include_metrics:
            export_data["performance_metrics"] = self.metrics
            export_data["completeness_analysis"] = self.get_completeness_metric()
        
        # Handle different formats
        if format == "csv":
            # Convert to CSV format (simplified)
            csv_data = self._convert_to_csv(export_data)
            if compress:
                import gzip
                return gzip.compress(csv_data.encode())
            return csv_data
        elif format == "prov-o":
            # Already in PROV-O format from storage
            if compress:
                import gzip
                return gzip.compress(json.dumps(export_data).encode())
            return export_data
        else:  # JSON
            if compress:
                import gzip
                return gzip.compress(json.dumps(export_data).encode())
            return export_data
    
    def _convert_to_csv(self, data: Dict[str, Any]) -> str:
        """Convert audit trail data to CSV format."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write entities
        writer.writerow(["ENTITY_ID", "TYPE", "TIMESTAMP", "ATTRIBUTES"])
        for entity in data.get("entities", []):
            writer.writerow([
                entity.get("id"),
                entity.get("type"),
                entity.get("timestamp"),
                json.dumps(entity.get("attributes", {}))
            ])
        
        # Write activities
        writer.writerow([])
        writer.writerow(["ACTIVITY_ID", "TYPE", "STARTED_AT", "ENDED_AT", "STATUS"])
        for activity in data.get("activities", []):
            writer.writerow([
                activity.get("id"),
                activity.get("type"),
                activity.get("started_at"),
                activity.get("ended_at", ""),
                activity.get("status")
            ])
        
        return output.getvalue()
    
    def query_decision_provenance(self, 
                                 entity_id: str,
                                 max_depth: int = 3) -> Dict[str, Any]:
        """
        Query complete provenance chain for a decision.
        
        Args:
            entity_id: ID of the entity to trace
            max_depth: Maximum depth to trace back
            
        Returns:
            Complete provenance chain
            
        Example:
            >>> provenance = audit.query_decision_provenance(inference_entity_id)
            >>> print(f"Decision was based on: {provenance['inputs']}")
        """
        provenance_chain = {
            "target_entity": None,
            "inputs": [],
            "processes": [],
            "agents": [],
            "decisions": [],
            "compliance_checks": [],
            "depth_reached": 0
        }
        
        # Get target entity
        entity_query = self.storage.query({
            "collection": "entity",
            "filters": {"id": entity_id},
            "limit": 1
        })
        
        if not entity_query:
            return provenance_chain
        
        target_entity = entity_query[0]
        provenance_chain["target_entity"] = target_entity
        
        # Trace back through relations
        visited = set()
        queue = [(entity_id, 0)]  # (entity_id, depth)
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if depth >= max_depth or current_id in visited:
                continue
            
            visited.add(current_id)
            
            # Find relations where this entity is the source (was generated by)
            relations = self.storage.query({
                "collection": "relation",
                "filters": {"source_id": current_id},
                "sort_by": "timestamp"
            })
            
            for relation in relations:
                rel_type = relation.get("type")
                target_id = relation.get("target_id")
                
                if rel_type == "wasGeneratedBy":
                    # Find the activity that generated this entity
                    activities = self.storage.query({
                        "collection": "activity",
                        "filters": {"id": target_id},
                        "limit": 1
                    })
                    
                    if activities:
                        provenance_chain["processes"].append({
                            "activity": activities[0],
                            "relation": relation,
                            "depth": depth
                        })
                        
                        # Find inputs to this activity
                        input_relations = self.storage.query({
                            "collection": "relation",
                            "filters": {"target_id": target_id, "type": "used"}
                        })
                        
                        for input_rel in input_relations:
                            input_entity_id = input_rel.get("source_id")
                            input_entities = self.storage.query({
                                "collection": "entity",
                                "filters": {"id": input_entity_id},
                                "limit": 1
                            })
                            
                            if input_entities:
                                provenance_chain["inputs"].append({
                                    "entity": input_entities[0],
                                    "relation": input_rel,
                                    "depth": depth + 1
                                })
                                
                                queue.append((input_entity_id, depth + 1))
                
                elif "wasAssociatedWith" in rel_type:
                    # Find the agent associated with this
                    agents = self.storage.query({
                        "collection": "agent",
                        "filters": {"id": target_id},
                        "limit": 1
                    })
                    
                    if agents:
                        provenance_chain["agents"].append({
                            "agent": agents[0],
                            "relation": relation,
                            "depth": depth
                        })
                
                elif rel_type == "wasInformedBy":
                    # This is a compliance check or decision
                    provenance_chain["compliance_checks"].append({
                        "relation": relation,
                        "depth": depth
                    })
            
            provenance_chain["depth_reached"] = max(provenance_chain["depth_reached"], depth)
        
        return provenance_chain
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance and completeness metrics."""
        completeness = self.get_completeness_metric()
        
        return {
            "trail_id": self.trail_id,
            "regulatory_context": self.regulatory_context,
            "performance": self.metrics,
            "completeness": completeness,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
    
    def cleanup_old_data(self, 
                        older_than_days: int = 90,
                        backup_before_cleanup: bool = True) -> Dict[str, int]:
        """
        Clean up old audit trail data.
        
        Args:
            older_than_days: Remove data older than this many days
            backup_before_cleanup: Whether to backup before cleanup
            
        Returns:
            Dictionary with cleanup statistics
        """
        cutoff_time = (datetime.datetime.utcnow() - 
                      datetime.timedelta(days=older_than_days)).isoformat()
        
        cleanup_stats = {
            "entities_removed": 0,
            "activities_removed": 0,
            "relations_removed": 0,
            "cutoff_time": cutoff_time
        }
        
        # Backup if requested
        if backup_before_cleanup:
            backup_data = self.export_trail(format="json", compress=True)
            backup_filename = f"audit_backup_{int(datetime.datetime.utcnow().timestamp())}.json.gz"
            
            with open(backup_filename, 'wb') as f:
                if isinstance(backup_data, dict):
                    import gzip
                    f.write(gzip.compress(json.dumps(backup_data).encode()))
                else:
                    f.write(backup_data)
            
            cleanup_stats["backup_file"] = backup_filename
        
        # Note: Actual cleanup implementation depends on storage backend
        # For JSONFileStorage, this would involve filtering and rewriting files
        # For production databases, this would use DELETE queries with timestamps
        
        warnings.warn("Cleanup functionality depends on storage backend implementation")
        
        return cleanup_stats


# Example usage and demonstration
if __name__ == "__main__":
    print("Demonstrating AuditTrailGenerator functionality...")
    
    # Initialize audit trail generator
    audit = AuditTrailGenerator(
        storage_path="./audit_logs",
        regulatory_context="EU_AI_ACT_HIGH_RISK"
    )
    
    # Log data ingestion
    dataset_info = {
        "dataset_id": "cleveland_heart_disease",
        "record_count": 303,
        "features": ["age", "sex", "cp", "trestbps", "chol"],
        "source": "UCI Machine Learning Repository",
        "ingestion_method": "batch_csv",
        "validation_status": "validated",
        "compliance_checks": ["pii_removed", "format_validated", "license_verified"],
        "quality_metrics": {"completeness": 0.98, "accuracy": 0.99}
    }
    
    dataset_entity_id = audit.log_data_ingestion(
        dataset_id="heart_disease_v1",
        dataset_info=dataset_info,
        data_hash="abc123def456789"
    )
    
    print(f"Logged dataset: {dataset_entity_id}")
    
    # Log model training
    model_info = {
        "model_id": "rf_heart_disease_classifier",
        "model_type": "RandomForest",
        "hyperparameters": {"n_estimators": 100, "max_depth": 10},
        "training_metrics": {"accuracy": 0.85, "auc": 0.91, "f1": 0.84},
        "validation_metrics": {"accuracy": 0.83, "auc": 0.89},
        "feature_names": ["age", "sex", "cp", "trestbps", "chol"],
        "target_names": ["no_disease", "disease"],
        "training_duration": 120.5,
        "hardware_info": {"cpus": 4, "memory_gb": 16, "gpu": False},
        "framework_version": "scikit-learn==1.0.0",
        "random_seed": 42,
        "compliance_checks": ["fairness_audit", "bias_assessment"],
        "fairness_metrics": {"disparate_impact": 0.85, "equal_opportunity": 0.88}
    }
    
    model_entity_id = audit.log_model_training(
        model_id="rf_classifier_v1",
        model_info=model_info,
        input_entity_ids=[dataset_entity_id]
    )
    
    print(f"Logged model: {model_entity_id}")
    
    # Log inference
    context = {
        "latency_ms": 45,
        "batch_size": 1,
        "request_id": "req_001",
        "user_id": "clinician_123",
        "hostname": "production-server-01",
        "python_version": "3.9.0",
        "framework_versions": {"scikit-learn": "1.0.0", "auditops": "1.0.0"},
        "compliance_checks": ["real_time_monitoring", "explanation_required"],
        "model_version": "1.0.0"
    }
    
    inference_entity_id = audit.log_inference(
        inference_id="inf_001",
        model_entity_id=model_entity_id,
        input_data={"age": 55, "sex": 1, "cp": 3, "trestbps": 130, "chol": 250},
        prediction={"class": 1, "confidence": 0.87, "probabilities": [0.13, 0.87]},
        explanation={
            "method": "SHAP",
            "feature_importance": {"age": 0.35, "chol": 0.28, "cp": 0.22, "trestbps": 0.15},
            "confidence": 0.92,
            "completeness_score": 0.95
        },
        context=context
    )
    
    print(f"Logged inference: {inference_entity_id}")
    
    # Log compliance check
    check_result = {
        "result": "PASSED",
        "details": "All thresholds met",
        "threshold": 0.70,
        "actual_value": 0.87,
        "deviation": 0.0,
        "severity": "INFO",
        "automated": True,
        "validation_method": "automated_threshold_check"
    }
    
    audit.log_compliance_check(
        check_type="performance_threshold",
        check_result=check_result,
        related_entity_ids=[inference_entity_id],
        regulatory_requirement="EU_AI_ACT_ARTICLE_13"
    )
    
    print("Logged compliance check")
    
    # Calculate and display metrics
    completeness = audit.get_completeness_metric(entity_type="inference")
    print(f"\nAudit Trail Completeness Metric (ATCM): {completeness['ATCM']:.3f}")
    print(f"Total entities captured: {completeness['total_entities']}")
    print(f"Field coverage: {completeness['field_coverage']}")
    
    # Export audit trail
    export_data = audit.export_trail(format="json", include_metrics=True)
    print(f"\nExported audit trail with {len(export_data.get('entities', []))} entities")
    
    # Query decision provenance
    provenance = audit.query_decision_provenance(inference_entity_id)
    print(f"\nDecision provenance depth: {provenance['depth_reached']}")
    print(f"Inputs used: {len(provenance['inputs'])}")
    print(f"Processes involved: {len(provenance['processes'])}")
    
    print("\nAuditTrailGenerator demonstration completed successfully!")
