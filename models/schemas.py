"""
NeuroForge Data Schemas
All Pydantic models for requests, responses, and internal state.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import uuid
from pydantic import BaseModel, Field


# ─── Enumerations ────────────────────────────────────────────────────────────

class TaskDomain(str, Enum):
    CLASSICAL_ML = "classical_ml"
    DEEP_LEARNING = "deep_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    LLM_FINETUNING = "llm_finetuning"
    HYBRID = "hybrid"
    MULTIMODAL = "multimodal"
    GRAPH_NEURAL = "graph_neural"
    TIME_SERIES = "time_series"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    UNKNOWN = "unknown"


class ComplexityLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    RESEARCH = "research"


class OutputDepth(str, Enum):
    SKETCH = "sketch"
    DETAILED = "detailed"
    COMPLETE = "complete"
    RESEARCH = "research"


class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    ANALYZER = "analyzer"
    ARCHITECT = "architect"
    CODER = "coder"
    OPTIMIZER = "optimizer"
    VALIDATOR = "validator"
    EXPLAINER = "explainer"


# ─── Architecture DNA ─────────────────────────────────────────────────────────

class LayerSpec(BaseModel):
    name: str
    type: str
    params: Dict[str, Any] = {}
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    notes: Optional[str] = None


class ConnectionSpec(BaseModel):
    from_layer: str
    to_layer: str
    connection_type: str = "sequential"  # sequential | skip | attention | gating


class TrainingSpec(BaseModel):
    optimizer: str
    learning_rate: float
    scheduler: Optional[str] = None
    batch_size: int
    epochs: int
    loss_function: str
    metrics: List[str] = []
    regularization: Dict[str, Any] = {}
    curriculum: Optional[List[str]] = None


class ComputeEstimate(BaseModel):
    parameters_millions: float
    flops_billions: Optional[float] = None
    gpu_memory_gb: Optional[float] = None
    training_time_estimate: Optional[str] = None
    recommended_hardware: str


class ArchitectureDNA(BaseModel):
    """
    Structured representation of any ML architecture.
    Can be serialized, compared, mutated, and visualized.
    """
    dna_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    domain: TaskDomain
    complexity: ComplexityLevel
    backbone: str
    layers: List[LayerSpec] = []
    connections: List[ConnectionSpec] = []
    training: Optional[TrainingSpec] = None
    compute: Optional[ComputeEstimate] = None
    novelty_score: float = 0.0          # 0–1, how novel vs standard architectures
    feasibility_score: float = 1.0      # 0–1, production feasibility
    anti_patterns_found: List[str] = []
    tags: List[str] = []
    rationale: str = ""


# ─── Dataset Profile ──────────────────────────────────────────────────────────

class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_pct: float
    unique_count: int
    sample_values: List[Any] = []
    stats: Dict[str, float] = {}


class DatasetProfile(BaseModel):
    path: str
    format: str
    rows: int
    columns: int
    size_mb: float
    column_profiles: List[ColumnProfile] = []
    inferred_task: Optional[TaskDomain] = None
    suggested_target: Optional[str] = None
    data_quality_score: float = 1.0
    recommendations: List[str] = []


# ─── Agent Trace (Memory Graph Node) ──────────────────────────────────────────

class AgentTrace(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    agent: AgentRole
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    input_summary: str
    output_summary: str
    full_output: str
    tokens_used: int = 0
    confidence: float = 1.0
    critique: Optional[str] = None


# ─── Session Memory ───────────────────────────────────────────────────────────

class SessionMemory(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    traces: List[AgentTrace] = []
    architecture_candidates: List[ArchitectureDNA] = []
    dataset_profile: Optional[DatasetProfile] = None
    final_architecture: Optional[ArchitectureDNA] = None
    conversation_history: List[Dict[str, str]] = []

    def add_trace(self, trace: AgentTrace) -> None:
        self.traces.append(trace)

    def get_context_summary(self) -> str:
        if not self.traces:
            return "No prior context."
        summaries = [f"[{t.agent.value}]: {t.output_summary}" for t in self.traces[-6:]]
        return "\n".join(summaries)


# ─── API Request / Response ───────────────────────────────────────────────────

class DesignRequest(BaseModel):
    problem_statement: str = Field(
        ...,
        description="Natural language description of the ML problem to solve.",
        example="Design a transformer-based model to classify medical X-ray images with limited labeled data."
    )
    depth: OutputDepth = OutputDepth.COMPLETE
    constraints: Optional[str] = Field(
        None,
        description="Hardware, latency, memory, or accuracy constraints."
    )
    dataset_path: Optional[str] = Field(
        None,
        description="Path to a local dataset file (CSV, Parquet, JSON) for profiling."
    )
    preferred_framework: Optional[str] = Field(
        None,
        description="PyTorch, TensorFlow, JAX, Keras, Scikit-learn, etc."
    )
    session_id: Optional[str] = Field(
        None,
        description="Continue an existing session."
    )
    compare_count: int = Field(
        1,
        ge=1,
        le=3,
        description="Number of candidate architectures to generate and compare."
    )


class AgentStep(BaseModel):
    agent: AgentRole
    status: str  # running | complete | skipped
    summary: str
    tokens_used: int = 0


class DesignResponse(BaseModel):
    session_id: str
    request_summary: str
    domain: TaskDomain
    complexity: ComplexityLevel
    depth: OutputDepth
    agent_pipeline: List[AgentStep]
    architectures: List[ArchitectureDNA]
    recommended: ArchitectureDNA
    full_design: str          # Complete markdown design document
    generated_code: str       # Ready-to-run Python code
    anti_patterns: List[str]
    compute_estimate: ComputeEstimate
    novelty_analysis: str
    training_guide: str
    evaluation_strategy: str
    total_tokens_used: int
    processing_time_sec: float


class CompareRequest(BaseModel):
    session_id: str
    architecture_ids: List[str]
    criteria: List[str] = ["accuracy", "speed", "memory", "simplicity", "scalability"]


class CompareResponse(BaseModel):
    session_id: str
    comparison_table: str
    winner_id: str
    rationale: str
    criteria_scores: Dict[str, Dict[str, float]]


class DatasetAnalysisRequest(BaseModel):
    dataset_path: str
    target_column: Optional[str] = None
    task_hint: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    version: str = "1.0.0"
    agents_available: List[str]
