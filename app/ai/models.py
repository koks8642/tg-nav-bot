"""Shared data contracts for the professional persona pipeline.

The engine is intentionally persona-agnostic.  Character-specific behaviour
comes from the profile data, while these structures stay reusable when rich
profiles are added for the other avatars.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ReplyPlan:
    respond: bool
    priority: str
    reason: str
    intent: str = "casual"
    register: str = "default"
    heat: int = 0
    needs_knowledge: bool = False
    search_query: str = ""
    entities: list[str] = field(default_factory=list)
    conversation_entities: list[str] = field(default_factory=list)
    emotion_target: str | None = None
    affinity_delta: int = 0
    risk_flags: list[str] = field(default_factory=list)
    memory_kind: str | None = None
    classifier_used: bool = False
    knowledge_scope: str = "relevant"
    world_scope: str = "native"

    def __post_init__(self) -> None:
        self.heat = max(0, min(3, int(self.heat)))
        self.affinity_delta = max(-3, min(3, int(self.affinity_delta)))
        if self.knowledge_scope not in {"relevant", "exact", "causal"}:
            self.knowledge_scope = "relevant"
        if self.world_scope not in {
                "native", "shared", "foreign", "conversation"}:
            self.world_scope = "native"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ReplyPlan | None":
        if not raw:
            return None
        allowed = cls.__dataclass_fields__
        return cls(**{k: v for k, v in raw.items() if k in allowed})


@dataclass
class MemoryEvent:
    kind: str
    summary: str
    importance: int = 1
    polarity: int = 0
    target: str | None = None
    persistent: bool = False
    source_msg_id: int | None = None
    count: int = 1
    first_seen: str = ""
    last_seen: str = ""
    resolved_by: int | None = None

    def __post_init__(self) -> None:
        self.importance = max(1, min(5, int(self.importance)))
        self.polarity = max(-3, min(3, int(self.polarity)))
        self.count = max(1, int(self.count))


@dataclass
class RelationshipState:
    affinity: int = 0
    trust: int = 0
    respect: int = 0
    familiarity: int = 0
    label: str = "нейтральное"
    reasons: list[str] = field(default_factory=list)


@dataclass
class ConversationState:
    topic: str = ""
    register: str = "default"
    heat: int = 0
    conflict: str = ""
    updated: str = ""


@dataclass
class KnowledgeItem:
    chapter: int
    text: str
    source: str = "summary"
    participants: list[str] = field(default_factory=list)
    perspective: str = "reportable"
    confidence: float = 1.0
    forbidden_secrets: list[str] = field(default_factory=list)
    relevance: str = "primary"
    epistemic_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeBundle:
    items: list[KnowledgeItem] = field(default_factory=list)
    query: str = ""

    @property
    def chapters(self) -> list[int]:
        return list(dict.fromkeys(i.chapter for i in self.items))

    @property
    def forbidden_secrets(self) -> list[str]:
        out: list[str] = []
        for item in self.items:
            out.extend(item.forbidden_secrets)
        return list(dict.fromkeys(out))

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query,
                "items": [item.to_dict() for item in self.items]}


@dataclass
class PromptBundle:
    system: str
    user: str
    compact_system: str
    compact_user: str
    selected_examples: list[str] = field(default_factory=list)
    selected_relationships: dict[str, str] = field(default_factory=dict)
    estimated_tokens: int = 0
    included_blocks: list[str] = field(default_factory=list)
    dropped_blocks: list[str] = field(default_factory=list)


@dataclass
class ModelCallResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    finish_reason: str = ""
    rate_limit_remaining_requests: str = ""
    rate_limit_remaining_tokens: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityReport:
    issues: list[str] = field(default_factory=list)
    severe: list[str] = field(default_factory=list)

    @property
    def should_retry(self) -> bool:
        return bool(self.severe)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
