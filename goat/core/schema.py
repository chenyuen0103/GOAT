from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence


def _clean_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


@dataclass(frozen=True)
class ExperimentConfig:
    """Stable experiment configuration surface used by package CLIs."""

    dataset: str
    seed: int = 0
    gt_domains: int = 0
    generated_domains: int = 0
    small_dim: Optional[int] = None
    label_source: str = "pseudo"
    em_match: str = "prototypes"
    em_select: str = "bic"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = _clean_dict(asdict(self))
        payload["extra"] = dict(self.extra)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentConfig":
        known = {
            "dataset",
            "seed",
            "gt_domains",
            "generated_domains",
            "small_dim",
            "label_source",
            "em_match",
            "em_select",
            "extra",
        }
        extra = dict(payload.get("extra") or {})
        extra.update({key: value for key, value in payload.items() if key not in known})
        return cls(
            dataset=str(payload["dataset"]),
            seed=int(payload.get("seed", 0)),
            gt_domains=int(payload.get("gt_domains", payload.get("gt", 0))),
            generated_domains=int(
                payload.get("generated_domains", payload.get("gen", 0))
            ),
            small_dim=(
                None
                if payload.get("small_dim") is None
                else int(payload.get("small_dim"))
            ),
            label_source=str(payload.get("label_source", "pseudo")),
            em_match=str(payload.get("em_match", "prototypes")),
            em_select=str(payload.get("em_select", "bic")),
            extra=extra,
        )


@dataclass(frozen=True)
class RunSpec:
    """A concrete dataset/seed/domain-chain setting."""

    dataset: str
    seed: int
    gt_domains: int
    generated_domains: int
    target: Optional[int] = None
    degree: Optional[float] = None
    condition: Optional[str] = None
    skew: Optional[float] = None
    majority_class: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(asdict(self))


@dataclass(frozen=True)
class MethodResult:
    """Curves and metrics for one method inside a run record."""

    name: str
    train_curve: Sequence[float | None] = field(default_factory=list)
    test_curve: Sequence[float | None] = field(default_factory=list)
    st_curve: Sequence[float | None] = field(default_factory=list)
    st_all_curve: Sequence[float | None] = field(default_factory=list)
    generated_curve: Sequence[float | None] = field(default_factory=list)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    duration_sec: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "train_curve": list(self.train_curve),
            "test_curve": list(self.test_curve),
            "st_curve": list(self.st_curve),
            "st_all_curve": list(self.st_all_curve),
            "generated_curve": list(self.generated_curve),
            "metrics": dict(self.metrics),
            "duration_sec": self.duration_sec,
        }
        return _clean_dict(payload)

    @classmethod
    def from_legacy(cls, name: str, payload: Mapping[str, Any]) -> "MethodResult":
        metric_keys = set(payload) - {
            "train_curve",
            "test_curve",
            "st_curve",
            "st_all_curve",
            "generated_curve",
            "duration_sec",
        }
        return cls(
            name=name,
            train_curve=list(payload.get("train_curve") or []),
            test_curve=list(payload.get("test_curve") or []),
            st_curve=list(payload.get("st_curve") or []),
            st_all_curve=list(payload.get("st_all_curve") or []),
            generated_curve=list(payload.get("generated_curve") or []),
            metrics={key: payload[key] for key in sorted(metric_keys)},
            duration_sec=payload.get("duration_sec"),
        )


@dataclass(frozen=True)
class RunRecord:
    """Canonical JSONL record schema for experiment outputs."""

    dataset: str
    seed: int
    methods: Mapping[str, MethodResult]
    config: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    elapsed_sec: Optional[float] = None
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "schema_version": self.schema_version,
                "dataset": self.dataset,
                "seed": self.seed,
                "config": dict(self.config),
                "methods": {
                    key: result.to_dict() for key, result in self.methods.items()
                },
                "metrics": dict(self.metrics),
                "artifacts": dict(self.artifacts),
                "elapsed_sec": self.elapsed_sec,
            }
        )

    @classmethod
    def from_legacy_curve_record(
        cls,
        payload: Mapping[str, Any],
        *,
        dataset: str = "unknown",
        artifacts: Optional[Mapping[str, str]] = None,
    ) -> "RunRecord":
        config = {
            key: payload[key]
            for key in ("gt_domains", "generated_domains")
            if key in payload
        }
        return cls(
            dataset=dataset,
            seed=int(payload.get("seed", 0)),
            methods={
                name: MethodResult.from_legacy(name, method_payload)
                for name, method_payload in (payload.get("methods") or {}).items()
            },
            config=config,
            artifacts=dict(artifacts or {}),
            elapsed_sec=payload.get("elapsed"),
        )

