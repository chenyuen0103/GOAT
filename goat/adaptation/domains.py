from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

DomainMethod = Literal["w2", "wasserstein", "ot", "fr", "fisher_rao", "natural", "eta"]


@dataclass(frozen=True)
class DomainGenerationConfig:
    method: DomainMethod = "w2"
    generated_domains: int = 0
    cov_type: str = "full"
    save_path: Optional[str] = None
    extra: dict[str, Any] | None = None


def generate_domains(config: DomainGenerationConfig, source: Any, target: Any):
    """Generate intermediate domains through the current legacy implementation.

    This provides one stable import location while preserving old semantics.
    Heavy dependencies are imported lazily only when this function is called.
    """

    method = config.method.lower()
    if method in {"w2", "wasserstein", "ot"}:
        from ot_util import generate_domains as generate_ot_domains

        return generate_ot_domains(config.generated_domains, source, target)
    if method in {"fr", "fisher_rao"}:
        from a_star_util import generate_fr_domains_between_optimized

        return generate_fr_domains_between_optimized(
            config.generated_domains,
            source,
            target,
            cov_type=config.cov_type,
            save_path=config.save_path,
            **(config.extra or {}),
        )
    if method in {"natural", "eta"}:
        from expansion_util import generate_gauss_domains

        return generate_gauss_domains(
            source,
            target,
            n_wsteps=config.generated_domains,
            **(config.extra or {}),
        )
    raise ValueError(f"Unsupported domain generation method: {config.method}")
