"""Feature registry: maps every design-matrix column to a semantic spec.

Every readout must trace back to a :class:`FeatureSpec`, so we can label a
``(target, source, timescale, channel)`` for each estimated coefficient.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureSpec:
    """Semantic description of one column of the feature matrix ``Psi``.

    ``kind`` is one of: ``"intercept"``, ``"filter"``, ``"own_history"``,
    ``"behavior"``, ``"stimulus"``.
    """

    name: str
    kind: str
    column_index: int
    source_id: str | None = None
    signal_kind: str | None = None
    timescale_s: float | None = None

    def label(self) -> str:
        if self.kind == "filter":
            return f"{self.source_id}[{self.signal_kind}]@{self.timescale_s:g}s"
        return self.name


@dataclass
class FeatureRegistry:
    """Ordered collection of :class:`FeatureSpec`, one per design-matrix column."""

    specs: list[FeatureSpec] = field(default_factory=list)

    def add(self, spec: FeatureSpec) -> None:
        if spec.column_index != len(self.specs):
            raise ValueError(
                f"column_index {spec.column_index} != next index {len(self.specs)}"
            )
        self.specs.append(spec)

    def add_intercept(self) -> None:
        self.add(FeatureSpec(name="intercept", kind="intercept",
                             column_index=len(self.specs)))

    def add_filter(self, source_id: str, signal_kind: str, timescale_s: float) -> None:
        idx = len(self.specs)
        self.add(FeatureSpec(
            name=f"filter:{source_id}:{signal_kind}:{timescale_s:g}",
            kind="filter", column_index=idx, source_id=source_id,
            signal_kind=signal_kind, timescale_s=float(timescale_s),
        ))

    def add_named(self, name: str, kind: str, source_id: str | None = None) -> None:
        self.add(FeatureSpec(name=name, kind=kind, column_index=len(self.specs),
                             source_id=source_id))

    def __len__(self) -> int:
        return len(self.specs)

    @property
    def n_features(self) -> int:
        return len(self.specs)

    @property
    def filter_specs(self) -> list[FeatureSpec]:
        return [s for s in self.specs if s.kind == "filter"]

    def index_of(self, predicate) -> list[int]:
        return [s.column_index for s in self.specs if predicate(s)]

    def column_names(self) -> list[str]:
        return [s.name for s in self.specs]
