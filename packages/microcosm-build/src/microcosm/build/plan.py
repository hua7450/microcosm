"""Typed stage plans: the declarative skeleton of a dataset build.

A :class:`StagePlan` replaces the imperative build driver: each
:class:`Stage` declares its name, the columns it consumes, the columns it
produces, and — when it imputes — the :class:`DonorSpec` naming the survey it
draws from and its citation. The executor (:meth:`StagePlan.run`) materializes
stages in order over a :class:`~microcosm.frame.Frame` and enforces the
charter's loudness rules structurally:

- a stage whose declared inputs are missing fails before running;
- a stage that does not produce its declared outputs fails after running;
- **any exception inside a stage aborts the build** — there is no mechanism
  to continue past a failed stage, so the ORG-style silent fallback (catch,
  log, impute zeros) cannot be written in this framework;
- every stage leaves a :class:`StageRecord` (produced columns, donor,
  non-zero shares) — the evidence trail the observatory and release
  manifests consume.

The donor graph is the set of ``(stage, donor)`` edges; :meth:`StagePlan.donors`
returns it for documentation and the sources diagram.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from microcosm.frame import Frame

__all__ = [
    "DonorSpec",
    "Stage",
    "StageObservation",
    "StagePlan",
    "StageRecord",
]


@dataclass(frozen=True)
class DonorSpec:
    """The donor survey an imputation stage draws from.

    Attributes:
        survey: Human-readable survey name (e.g. ``"Fed SCF 2022"``).
        source: REQUIRED citation or URL for the survey vintage.
        notes: Free-text caveats (sample restrictions, vintages, uprating).

    Raises:
        ValueError: On an empty survey or source — an imputation from an
            unnamed donor is unauditable.
    """

    survey: str
    source: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.survey:
            raise ValueError("DonorSpec.survey must be non-empty.")
        if not self.source:
            raise ValueError(
                f"DonorSpec for {self.survey!r}: source citation is required."
            )


@dataclass(frozen=True)
class Stage:
    """One typed build stage.

    Attributes:
        name: Unique stage name (the manifest/progress label).
        transform: ``transform(frame) -> Frame`` — the stage body. Pure with
            respect to the input frame (frames are immutable); donor loading
            happens inside and any failure aborts the build.
        produces: Columns the stage must add (validated after the
            transform). Empty is allowed for assert/report-only stages.
        rewrites: Declared outputs that may have an earlier canonical producer
            and are intentionally replaced by this stage.
        consumes: Columns that must exist (on any entity) before the stage
            runs.
        donor: The donor survey, when the stage imputes. ``None`` for
            derivations from columns already on the frame.
    """

    name: str
    transform: Callable[[Frame], Frame]
    produces: tuple[str, ...] = ()
    rewrites: tuple[str, ...] = ()
    consumes: tuple[str, ...] = ()
    donor: DonorSpec | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Stage.name must be non-empty.")
        if not callable(self.transform):
            raise TypeError(
                f"Stage {self.name!r}: transform must be callable, got "
                f"{type(self.transform).__name__}."
            )


@dataclass(frozen=True)
class StageRecord:
    """Evidence one stage ran: what it produced and how populated it is.

    Attributes:
        stage: The stage's name.
        produced: The columns it added.
        donor_survey: The donor survey name, if the stage imputes.
        nonzero_share: Produced column -> share of records with a non-zero
            (or True) value — the layer-population evidence the parity gate
            and the observatory read.
        seconds: Wall-clock duration of the transform.
    """

    stage: str
    produced: tuple[str, ...]
    donor_survey: str | None
    nonzero_share: Mapping[str, float] = field(default_factory=dict)
    seconds: float = 0.0


@dataclass(frozen=True)
class StageObservation:
    """Aggregate-only notification emitted around one stage execution."""

    stage_id: str
    status: Literal["started", "completed", "failed"]
    elapsed_seconds: float
    produced_column_count: int
    entity_row_counts: Mapping[str, int]


class StagePlan:
    """An ordered, validated sequence of :class:`Stage` steps.

    Args:
        stages: The stages, in execution order.

    Raises:
        ValueError: On duplicate stage names, two stages declaring the same
            produced column (one canonical producer per column), or a stage
            consuming a column no earlier stage produces (columns expected
            on the *input* frame are checked at :meth:`run` instead, where
            the frame is known).
    """

    def __init__(self, stages: Iterable[Stage]) -> None:
        materialized = tuple(stages)
        names: set[str] = set()
        producers: dict[str, str] = {}
        for stage in materialized:
            if not isinstance(stage, Stage):
                raise TypeError(
                    f"StagePlan entries must be Stage instances, got "
                    f"{type(stage).__name__}."
                )
            if stage.name in names:
                raise ValueError(f"Duplicate stage name {stage.name!r}.")
            names.add(stage.name)
            for column in stage.produces:
                if column in producers:
                    if column not in stage.rewrites:
                        raise ValueError(
                            f"Column {column!r} is declared by two stages "
                            f"({producers[column]!r} and {stage.name!r}); every "
                            "column has one canonical producer unless a later "
                            "stage explicitly declares it as a rewrite."
                        )
                else:
                    producers[column] = stage.name
        self._stages = materialized

    @property
    def stages(self) -> tuple[Stage, ...]:
        return self._stages

    def donors(self) -> tuple[tuple[str, DonorSpec], ...]:
        """The donor graph: ``(stage name, donor)`` edges, in plan order."""
        return tuple(
            (stage.name, stage.donor)
            for stage in self._stages
            if stage.donor is not None
        )

    def run(
        self,
        frame: Frame,
        *,
        log: Callable[[str], None] = lambda message: None,
        observer: Callable[[StageObservation], None] | None = None,
    ) -> tuple[Frame, tuple[StageRecord, ...]]:
        """Execute the plan over ``frame``.

        Args:
            frame: The assembled input frame.
            log: Progress sink (one line per stage).
            observer: Optional aggregate-only stage lifecycle callback.

        Returns:
            The final frame and one :class:`StageRecord` per stage.

        Raises:
            ValueError: If a stage's consumed column is absent when it runs,
                or a declared produced column is still absent after it runs.
            Exception: Whatever a stage raises — stage failures abort the
                build by design; there is no skip/fallback path.
        """
        records: list[StageRecord] = []
        current = frame
        for stage in self._stages:
            started = time.perf_counter()
            if observer is not None:
                observer(
                    StageObservation(
                        stage_id=stage.name,
                        status="started",
                        elapsed_seconds=0.0,
                        produced_column_count=0,
                        entity_row_counts=_entity_row_counts(current),
                    )
                )
            try:
                for column in stage.consumes:
                    try:
                        current.column_entity(column)
                    except ValueError as error:
                        raise ValueError(
                            f"Stage {stage.name!r} consumes {column!r}, which is "
                            "not on the frame when the stage runs."
                        ) from error
                result = stage.transform(current)
                if not isinstance(result, Frame):
                    raise TypeError(
                        f"Stage {stage.name!r} must return a Frame, got "
                        f"{type(result).__name__}."
                    )
                shares: dict[str, float] = {}
                for column in stage.produces:
                    try:
                        entity = result.column_entity(column)
                    except ValueError as error:
                        raise ValueError(
                            f"Stage {stage.name!r} declared it produces "
                            f"{column!r} but the column is absent after the "
                            "stage ran."
                        ) from error
                    shares[column] = _nonzero_share(result.table(entity)[column])
            except Exception:
                if observer is not None:
                    observer(
                        StageObservation(
                            stage_id=stage.name,
                            status="failed",
                            elapsed_seconds=time.perf_counter() - started,
                            produced_column_count=0,
                            entity_row_counts=_entity_row_counts(current),
                        )
                    )
                raise
            elapsed = time.perf_counter() - started
            if observer is not None:
                observer(
                    StageObservation(
                        stage_id=stage.name,
                        status="completed",
                        elapsed_seconds=elapsed,
                        produced_column_count=len(stage.produces),
                        entity_row_counts=_entity_row_counts(result),
                    )
                )
            current = result
            record = StageRecord(
                stage=stage.name,
                produced=stage.produces,
                donor_survey=stage.donor.survey if stage.donor else None,
                nonzero_share=shares,
                seconds=elapsed,
            )
            records.append(record)
            donor_note = f" <- {stage.donor.survey}" if stage.donor is not None else ""
            log(
                f"stage {stage.name}{donor_note}: "
                f"{len(stage.produces)} column(s) in {elapsed:.1f}s"
            )
        return current, tuple(records)


def _entity_row_counts(frame: Frame) -> dict[str, int]:
    return {entity: int(len(frame.table(entity))) for entity in frame.entities}


def _nonzero_share(column: pd.Series) -> float:
    """Share of records carrying signal: non-zero numbers or True booleans.

    Non-numeric, non-boolean columns report the share of non-empty values.
    """
    if column.dtype == bool:
        return float(column.mean())
    if pd.api.types.is_numeric_dtype(column):
        values = pd.to_numeric(column, errors="coerce").fillna(0.0)
        return float((values != 0.0).mean())
    return float((column.astype(str).str.len() > 0).mean())
