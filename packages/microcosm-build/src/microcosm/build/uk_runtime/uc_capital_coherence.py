"""Late UK Universal Credit capital and take-up coherence stage.

Ordering and determinism contract (adversarial-review finding 4): this stage
runs after the last ``universal_credit_reported`` writer and BEFORE
``cgt_incidence_clone``, so the redraw sees only pre-clone benunit ids and
un-split design/prior-mass weights; clone twins then copy the already-drawn
values byte-for-byte, which is why clone re-keying cannot desynchronize them.
The redraw is deterministic in the twin-build sense used across the spine:
identical frame plus the declared seed reproduces identical draws (uniforms
are identity-keyed by ``benunit_id``; the donor CDF sorts by (capital,
benunit_id) with a stable sort). A different vintage or upstream frame
legitimately produces different draws, as with every seeded stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.frs_spine import UC_CAPITAL_UNAVAILABLE
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.spi_support import (
    BASE_FRS_SUPPORT_CHANNEL,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    support_channel_column,
)
from microcosm.build.uk_runtime.uc_relationships import (
    frs_uc_couple_mask,
)
from microcosm.frame import Frame

UC_CAPITAL_REDRAW_OUTPUT = "frs_benunit_capital"
UC_CAPITAL_REDRAW_SEED = 0
UC_CAPITAL_REDRAW_SALT = UC_CAPITAL_REDRAW_OUTPUT
UC_CAPITAL_COHERENCE_OUTPUT_COLUMNS = ("uc_reported_capital",)


@dataclass(frozen=True)
class UKUCCapitalCoherenceResult:
    """Output frame and receipt counts for the late coherence transform."""

    frame: Frame
    post_fill_reporter_count: int
    redrawn_spi_reporter_count: int
    refreshed_would_claim_count: int

    def evidence(self) -> dict[str, object]:
        """Return JSON-safe stage evidence."""

        return {
            "stage": "uc_capital_coherence",
            "post_fill_reporter_count": self.post_fill_reporter_count,
            "redrawn_spi_reporter_count": self.redrawn_spi_reporter_count,
            "refreshed_would_claim_count": self.refreshed_would_claim_count,
            "redraw_seed": UC_CAPITAL_REDRAW_SEED,
            "redraw_salt": UC_CAPITAL_REDRAW_SALT,
        }


@dataclass(frozen=True)
class UKUCCapitalCoherenceStageTransform:
    """Redraw SPI reporter capital and refresh UC take-up after SPI income."""

    stage: SourceStageSpec
    last_result: UKUCCapitalCoherenceResult | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        _assert_stage_parameters(self.stage)
        result = cohere_uc_capital(frame)
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UC_CAPITAL_COHERENCE_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        """Return the completed stage's redraw and refresh receipt."""

        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


def cohere_uc_capital(frame: Frame) -> UKUCCapitalCoherenceResult:
    """Make late SPI UC receipt, FRS capital, and take-up flags coherent."""

    validate_uk_national_frame(frame)
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    _require_columns(
        person,
        ("person_benunit_id", "person_household_id", "universal_credit_reported"),
        label="person",
    )
    _require_columns(
        benunit,
        (
            "benunit_id",
            support_channel_column("benunit"),
            "frs_benunit_capital",
            "dependent_children",
            "would_claim_uc",
        ),
        label="benunit",
    )
    _require_columns(household, ("household_id",), label="household")

    reporter = _post_fill_reporter_anchor(person, benunit)
    capital = pd.to_numeric(
        benunit[UC_CAPITAL_REDRAW_OUTPUT], errors="coerce"
    ).to_numpy(dtype=float, na_value=np.nan, copy=True)
    # Sentinel equality is exact (#833): -1.0 is a fixed literal every
    # producer writes verbatim, and an isclose band would reclassify a
    # corrupted near-sentinel value as a declared absence.
    valid_domain = np.isfinite(capital) & (
        (capital == UC_CAPITAL_UNAVAILABLE) | (capital >= 0.0)
    )
    if not valid_domain.all():
        raise ValueError(
            "frs_benunit_capital values must be exactly the named unavailable "
            "sentinel or nonnegative; the open interval between them has no "
            "meaning under the -1 contract."
        )

    channel = benunit[support_channel_column("benunit")].astype(str)
    base = channel.eq(BASE_FRS_SUPPORT_CHANNEL).to_numpy(dtype=bool)
    spi = channel.eq(SPI_SYNTHETIC_SUPPORT_CHANNEL).to_numpy(dtype=bool)
    if np.any(~(base | spi)):
        raise ValueError("UC capital coherence requires only FRS and SPI channels.")
    redraw = spi & reporter
    if redraw.any():
        _redraw_spi_reporter_capital(
            benunit,
            person=person,
            household=household,
            household_weights=frame.weights_for("household").values,
            reporter=reporter,
            base=base,
            redraw=redraw,
            capital=capital,
        )

    previous_would_claim = _boolean_values(benunit["would_claim_uc"])
    refreshed_would_claim = previous_would_claim | reporter
    benunit[UC_CAPITAL_REDRAW_OUTPUT] = capital
    benunit["uc_reported_capital"] = capital.copy()
    benunit["would_claim_uc"] = refreshed_would_claim

    result_frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result_frame)
    return UKUCCapitalCoherenceResult(
        frame=result_frame,
        post_fill_reporter_count=int(reporter.sum()),
        redrawn_spi_reporter_count=int(redraw.sum()),
        refreshed_would_claim_count=int((~previous_would_claim & reporter).sum()),
    )


def _post_fill_reporter_anchor(
    person: pd.DataFrame, benunit: pd.DataFrame
) -> np.ndarray:
    amounts = pd.to_numeric(
        person["universal_credit_reported"], errors="coerce"
    ).fillna(0.0)
    reporter_ids = person.loc[amounts > 0.0, "person_benunit_id"]
    return benunit["benunit_id"].isin(reporter_ids).to_numpy(dtype=bool)


def _redraw_spi_reporter_capital(
    benunit: pd.DataFrame,
    *,
    person: pd.DataFrame,
    household: pd.DataFrame,
    household_weights: np.ndarray,
    reporter: np.ndarray,
    base: np.ndarray,
    redraw: np.ndarray,
    capital: np.ndarray,
) -> None:
    weights = _household_to_benunit_weights(
        benunit,
        person=person,
        household=household,
        household_weights=household_weights,
    )
    child_band = _dependent_children_band(benunit["dependent_children"])
    couple = frs_uc_couple_mask(person, benunit)
    # Domain-validated upstream: every non-sentinel value is >= 0.
    available = capital >= 0.0
    donor = base & reporter & available & (weights > 0.0)
    target_ids = benunit["benunit_id"].to_numpy()
    draws = stable_identity_uniforms(
        target_ids,
        seed=UC_CAPITAL_REDRAW_SEED,
        salt=UC_CAPITAL_REDRAW_SALT,
    )

    for band, is_couple in sorted(
        set(zip(child_band[redraw], couple[redraw], strict=True))
    ):
        target_cell = redraw & (child_band == band) & (couple == is_couple)
        donor_cell = donor & (child_band == band) & (couple == is_couple)
        if not donor_cell.any():
            label = "3+" if band == 3 else str(band)
            raise ValueError(
                "UC capital redraw has no positive-weight base-FRS reporter "
                f"donors for dependent_children={label}, couple={is_couple}."
            )
        donor_rows = pd.DataFrame(
            {
                "benunit_id": target_ids[donor_cell],
                "capital": capital[donor_cell],
                "weight": weights[donor_cell],
            }
        ).sort_values(["capital", "benunit_id"], kind="mergesort")
        donor_values = donor_rows["capital"].to_numpy(dtype=float)
        donor_weights = donor_rows["weight"].to_numpy(dtype=float)
        cdf = np.cumsum(donor_weights) / float(donor_weights.sum())
        selected = np.searchsorted(cdf, draws[target_cell], side="right")
        capital[target_cell] = donor_values[np.minimum(selected, len(cdf) - 1)]


def _household_to_benunit_weights(
    benunit: pd.DataFrame,
    *,
    person: pd.DataFrame,
    household: pd.DataFrame,
    household_weights: np.ndarray,
) -> np.ndarray:
    placements = person[["person_benunit_id", "person_household_id"]].drop_duplicates()
    counts = placements.groupby("person_benunit_id", sort=False)[
        "person_household_id"
    ].nunique()
    if (counts != 1).any():
        raise ValueError("Every benefit unit must map to exactly one household.")
    household_by_benunit = placements.set_index("person_benunit_id")[
        "person_household_id"
    ]
    weight_by_household = pd.Series(
        np.asarray(household_weights, dtype=float),
        index=household["household_id"],
    )
    mapped_households = benunit["benunit_id"].map(household_by_benunit)
    weights = mapped_households.map(weight_by_household)
    if weights.isna().any():
        raise ValueError("Household weights do not cover every benefit unit.")
    values = weights.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("Mapped benefit-unit weights must be finite and nonnegative.")
    return values


def _dependent_children_band(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    if (
        not np.isfinite(numeric).all()
        or (numeric < 0.0).any()
        or not np.equal(numeric, np.floor(numeric)).all()
    ):
        raise ValueError("dependent_children must contain nonnegative integers.")
    return np.minimum(numeric, 3.0).astype(np.int8)


def _boolean_values(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.to_numpy(dtype=bool)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.isin((0, 1)).all():
        raise ValueError(f"{values.name} must contain only boolean/0/1 values.")
    return numeric.to_numpy(dtype=bool)


def _assert_stage_parameters(stage: SourceStageSpec) -> None:
    redraw = [
        operation
        for operation in stage.operations
        if operation.kind == "redraw_spi_reporter_capital"
    ]
    if len(redraw) != 1:
        raise ValueError(
            "uc_capital_coherence must declare one redraw_spi_reporter_capital "
            "operation."
        )
    parameters = redraw[0].parameters
    expected = {
        "output": UC_CAPITAL_REDRAW_OUTPUT,
        "seed": UC_CAPITAL_REDRAW_SEED,
        "salt": UC_CAPITAL_REDRAW_SALT,
        "couple_status": "is_uc_couple",
    }
    actual = {key: parameters.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            "uc_capital_coherence redraw parameters drifted: "
            f"expected {expected}, got {actual}."
        )


def _require_columns(
    frame: pd.DataFrame, columns: tuple[str, ...], *, label: str
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"UC capital coherence {label} columns missing: {missing}.")
