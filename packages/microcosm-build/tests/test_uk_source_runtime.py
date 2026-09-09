from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
)
from microcosm.build.uk_runtime.source_runtime import (
    uk_source_operation_handlers,
    uk_stage_implementations,
)
from microcosm.frame import EntitySchema, ExportContract, Frame, WeightKind, Weights
from microcosm.frame.schema import VariableMetadata


class StubRulesEngine:
    country = "uk"

    def __init__(self, values: Sequence[float] = (12.0, 34.0)) -> None:
        self._values = np.asarray(values, dtype=float)

    def variable_metadata(self, name: str) -> VariableMetadata:
        return VariableMetadata(
            name=name, entity="person", dtype="float", period="year"
        )

    def variables(self) -> Sequence[str]:
        return ("projected_income",)

    def entity_schema(self) -> EntitySchema:
        return _schema()

    def materialize(
        self,
        bundle: Frame,
        variables: Sequence[str],
        period: int | str,
    ) -> Mapping[str, np.ndarray]:
        assert variables == ("projected_income",)
        assert period == 2023
        assert bundle.n("person") == 2
        return {"projected_income": self._values}

    def export_contract(self) -> ExportContract:
        return ExportContract.empty()

    def write_dataset(
        self,
        bundle: Frame,
        path: str,
        period: int | str,
    ) -> None:
        raise NotImplementedError


def _schema() -> EntitySchema:
    return EntitySchema(group_entities=("household",))


def _frame() -> Frame:
    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "person_household_id": [10, 20],
                }
            ),
            "household": pd.DataFrame({"household_id": [10, 20]}),
        },
        _schema(),
        {
            "household": Weights(
                values=np.asarray([1.0, 2.0]),
                kind=WeightKind.DESIGN,
            )
        },
    )


def _operation() -> SourceOperationSpec:
    return SourceOperationSpec.from_mapping(
        {
            "kind": "materialize_rules_engine_predictors",
            "predictors": ["projected_income"],
        }
    )


def _context(*, engine: object, country: str | None = "uk") -> SourceRuntimeContext:
    extra: dict[str, object] = {"frame": _frame(), "rules_engine": engine}
    if country is not None:
        extra["country"] = country
    return SourceRuntimeContext(
        config=SourceRuntimeConfig(target_year=2023, extra=extra),
        tables={},
    )


def test_uk_stage_implementations_names_whole_stage_transforms() -> None:
    def retained(frame: Frame) -> Frame:
        return frame

    def hmrc(frame: Frame) -> Frame:
        return frame

    assert uk_stage_implementations(
        retained_leaves_transform=retained,
        hmrc_income_transform=hmrc,
        was_wealth_transform=retained,
        uc_deduction_attributes_transform=hmrc,
        regional_property_uprating_transform=hmrc,
        lcfs_consumption_transform=retained,
        etb_vat_transform=hmrc,
        etb_services_transform=retained,
        cgt_incidence_clone_transform=retained,
        cgt_band_donors_transform=hmrc,
        hmrc_cgt_gains_spine_transform=retained,
        salary_sacrifice_transform=hmrc,
        student_loans_transform=retained,
    ) == {
        "frs_hmrc_retained_leaves": retained,
        "hmrc_spi_income": hmrc,
        "was_wealth": retained,
        "uc_deduction_attributes": hmrc,
        "regional_property_uprating": hmrc,
        "lcfs_consumption": retained,
        "etb_vat": hmrc,
        "etb_services": retained,
        "cgt_incidence_clone": retained,
        "cgt_band_donors": hmrc,
        "hmrc_cgt_gains_spine": retained,
        "salary_sacrifice": hmrc,
        "student_loans": retained,
    }


def test_materialize_rules_engine_predictors_adds_declared_columns() -> None:
    handler = uk_source_operation_handlers()["materialize_rules_engine_predictors"]

    result = handler(None, _operation(), _context(engine=StubRulesEngine()))

    assert isinstance(result, Frame)
    assert result.table("person")["projected_income"].tolist() == [12.0, 34.0]
    assert result.weights_for("household").values.tolist() == [1.0, 2.0]


def test_materialize_rules_engine_predictors_refuses_country_mismatch() -> None:
    handler = uk_source_operation_handlers()["materialize_rules_engine_predictors"]
    engine = StubRulesEngine()
    engine.country = "us"

    with pytest.raises(SourceRuntimeError, match="does not match dataset country"):
        handler(None, _operation(), _context(engine=engine))


def test_materialize_rules_engine_predictors_asserts_country_without_extra() -> None:
    # Regression for the adversarial-review bypass: an absent ``country``
    # extra must not skip the engine assertion — the UK handler map serves
    # country "uk" by construction, so a wrong-country engine is refused
    # even when the caller forgets the optional extra.
    handler = uk_source_operation_handlers()["materialize_rules_engine_predictors"]
    engine = StubRulesEngine()
    engine.country = "us"

    with pytest.raises(SourceRuntimeError, match="does not match dataset country"):
        handler(None, _operation(), _context(engine=engine, country=None))


def test_materialize_rules_engine_predictors_refuses_non_uk_context() -> None:
    handler = uk_source_operation_handlers()["materialize_rules_engine_predictors"]

    with pytest.raises(SourceRuntimeError, match="only serve country 'uk'"):
        handler(
            None,
            _operation(),
            _context(engine=StubRulesEngine(), country="us"),
        )


class _PeriodRecordingEngine(StubRulesEngine):
    def __init__(self) -> None:
        super().__init__()
        self.periods: list[int | str] = []

    def materialize(
        self,
        bundle: Frame,
        variables: Sequence[str],
        period: int | str,
    ) -> Mapping[str, np.ndarray]:
        self.periods.append(period)
        return {"projected_income": self._values}


def test_materialize_rules_engine_predictors_honours_a_declared_year_rule() -> None:
    # The context says 2023; the declared rule names the release calibration
    # year, and the declaration wins (#862).
    handler = uk_source_operation_handlers()["materialize_rules_engine_predictors"]
    engine = _PeriodRecordingEngine()
    operation = SourceOperationSpec.from_mapping(
        {
            "kind": "materialize_rules_engine_predictors",
            "predictors": ["projected_income"],
            "year_rule": "calibration_year",
        }
    )

    result = handler(None, operation, _context(engine=engine))

    assert engine.periods == [2025]
    assert result.table("person")["projected_income"].tolist() == [12.0, 34.0]


def test_materialize_rules_engine_predictors_refuses_an_unknown_year_rule() -> None:
    handler = uk_source_operation_handlers()["materialize_rules_engine_predictors"]
    operation = SourceOperationSpec.from_mapping(
        {
            "kind": "materialize_rules_engine_predictors",
            "predictors": ["projected_income"],
            "year_rule": "frame_period",
        }
    )

    with pytest.raises(SourceRuntimeError, match="Unknown UK year_rule"):
        handler(None, operation, _context(engine=_PeriodRecordingEngine()))
