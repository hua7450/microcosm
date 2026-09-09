from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime import spi_income, spi_spine
from microcosm.build.uk_runtime.content_identity import uk_frame_content_identity
from microcosm.build.uk_runtime.hmrc_income import (
    HMRC_SPI_BUILD_PERIOD,
    HMRC_SPI_INCOME_BAND_LOWER_BOUNDS,
    HMRC_SPI_INCOME_COMPONENTS,
    HMRCIncomeBandTargetRecord,
    HMRCIncomeSourceProvenance,
    HMRCIncomeTargetSet,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.spi_income import (
    FRS_ONLY_SPI_FILL_PERSON_COLUMNS,
    SPI_DONOR_REQUIRED_COLUMNS,
    SPI_INCOME_QRF_OUTPUT_COLUMNS,
    impute_uk_spi_income_support,
)
from microcosm.build.uk_runtime.spi_spine import (
    EMPLOYER_PENSION_CONTRIBUTIONS_COLUMN,
    UKFRSHMRCSpineLeavesStageTransform,
    UKSPIIncomeSpineStageTransform,
    UKSPISupportChannelStageTransform,
    _assert_income_stage_parameters,
    _support_stage_parameters,
)
from microcosm.build.uk_runtime.spi_support import (
    HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    UKSPISupportResult,
    build_uk_spi_support_channel,
    support_channel_column,
)
from microcosm.build.uk_runtime.terminal_gates import UKZeroWeightStratumDeclaration
from microcosm.frame import WeightKind


def _base_frame(*, time_period: str = "2023") -> object:
    person = pd.DataFrame(
        {
            "person_id": [2001, 1001],
            "person_household_id": [2, 1],
            "person_benunit_id": [201, 101],
            "age": [44, 40],
            "gender": ["FEMALE", "MALE"],
            "employment_income": [20.0, 10.0],
            "self_employment_income": [0.0, 0.0],
            "savings_interest_income": [2.0, 1.0],
            "dividend_income": [20.0, 10.0],
            "private_pension_income": [0.0, 0.0],
            "property_income": [0.0, 0.0],
            "employee_pension_contributions": [3.0, 2.0],
            "employer_pension_contributions": [9.0, 6.0],
            "hmrc_spi_pay": [0.0, 0.0],
            "hmrc_spi_unemployment_benefit_income": [0.0, 0.0],
            "hmrc_spi_incapacity_benefit_income": [0.0, 0.0],
            "ossben_identifiable_subset": [0.0, 0.0],
            "srp_regular_code5": [0.0, 0.0],
            **{
                column: [0.0, 0.0]
                for column in FRS_ONLY_SPI_FILL_PERSON_COLUMNS
                if column
                not in {
                    "employee_pension_contributions",
                    "maternity_allowance_reported",
                }
            },
        }
    )
    benunit = pd.DataFrame({"benunit_id": [101, 201]})
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "household_weight": [10.0, 20.0],
            "region": ["LONDON", "SCOTLAND"],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=time_period,
        weight_kind=WeightKind.DESIGN,
    )


def _leaves_stage(
    tmp_path: Path,
    *,
    extra_adult_rows: tuple[dict[str, object], ...] = (),
) -> SourceStageSpec:
    adult = pd.DataFrame(
        [
            {"sernum": 1, "person": 1, "inearns": 4.0},
            {"sernum": 2, "person": 1, "inearns": 5.0},
            *extra_adult_rows,
        ]
    )
    benefits = pd.DataFrame(
        {
            "sernum": [1, 2, 2],
            "person": [1, 1, 1],
            "benefit": [14, 17, 5],
            "benamt": [1.0, 2.0, 3.0],
            "var2": [0, 0, 0],
        }
    )
    adult_path = tmp_path / "adult.tab"
    benefits_path = tmp_path / "benefits.tab"
    adult.to_csv(adult_path, sep="\t", index=False)
    benefits.to_csv(benefits_path, sep="\t", index=False)

    def artifact(path: Path, table: str) -> dict[str, object]:
        import hashlib

        return {
            "role": "frs_table",
            "table": table,
            "kind": "licensed_microdata",
            "format": "tab",
            "locator": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
            "runtime_sha256_required": True,
        }

    return SourceStageSpec.from_mapping(
        {
            "stage": "frs_hmrc_spine_leaves",
            "survey": "Synthetic FRS",
            "source": "local synthetic tabs",
            "grain": "person",
            "artifacts": [
                artifact(adult_path, "adult"),
                artifact(benefits_path, "benefits"),
            ],
            "operations": [
                {"kind": "retain_adjudicated_frs_hmrc_leaves"},
                {
                    "kind": "derive",
                    "output": EMPLOYER_PENSION_CONTRIBUTIONS_COLUMN,
                },
            ],
            "outputs": [
                "hmrc_spi_pay",
                "hmrc_spi_unemployment_benefit_income",
                "hmrc_spi_incapacity_benefit_income",
                "ossben_identifiable_subset",
                "srp_regular_code5",
                EMPLOYER_PENSION_CONTRIBUTIONS_COLUMN,
            ],
        }
    )


def test_spine_leaves_align_by_raw_person_id_not_position(tmp_path: Path) -> None:
    transform = UKFRSHMRCSpineLeavesStageTransform(
        tmp_path,
        stage=_leaves_stage(tmp_path),
    )

    result = transform(_base_frame())
    person = result.table("person")

    assert person["person_id"].tolist() == [2001, 1001]
    assert person["hmrc_spi_pay"].tolist() == pytest.approx(
        [5.0 * 365.25 / 7.0, 4.0 * 365.25 / 7.0]
    )
    assert person["hmrc_spi_incapacity_benefit_income"].tolist() == pytest.approx(
        [2.0 * 365.25 / 7.0, 0.0]
    )
    assert person[EMPLOYER_PENSION_CONTRIBUTIONS_COLUMN].tolist() == [9.0, 6.0]


def test_spine_leaves_fail_closed_on_unknown_raw_person(tmp_path: Path) -> None:
    stage = _leaves_stage(
        tmp_path,
        extra_adult_rows=({"sernum": 9, "person": 1, "inearns": 1.0},),
    )

    with pytest.raises(ValueError, match="absent from the raw spine"):
        UKFRSHMRCSpineLeavesStageTransform(tmp_path, stage=stage)(_base_frame())


def test_spine_leaves_sampled_rung_restricts_to_surviving_people(
    tmp_path: Path,
) -> None:
    # A #627 rung subsamples households after frs_spine, so raw-tab person
    # coverage legitimately exceeds the frame; the sampled posture restricts
    # the raw surface to the survivors while f100 keeps the strict fence.
    stage = _leaves_stage(
        tmp_path,
        extra_adult_rows=({"sernum": 9, "person": 1, "inearns": 1.0},),
    )

    result = UKFRSHMRCSpineLeavesStageTransform(
        tmp_path, stage=stage, sampled_rung=True
    )(_base_frame())
    person = result.table("person")

    assert person["person_id"].tolist() == [2001, 1001]
    assert person["hmrc_spi_pay"].tolist() == pytest.approx(
        [5.0 * 365.25 / 7.0, 4.0 * 365.25 / 7.0]
    )


def _support_stage() -> SourceStageSpec:
    return SourceStageSpec.from_mapping(
        {
            "stage": "spi_support_channel",
            "survey": "Synthetic FRS",
            "source": "local synthetic frame",
            "grain": "household",
            "artifacts": [],
            "operations": [
                {
                    "kind": "stack_zero_weight_donors",
                    "count": 10000,
                    "seed": 42,
                    "draw": "uniform_without_replacement",
                },
                {
                    "kind": "gate_zero_weight_strata",
                    "declarations": [
                        {
                            "name": "e7_spi_synthetic_preclone",
                            "selector": {HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN: True},
                            "maximum_zero_weight_rows": 10000,
                            "reason": "synthetic fixture",
                        }
                    ],
                },
                {
                    "kind": "allocate_zero_weight_prior_mass",
                    "share": 0.5,
                    "strata": ["region"],
                },
            ],
            "outputs": ["household_is_spi_synthetic"],
        }
    )


def test_support_transform_refuses_manifest_code_drift() -> None:
    stage = SourceStageSpec.from_mapping(
        {
            **_support_stage().__dict__,
            "operations": [
                {"kind": "stack_zero_weight_donors", "count": 9999, "seed": 42},
                {"kind": "gate_zero_weight_strata", "declarations": []},
                {
                    "kind": "allocate_zero_weight_prior_mass",
                    "share": 0.5,
                    "strata": ["region"],
                },
            ],
        }
    )

    with pytest.raises(ValueError, match="count drifted"):
        UKSPISupportChannelStageTransform(stage=stage)(_base_frame())


def test_preclone_support_gate_passes_at_limit_and_fails_above() -> None:
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "household_weight": [10.0, 20.0],
            "region": ["LONDON", "LONDON"],
        }
    )
    declaration = UKZeroWeightStratumDeclaration(
        name="e7_spi_synthetic_preclone",
        selector={HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN: True},
        maximum_zero_weight_rows=1,
        reason="synthetic fixture",
    )
    with pytest.raises(ValueError, match="exceed"):
        build_uk_spi_support_channel(
            person=_base_frame().table("person"),
            benunit=_base_frame().table("benunit"),
            household=household,
            spi_household_count=2,
            zero_weight_declarations=(declaration,),
        )


class _FakeQRF:
    events: list[tuple[str, tuple[str, ...], list[float]]] = []

    def __init__(self, n_estimators: int, seed: int) -> None:
        self.seed = seed
        self.targets: tuple[str, ...] = ()
        self.weight_kind = "unknown"

    def fit(self, frame, predictors, targets, weights):
        self.targets = tuple(targets)
        self.weight_kind = weights
        label = "stage1" if "hmrc_spi_pay" in self.targets else "stage2"
        dividend = (
            frame.table("person")["dividend_income"].tolist()
            if "dividend_income" in frame.table("person")
            else []
        )
        self.events.append((label, tuple(targets), dividend))
        return self

    def predict(self, predictors):
        rows = len(predictors)
        values: dict[str, np.ndarray] = {}
        for column in self.targets:
            if column == "hmrc_spi_miscellaneous_employment_income":
                values[column] = np.zeros(rows)
            elif column == "dividend_income":
                values[column] = 100.0 + np.arange(rows, dtype=float)
            elif column in {"gift_aid", "charitable_investment_gifts"}:
                values[column] = np.ones(rows)
            elif column == "savings_interest_income":
                values[column] = np.full(rows, 5.0)
            else:
                values[column] = np.full(rows, 2.0)
        return pd.DataFrame(values)


def _donor_file(tmp_path: Path) -> Path:
    path = tmp_path / "put2223uk.tab"
    row = {column: 0.0 for column in SPI_DONOR_REQUIRED_COLUMNS}
    row.update(
        {
            "SEX": 1,
            "FACT": 1.0,
            "GORCODE": 8,
            "AGERANGE": 1,
            "PAY": 10.0,
            "TEI": 10.0,
            "TI": 10.0,
        }
    )
    pd.DataFrame([row]).to_csv(path, sep="\t", index=False)
    return path


def _synthetic_hmrc_targets(path: Path) -> HMRCIncomeTargetSet:
    upper_bounds = (*HMRC_SPI_INCOME_BAND_LOWER_BOUNDS[1:], None)
    targets = tuple(
        HMRCIncomeBandTargetRecord(
            name=f"synthetic/{component}/{measure}/{lower_bound}",
            component=component,
            measure=measure,
            unit="people" if measure == "count" else "GBP",
            value=1_000.0 if measure == "count" else 1_000_000.0,
            period=HMRC_SPI_BUILD_PERIOD,
            total_income_lower_bound=lower_bound,
            total_income_upper_bound=upper_bound,
        )
        for lower_bound, upper_bound in zip(
            HMRC_SPI_INCOME_BAND_LOWER_BOUNDS,
            upper_bounds,
            strict=True,
        )
        for component in HMRC_SPI_INCOME_COMPONENTS
        for measure in ("count", "amount")
    )
    return HMRCIncomeTargetSet(
        source=HMRCIncomeSourceProvenance(
            local_path=path,
            sha256="0" * 64,
            publication_url="https://example.test/hmrc",
            ods_url="https://example.test/hmrc.ods",
            source_vintage="2023-24",
            source_tax_year="2023-24",
            source_tax_year_start=2023,
            build_period=HMRC_SPI_BUILD_PERIOD,
            table_names=("Table_3_6", "Table_3_7"),
            size_bytes=path.stat().st_size,
            mime_type="application/vnd.oasis.opendocument.spreadsheet",
        ),
        targets=targets,
    )


def test_spi_income_zero_initializes_frs_charity_and_redraws_dividends_after_stage2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeQRF.events = []
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    monkeypatch.setattr(
        spi_income,
        "_verify_spi_donor_identity",
        lambda path: SimpleNamespace(path=path),
    )
    monkeypatch.setattr(
        spi_income,
        "_refresh_disability_derived_inputs",
        lambda person, *, spi_people, category_rates=None, flag_rates=None: person,
    )
    support = build_uk_spi_support_channel(
        person=_base_frame().table("person"),
        benunit=_base_frame().table("benunit"),
        household=pd.DataFrame(
            {
                "household_id": [1, 2],
                "household_weight": [10.0, 20.0],
                "region": ["LONDON", "SCOTLAND"],
            }
        ),
        spi_household_count=2,
        zero_weight_declarations=(
            UKZeroWeightStratumDeclaration(
                name="e7_spi_synthetic_preclone",
                selector={HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN: True},
                maximum_zero_weight_rows=2,
                reason="synthetic fixture",
            ),
        ),
    )

    result = impute_uk_spi_income_support(
        support,
        _donor_file(tmp_path),
        donor_sample_size=None,
        initialize_frs_channel_columns={
            "gift_aid": 0.0,
            "charitable_investment_gifts": 0.0,
        },
        stage1_base_redraw_columns=("dividend_income",),
    )
    person = result.person
    base = person[support_channel_column("person")] != SPI_SYNTHETIC_SUPPORT_CHANNEL
    spi = ~base

    assert person.loc[base, "gift_aid"].tolist() == [0.0, 0.0]
    assert person.loc[base, "charitable_investment_gifts"].tolist() == [0.0, 0.0]
    # Unmeasured on the FRS instrument, so the base channel carries the
    # stage-time zero — the explicit initialization above and this fill are
    # now the same semantics, and the artifact ships no NaN.
    assert person.loc[base, "hmrc_spi_employment_benefits"].eq(0.0).all()
    assert person.loc[base, "dividend_income"].tolist() == [100.0, 101.0]
    assert person.loc[spi, "dividend_income"].tolist() == [100.0, 101.0]
    assert _FakeQRF.events[1][0] == "stage2"
    assert _FakeQRF.events[1][2] == [20.0, 10.0]
    assert set(SPI_INCOME_QRF_OUTPUT_COLUMNS) <= set(person.columns)


def test_reviewed_absent_incapacity_signal_raises(tmp_path: Path) -> None:
    _FakeQRF.events = []
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    monkeypatch.setattr(
        spi_income,
        "_verify_spi_donor_identity",
        lambda path: SimpleNamespace(path=path),
    )
    monkeypatch.setattr(
        spi_income,
        "_refresh_disability_derived_inputs",
        lambda person, *, spi_people, category_rates=None, flag_rates=None: person,
    )
    support = build_uk_spi_support_channel(
        person=_base_frame()
        .table("person")
        .assign(incapacity_benefit_reported=[1.0, 0.0]),
        benunit=_base_frame().table("benunit"),
        household=pd.DataFrame(
            {
                "household_id": [1, 2],
                "household_weight": [10.0, 20.0],
                "region": ["LONDON", "SCOTLAND"],
            }
        ),
        spi_household_count=2,
        zero_weight_declarations=(
            UKZeroWeightStratumDeclaration(
                name="e7_spi_synthetic_preclone",
                selector={HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN: True},
                maximum_zero_weight_rows=2,
                reason="synthetic fixture",
            ),
        ),
    )

    with pytest.raises(ValueError, match="now carries non-default source signal"):
        try:
            impute_uk_spi_income_support(
                support,
                _donor_file(tmp_path),
                donor_sample_size=None,
            )
        finally:
            monkeypatch.undo()


def _committed_stage(name: str) -> SourceStageSpec:
    from microcosm.build.country_spec import load_country_spec

    spec = load_country_spec("uk")
    assert spec.sources is not None
    return spec.sources.stage_map()[name]


def _with_mutated_operation(
    stage: SourceStageSpec, kind: str, **overrides: object
) -> SourceStageSpec:
    operations = []
    for operation in stage.operations:
        payload: dict[str, object] = {
            "kind": operation.kind,
            **dict(operation.parameters),
        }
        if operation.kind == kind:
            payload.update(overrides)
        operations.append(payload)
    return SourceStageSpec.from_mapping({**stage.__dict__, "operations": operations})


def test_spi_spine_parsed_inputs_match_the_path_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _FakeQRF.events = []
    monkeypatch.setattr(spi_income, "QRF", _FakeQRF)
    monkeypatch.setattr(
        spi_income,
        "_refresh_disability_derived_inputs",
        lambda person, *, spi_people, category_rates=None, flag_rates=None: person,
    )

    # This test checks parsed/path source resolution, independently of the engine.
    # Real index coverage lives in test_uk_spi_income.py.
    def uprating_factors(year):
        return (
            {column: 1.25 for column in SPI_INCOME_QRF_OUTPUT_COLUMNS},
            {"from_period": 2022, "to_period": year, "basis": "synthetic_test"},
        )

    monkeypatch.setattr(spi_income, "_spi_income_uprating_factors", uprating_factors)
    support_frame = UKSPISupportChannelStageTransform(
        stage=_committed_stage("spi_support_channel"),
        sample_fraction=0.0002,
    )(_base_frame(time_period=HMRC_SPI_BUILD_PERIOD))
    income_stage = _with_mutated_operation(
        _committed_stage("hmrc_spi_income_spine"),
        "fit_weighted_qrf_stage1",
        sample_size=1,
    )
    donor_path = _donor_file(tmp_path)
    ods_path = tmp_path / "synthetic-hmrc.ods"
    ods_path.write_bytes(b"synthetic hmrc source")
    source_targets = _synthetic_hmrc_targets(ods_path)
    ods_identity = object()
    resolved: list[str] = []

    def verify_donor(path):
        assert path == donor_path
        resolved.append("donor")
        return SimpleNamespace(path=donor_path.resolve())

    def verify_targets(path):
        assert path == ods_path
        resolved.append("target identity")
        return ods_identity

    def materialize_targets(identity, *, build_period):
        assert identity is ods_identity
        assert build_period == HMRC_SPI_BUILD_PERIOD
        resolved.append("targets")
        return source_targets

    monkeypatch.setattr(spi_spine, "verify_spi_donor_identity", verify_donor)
    monkeypatch.setattr(spi_spine, "verify_hmrc_spi_collated_ods", verify_targets)
    monkeypatch.setattr(
        spi_spine,
        "materialize_hmrc_spi_income_band_targets",
        materialize_targets,
    )
    path_transform = UKSPIIncomeSpineStageTransform(
        donor_path,
        ods_path,
        stage=income_stage,
        donor_sample_size=1,
        sampled_rung=True,
    )
    from_path = path_transform(support_frame)
    assert resolved == ["donor", "target identity", "targets"]

    def unexpected_loader(*_args, **_kwargs):
        raise AssertionError("parsed inputs must bypass source resolution")

    monkeypatch.setattr(
        spi_spine,
        "verify_spi_donor_identity",
        unexpected_loader,
    )
    monkeypatch.setattr(
        spi_spine,
        "verify_hmrc_spi_collated_ods",
        unexpected_loader,
    )
    monkeypatch.setattr(
        spi_spine,
        "materialize_hmrc_spi_income_band_targets",
        unexpected_loader,
    )
    monkeypatch.setattr(
        spi_income,
        "_verify_spi_donor_identity",
        unexpected_loader,
    )
    seam_transform = UKSPIIncomeSpineStageTransform(
        donor_path,
        ods_path,
        stage=income_stage,
        donor_sample_size=1,
        sampled_rung=True,
        donor_table=pd.read_csv(donor_path, delimiter="\t"),
        source_targets=source_targets,
    )
    from_seam = seam_transform(support_frame)

    assert uk_frame_content_identity(from_path) == uk_frame_content_identity(from_seam)
    assert path_transform.checkpoint_metadata() == seam_transform.checkpoint_metadata()


def test_income_stage_parameters_accept_the_committed_manifest() -> None:
    _assert_income_stage_parameters(
        _committed_stage("hmrc_spi_income_spine"),
        seed=42,
        qrf_estimators=100,
        donor_sample_size=100_000,
    )


def test_income_stage_parameters_refuse_redraw_column_drift() -> None:
    stage = _with_mutated_operation(
        _committed_stage("hmrc_spi_income_spine"),
        "redraw_columns_from_fitted_qrf",
        columns=["savings_interest_income"],
    )
    with pytest.raises(ValueError, match="base redraw columns drifted"):
        _assert_income_stage_parameters(
            stage, seed=42, qrf_estimators=100, donor_sample_size=100_000
        )


def test_income_stage_parameters_refuse_frs_initialization_drift() -> None:
    stage = _with_mutated_operation(
        _committed_stage("hmrc_spi_income_spine"),
        "fit_weighted_qrf_stage1",
        initialize_frs_channel_columns={
            "gift_aid": 1.0,
            "charitable_investment_gifts": 0.0,
        },
    )
    with pytest.raises(ValueError, match="initialization map drifted"):
        _assert_income_stage_parameters(
            stage, seed=42, qrf_estimators=100, donor_sample_size=100_000
        )


def test_income_stage_parameters_refuse_stage2_output_drift() -> None:
    committed = _committed_stage("hmrc_spi_income_spine")
    stage2 = next(
        operation
        for operation in committed.operations
        if operation.kind == "fit_weighted_qrf_stage2"
    )
    stage = _with_mutated_operation(
        committed,
        "fit_weighted_qrf_stage2",
        outputs=list(stage2.parameters["outputs"])[:-1],
    )
    with pytest.raises(ValueError, match="stage-2 outputs drifted"):
        _assert_income_stage_parameters(
            stage, seed=42, qrf_estimators=100, donor_sample_size=100_000
        )


def test_support_stage_parameters_accept_the_committed_manifest() -> None:
    count, share, strata, declarations = _support_stage_parameters(
        _committed_stage("spi_support_channel"), seed=42
    )
    assert count == 10000
    assert share == 0.5
    assert strata == ("region",)
    assert len(declarations) == 1


def test_support_transform_refuses_missing_builder_weight_kind(monkeypatch) -> None:
    frame = _base_frame()

    def _stub_builder(*_args, **_kwargs) -> UKSPISupportResult:
        return UKSPISupportResult(
            person=frame.table("person").copy(),
            benunit=frame.table("benunit").copy(),
            household=frame.table("household").copy(),
            id_multiplier=1,
            spi_household_ids=(),
            household_weight_kind=None,
        )

    monkeypatch.setattr(
        "microcosm.build.uk_runtime.spi_spine.build_uk_spi_support_channel",
        _stub_builder,
    )
    transform = UKSPISupportChannelStageTransform(
        stage=_committed_stage("spi_support_channel")
    )

    with pytest.raises(ValueError, match="importance household weights"):
        transform(frame)


def test_support_stage_parameters_refuse_gate_declaration_drift() -> None:
    committed = _committed_stage("spi_support_channel")
    gate = next(
        operation
        for operation in committed.operations
        if operation.kind == "gate_zero_weight_strata"
    )
    declaration = dict(gate.parameters["declarations"][0])
    declaration["maximum_zero_weight_rows"] = 20000
    stage = _with_mutated_operation(
        committed, "gate_zero_weight_strata", declarations=[declaration]
    )
    with pytest.raises(ValueError, match="gate declaration drifted"):
        _support_stage_parameters(stage, seed=42)
