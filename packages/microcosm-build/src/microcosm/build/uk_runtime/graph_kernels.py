"""Kernel family for the UK FRS spine graph.

Real-build kernels reconstruct a minimal UK :class:`~microcosm.frame.Frame`
from only their declared ``KernelContext.tables``, invoke the existing stage
transform unchanged, and project the owned cells back out.  Structural stages
return new-row source lineage through :attr:`KernelResult.expand`, cell
overlays through ``columns``, and explicit weights; the graph executor carries
the rows and remaps memberships.

Production execution binds transforms through the caller.  The hermetic H2
bundle carries a data-only marker from which an otherwise unbound registry
lazily reconstructs those same transforms.  Without that marker, unbound stage
kernels still fail explicitly rather than replaying recorded deltas: such a
replay would make parity agree with itself.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from microcosm.frame import Frame, MassChangeRecord, WeightKind
from microcosm.frame.adapters import policyengine_uk as uk_engine_adapter
from microcosm.graph import (
    Capabilities,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Numeric,
    SeedSource,
    StructuralDelta,
    load_source,
    source_hash,
)
from microcosm.graph.population import dtype_for_token

from . import uc_relationships
from .national_frame import UK_NATIONAL_SCHEMA
from .rowwise_geography import id_multiplier_for_values

if TYPE_CHECKING:
    from microcosm.build.source_manifest import SourceStageSpec

__all__ = [
    "UKClaimKernel",
    "UKCreateKernel",
    "UKExpandStageKernel",
    "UKIdentityKernel",
    "UKStageKernel",
    "build_uk_registry",
    "fixture_stage_plan_inputs",
]


_STAGE_MODULES = {
    "frs_spine": "frs_spine",
    "frs_employment": "frs_employment",
    "frs_council_tax": "frs_council_tax",
    "frs_disability": "frs_disability",
    "frs_education": "frs_education",
    "frs_legacy_proxies": "frs_legacy_proxies",
    "frs_education_grant_split": "frs_education_grants",
    "frs_take_up": "frs_take_up",
    "frs_person_draws": "frs_person_draws",
    "frs_household_draws": "frs_household_draws",
    "frs_brma": "frs_brma",
    "was_wealth": "was_wealth",
    "regional_property_uprating": "regional_uprating",
    "lcfs_consumption": "lcfs_consumption",
    "etb_vat": "etb_vat",
    "etb_services": "etb_services",
    "frs_hmrc_spine_leaves": "frs_hmrc_leaves",
    "spi_support_channel": "spi_spine",
    "hmrc_spi_income_spine": "spi_spine",
    "uc_reporter_redraw": "uc_reporter_redraw",
    "uc_capital_coherence": "uc_capital_coherence",
    "uc_deduction_attributes": "uc_deduction_attributes",
    "cgt_incidence_clone": "cgt_structure",
    "cgt_band_donors": "cgt_structure",
    "hmrc_cgt_gains_spine": "cgt_imputation",
    "salary_sacrifice": "salary_sacrifice",
    "student_loans": "student_loans",
    "age_tail": "age_tail",
}

# Imported modules are not traversed by ``source_hash``. Bind relationship
# helpers and the adapter's input-retention checks into every consuming stage.
_STAGE_HELPER_MODULES = {
    "frs_spine": (uc_relationships,),
    "frs_legacy_proxies": (uk_engine_adapter,),
    "frs_education_grant_split": (uk_engine_adapter,),
    "frs_brma": (uk_engine_adapter,),
    "was_wealth": (uk_engine_adapter,),
    "lcfs_consumption": (uk_engine_adapter,),
    "etb_vat": (uk_engine_adapter,),
    "etb_services": (uk_engine_adapter,),
    "uc_reporter_redraw": (uc_relationships, uk_engine_adapter),
    "uc_capital_coherence": (uc_relationships,),
}

_COMPUTE = Capabilities(
    determinism=Determinism.DETERMINISTIC,
    numeric=Numeric.BITWISE,
    seed_source=SeedSource.NONE,
)
_FILTER = Capabilities(
    determinism=Determinism.DETERMINISTIC,
    numeric=Numeric.BITWISE,
    seed_source=SeedSource.NONE,
    structural=StructuralDelta.FILTER,
)
_CREATE = Capabilities(
    determinism=Determinism.DETERMINISTIC,
    numeric=Numeric.BITWISE,
    seed_source=SeedSource.NONE,
    structural=StructuralDelta.CREATE,
)
_EXPAND = Capabilities(
    determinism=Determinism.DETERMINISTIC,
    numeric=Numeric.BITWISE,
    seed_source=SeedSource.NONE,
    structural=StructuralDelta.EXPAND,
)


def _stage_module(stage: str):
    try:
        leaf = _STAGE_MODULES[stage]
    except KeyError as error:
        raise ValueError(
            f"No implementation module is registered for {stage!r}."
        ) from error
    return importlib.import_module(f"microcosm.build.uk_runtime.{leaf}")


def _implementation_hash(kernel: object, stage: str, transform: object | None) -> str:
    # The stage module is the behavior-bearing source in every mode. Hashing
    # an injected transform's dynamic test wrapper would make
    # hermetic registries unhashable and, more importantly, would fail to bind
    # production edits made elsewhere in that stage's module.
    del transform
    return source_hash(
        type(kernel), _stage_module(stage), *_STAGE_HELPER_MODULES.get(stage, ())
    )


def _mass_log_payload(before: Frame, after: Frame) -> list[dict[str, object]]:
    prefix_length = len(before.mass_log)
    if after.mass_log[:prefix_length] != before.mass_log:
        raise ValueError("A UK stage replaced its incumbent Frame mass records.")
    return [
        {
            "entity": record.entity,
            "old_total": record.old_total,
            "new_total": record.new_total,
            "declared_factor": record.declared_factor,
            "reason": record.reason,
        }
        for record in after.mass_log[prefix_length:]
    ]


def _invoke_transform(transform: object, frame: Frame, context: KernelContext):
    """Invoke a stage, giving context-bound adapters only declared sources."""

    run_with_sources = getattr(transform, "run_with_sources", None)
    if callable(run_with_sources):
        return run_with_sources(frame, context.sources)
    return transform(frame)


def _json_mapping(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read UK parity fixture {label} at {path}.") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"UK parity fixture {label} at {path} must be an object.")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"UK parity fixture {label} must be an object.")
    return value


def _fixture_input(
    source: Path,
    inputs: Mapping[str, object],
    name: str,
) -> Path:
    relative = inputs.get(name)
    if not isinstance(relative, str) or not relative:
        raise ValueError(
            f"UK parity fixture inputs.{name} must be a non-empty relative path."
        )
    if Path(relative).is_absolute():
        raise ValueError(f"UK parity fixture inputs.{name} must be relative.")
    root = source.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"UK parity fixture inputs.{name} escapes its source directory."
        ) from error
    if not path.exists():
        raise ValueError(f"UK parity fixture input {name!r} is absent at {path}.")
    return path


def _fixture_hmrc_income_targets(path: Path):
    from .hmrc_income import (
        HMRCIncomeBandTargetRecord,
        HMRCIncomeSourceProvenance,
        HMRCIncomeTargetSet,
    )

    payload = _json_mapping(path, label="HMRC income targets")
    raw_source = dict(_mapping(payload.get("source"), label="HMRC source"))
    local_path = raw_source.get("local_path")
    table_names = raw_source.get("table_names")
    if not isinstance(local_path, str):
        raise ValueError("UK parity fixture HMRC source.local_path must be a string.")
    if not isinstance(table_names, list) or not all(
        isinstance(value, str) for value in table_names
    ):
        raise ValueError(
            "UK parity fixture HMRC source.table_names must be a string list."
        )
    raw_source["local_path"] = Path(local_path)
    raw_source["table_names"] = tuple(table_names)
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list):
        raise ValueError("UK parity fixture HMRC targets must be a list.")
    return HMRCIncomeTargetSet(
        source=HMRCIncomeSourceProvenance(**raw_source),
        targets=tuple(
            HMRCIncomeBandTargetRecord(
                **dict(_mapping(value, label="HMRC target record"))
            )
            for value in raw_targets
        ),
    )


def _fixture_cgt_distribution(path: Path):
    from .hmrc_capital_gains import (
        HMRCCapitalGainsBandTotal,
        HMRCCapitalGainsCell,
        HMRCCapitalGainsIncomeTotal,
        HMRCCapitalGainsJointDistribution,
        HMRCCapitalGainsSourceProvenance,
    )

    payload = _json_mapping(path, label="CGT distribution")
    raw_source = dict(_mapping(payload.get("source"), label="CGT source"))
    local_path = raw_source.get("local_path")
    if not isinstance(local_path, str):
        raise ValueError("UK parity fixture CGT source.local_path must be a string.")
    raw_source["local_path"] = Path(local_path)

    def records(name: str) -> list[Mapping[str, object]]:
        raw = payload.get(name)
        if not isinstance(raw, list):
            raise ValueError(f"UK parity fixture CGT {name} must be a list.")
        return [_mapping(value, label=f"CGT {name} record") for value in raw]

    return HMRCCapitalGainsJointDistribution(
        cells=tuple(HMRCCapitalGainsCell(**dict(value)) for value in records("cells")),
        band_totals=tuple(
            HMRCCapitalGainsBandTotal(**dict(value)) for value in records("band_totals")
        ),
        income_totals=tuple(
            HMRCCapitalGainsIncomeTotal(**dict(value))
            for value in records("income_totals")
        ),
        source=HMRCCapitalGainsSourceProvenance(**raw_source),
        total_individuals=float(payload["total_individuals"]),
        total_gains=float(payload["total_gains"]),
    )


def _fixture_descriptor(
    source: Path,
) -> tuple[Mapping[str, object], dict[str, SourceStageSpec]]:
    """Parse the H2 bundle's descriptor and its 27 stage specs, keyed by name."""

    from microcosm.build.source_manifest import SourceStageSpec

    descriptor = _json_mapping(source / "fixture.json", label="descriptor")
    if descriptor.get("schema_version") != "uk-spine-parity-fixture.v1":
        raise ValueError(
            "UK parity fixture schema_version must be 'uk-spine-parity-fixture.v1'."
        )
    stage_payloads = _mapping(descriptor.get("stages"), label="stages")
    stages: dict[str, SourceStageSpec] = {}
    for name, payload in stage_payloads.items():
        if not isinstance(name, str):
            raise ValueError("UK parity fixture stage names must be strings.")
        stage = SourceStageSpec.from_mapping(_mapping(payload, label=f"stage {name!r}"))
        if stage.stage != name:
            raise ValueError(
                f"UK parity fixture stage key {name!r} contains {stage.stage!r}."
            )
        stages[name] = stage
    if set(stages) != set(_STAGE_MODULES):
        missing = sorted(set(_STAGE_MODULES) - set(stages))
        extra = sorted(set(stages) - set(_STAGE_MODULES))
        raise ValueError(
            "UK parity fixture must describe the current 28-stage spine "
            f"(missing={missing}, extra={extra})."
        )
    return descriptor, stages


def _fixture_implementations(source: Path) -> Mapping[str, object]:
    """Reconstruct current transforms from one data-only H2 descriptor."""

    from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

    from .age_tail import UKAgeTailStageTransform
    from .cgt_imputation import UKCGTPolicyParameters, uk_cgt_spine_stage_transform
    from .cgt_structure import (
        UKCGTBandDonorStageTransform,
        UKCGTIncidenceCloneStageTransform,
    )
    from .etb_services import UKETBServicesStageTransform
    from .etb_vat import UKETBVATStageTransform
    from .frs_brma import UKFRSBRMAStageTransform
    from .frs_council_tax import UKFRSCouncilTaxStageTransform
    from .frs_disability import UKFRSDisabilityStageTransform
    from .frs_education import UKFRSEducationStageTransform
    from .frs_education_grants import UKFRSEducationGrantSplitStageTransform
    from .frs_employment import UKFRSEmploymentStageTransform
    from .frs_household_draws import UKFRSHouseholdDrawsStageTransform
    from .frs_legacy_proxies import UKFRSLegacyProxiesStageTransform
    from .frs_person_draws import UKFRSPersonDrawsStageTransform
    from .frs_take_up import UKFRSTakeUpStageTransform
    from .lcfs_consumption import UKLCFSConsumptionStageTransform
    from .regional_uprating import UKRegionalPropertyUpratingStageTransform
    from .salary_sacrifice import UKSalarySacrificeStageTransform
    from .spi_spine import (
        UKFRSHMRCSpineLeavesStageTransform,
        UKSPIIncomeSpineStageTransform,
        UKSPISupportChannelStageTransform,
    )
    from .student_loans import UKStudentLoansStageTransform
    from .take_up_contract import load_uk_take_up_contract
    from .uc_capital_coherence import UKUCCapitalCoherenceStageTransform
    from .uc_deduction_attributes import UKUCDeductionAttributesStageTransform
    from .uc_reporter_redraw import UKUCReporterRedrawStageTransform
    from .was_wealth import UKWASWealthStageTransform

    descriptor, stages = _fixture_descriptor(source)

    config = _mapping(descriptor.get("config"), label="config")
    inputs = _mapping(descriptor.get("inputs"), label="inputs")
    raw_dir = _fixture_input(source, inputs, "frs_raw")
    was = pd.read_csv(
        _fixture_input(source, inputs, "was"), float_precision="round_trip"
    )
    lcfs_household = pd.read_csv(
        _fixture_input(source, inputs, "lcfs_household"),
        float_precision="round_trip",
    )
    lcfs_person = pd.read_csv(
        _fixture_input(source, inputs, "lcfs_person"),
        float_precision="round_trip",
    )
    etb = pd.read_csv(
        _fixture_input(source, inputs, "etb"), float_precision="round_trip"
    )
    spi_path = _fixture_input(source, inputs, "spi_donor")
    spi_donor = pd.read_csv(spi_path, float_precision="round_trip")
    hmrc_targets_path = _fixture_input(source, inputs, "hmrc_income_targets")
    income_targets = _fixture_hmrc_income_targets(hmrc_targets_path)
    cgt_distribution = _fixture_cgt_distribution(
        _fixture_input(source, inputs, "cgt_distribution")
    )
    cgt_parameters = UKCGTPolicyParameters(
        **dict(_mapping(descriptor.get("cgt_parameters"), label="CGT parameters"))
    )

    engine = PolicyEngineUKEngine()
    contract = load_uk_take_up_contract()
    qrf_estimators = int(config["qrf_estimators"])
    donor_sample_size = int(config["spi_donor_sample_size"])
    sample_fraction = float(config["spi_sample_fraction"])
    calibration_year = int(config["student_loans_calibration_year"])
    return MappingProxyType(
        {
            "frs_employment": UKFRSEmploymentStageTransform(
                raw_dir, stage=stages["frs_employment"]
            ),
            "frs_council_tax": UKFRSCouncilTaxStageTransform(
                raw_dir, stage=stages["frs_council_tax"]
            ),
            "frs_disability": UKFRSDisabilityStageTransform(
                stage=stages["frs_disability"]
            ),
            "frs_education": UKFRSEducationStageTransform(
                raw_dir, stage=stages["frs_education"]
            ),
            "frs_legacy_proxies": UKFRSLegacyProxiesStageTransform(
                raw_dir,
                stage=stages["frs_legacy_proxies"],
                engine=engine,
            ),
            "frs_education_grant_split": UKFRSEducationGrantSplitStageTransform(
                stage=stages["frs_education_grant_split"], engine=engine
            ),
            "frs_take_up": UKFRSTakeUpStageTransform(
                contract=contract, stage=stages["frs_take_up"]
            ),
            "frs_person_draws": UKFRSPersonDrawsStageTransform(
                contract=contract, stage=stages["frs_person_draws"]
            ),
            "frs_household_draws": UKFRSHouseholdDrawsStageTransform(
                contract=contract, stage=stages["frs_household_draws"]
            ),
            "frs_brma": UKFRSBRMAStageTransform(
                stage=stages["frs_brma"], engine=engine
            ),
            "was_wealth": UKWASWealthStageTransform(
                stage=stages["was_wealth"], engine=engine, donor=was
            ),
            "regional_property_uprating": UKRegionalPropertyUpratingStageTransform(
                stage=stages["regional_property_uprating"]
            ),
            "lcfs_consumption": UKLCFSConsumptionStageTransform(
                stage=stages["lcfs_consumption"],
                engine=engine,
                lcfs_household=lcfs_household,
                lcfs_person=lcfs_person,
                was_donor=was,
            ),
            "etb_vat": UKETBVATStageTransform(
                stage=stages["etb_vat"], engine=engine, donor=etb
            ),
            "etb_services": UKETBServicesStageTransform(
                stage=stages["etb_services"], engine=engine, donor=etb
            ),
            "frs_hmrc_spine_leaves": UKFRSHMRCSpineLeavesStageTransform(
                raw_dir, stage=stages["frs_hmrc_spine_leaves"]
            ),
            "spi_support_channel": UKSPISupportChannelStageTransform(
                stage=stages["spi_support_channel"],
                sample_fraction=sample_fraction,
            ),
            "hmrc_spi_income_spine": UKSPIIncomeSpineStageTransform(
                spi_path,
                hmrc_targets_path,
                stage=stages["hmrc_spi_income_spine"],
                qrf_estimators=qrf_estimators,
                donor_sample_size=donor_sample_size,
                sampled_rung=True,
                donor_table=spi_donor,
                source_targets=income_targets,
            ),
            "uc_reporter_redraw": UKUCReporterRedrawStageTransform(
                stage=stages["uc_reporter_redraw"], engine=engine
            ),
            "uc_capital_coherence": UKUCCapitalCoherenceStageTransform(
                stage=stages["uc_capital_coherence"]
            ),
            "uc_deduction_attributes": UKUCDeductionAttributesStageTransform(
                stage=stages["uc_deduction_attributes"]
            ),
            "cgt_incidence_clone": UKCGTIncidenceCloneStageTransform(
                stage=stages["cgt_incidence_clone"]
            ),
            "cgt_band_donors": UKCGTBandDonorStageTransform(
                stage=stages["cgt_band_donors"]
            ),
            "hmrc_cgt_gains_spine": uk_cgt_spine_stage_transform(
                stages["hmrc_cgt_gains_spine"],
                cgt_distribution.source.local_path,
                distribution=cgt_distribution,
                parameters=cgt_parameters,
            ),
            "salary_sacrifice": UKSalarySacrificeStageTransform(
                stage=stages["salary_sacrifice"]
            ),
            "student_loans": UKStudentLoansStageTransform(
                stage=stages["student_loans"], calibration_year=calibration_year
            ),
            "age_tail": UKAgeTailStageTransform(stage=stages["age_tail"]),
        }
    )


def fixture_stage_plan_inputs(
    source: Path,
) -> tuple[tuple[SourceStageSpec, ...], Mapping[str, object]]:
    """The H2 bundle as legacy StagePlan inputs: ordered specs, every transform.

    The graph's CREATE node loads the captured root frame, so the kernel
    mapping carries no ``frs_spine`` transform; the legacy oracle applies one
    built from the same ``frs_raw`` tables, added here. The descriptor keys
    stages by name, so the committed UK manifest supplies the order the
    legacy plan runs them in.
    """

    from microcosm.build.country_spec import load_country_spec

    from .frs_spine import UKFRSSpineStageTransform

    descriptor, stages = _fixture_descriptor(source)
    committed = load_country_spec("uk")
    if committed.sources is None:
        raise ValueError("The committed UK country spec declares no sources.")
    ordered = tuple(
        stages[stage.stage]
        for stage in committed.sources.stages
        if stage.stage in stages
    )
    if len(ordered) != len(stages):
        unordered = sorted(set(stages) - {stage.stage for stage in ordered})
        raise ValueError(
            "UK parity fixture stages missing from the committed spine order: "
            f"{unordered}."
        )
    inputs = _mapping(descriptor.get("inputs"), label="inputs")
    raw_dir = _fixture_input(source, inputs, "frs_raw")
    implementations = dict(_fixture_implementations(source))
    implementations["frs_spine"] = UKFRSSpineStageTransform(
        raw_dir, stage=stages["frs_spine"]
    )
    return ordered, MappingProxyType(implementations)


class _FixtureTransformResolver:
    """Lazily bind the H2 fixture marker without weakening normal registries."""

    def __init__(self) -> None:
        self._by_source: dict[Path, Mapping[str, object]] = {}

    def resolve(self, stage: str, context: KernelContext) -> object | None:
        raw_source = context.sources.get("frs")
        if raw_source is None:
            return None
        source = Path(raw_source).resolve()
        if not (source / "fixture.json").is_file():
            return None
        implementations = self._by_source.get(source)
        if implementations is None:
            implementations = _fixture_implementations(source)
            self._by_source[source] = implementations
        try:
            return implementations[stage]
        except KeyError as error:
            raise ValueError(
                f"UK parity fixture has no transform for stage {stage!r}."
            ) from error


def _minimal_frame(context: KernelContext) -> Frame:
    missing = set(UK_NATIONAL_SCHEMA.entities) - context.tables.keys()
    if missing:
        raise ValueError(
            f"UK stage {context.node.id!r} lacks declared slices for {sorted(missing)}."
        )
    if "household" not in context.weights:
        raise ValueError(f"UK stage {context.node.id!r} lacks household weights.")
    mass_log: tuple[MassChangeRecord, ...] = ()
    if context.params.get("stage") == "hmrc_spi_income_spine":
        # KernelContext deliberately has no legacy Frame.mass_log channel.  The
        # SPI income transform consumes only the immediately preceding support
        # allocation reason, whose exact current-spine record is reconstructible
        # from the conserved household total and reviewed public reason.
        from .spi_support import SPI_PRIOR_MASS_CHANGE_REASON

        total = context.weights["household"].total
        mass_log = (
            MassChangeRecord(
                entity="household",
                old_total=total,
                new_total=total,
                declared_factor=1.0,
                reason=SPI_PRIOR_MASS_CHANGE_REASON,
            ),
        )
    return Frame(
        {
            entity: context.tables[entity].copy(deep=True)
            for entity in UK_NATIONAL_SCHEMA.entities
        },
        UK_NATIONAL_SCHEMA,
        {"household": context.weights["household"]},
        context.strata.copy(deep=True),
        mass_log=mass_log,
        metadata={"time_period": str(context.params.get("time_period", "2024"))},
    )


def _id_index(frame: Frame, entity: str) -> pd.Index:
    id_column = frame.schema.entity_id_column(entity)
    return pd.Index(frame.table(entity)[id_column].to_numpy(copy=True), name=id_column)


def _cast_owned(series: pd.Series, dtype: str) -> pd.Series:
    target = dtype_for_token(dtype)
    if series.dtype != target:
        series = series.astype(target)
    return series


def _owned_series(frame: Frame, entity: str, column: str, dtype: str) -> pd.Series:
    values = _cast_owned(frame.table(entity)[column].copy(deep=True), dtype)
    values.index = _id_index(frame, entity)
    values.name = column
    return values


def _normalize_create_frame(frame: Frame, context: KernelContext) -> Frame:
    tables = {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
    for owned in context.node.outputs:
        tables[owned.entity][owned.column] = _cast_owned(
            tables[owned.entity][owned.column], owned.dtype
        ).array
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata={
            **frame.metadata,
            "time_period": str(context.params.get("time_period", "2024")),
        },
    )


class UKCreateKernel(KernelBase):
    """Load fixture CSV tables or invoke the bound FRS root transform."""

    ref = "uk.create@1"
    capabilities = _CREATE

    def __init__(self, transform: object | None = None) -> None:
        self.transform = transform

    def implementation_hash(self) -> str:
        return _implementation_hash(self, "frs_spine", self.transform)

    def run(self, context: KernelContext) -> KernelResult:
        source = context.sources["frs"]
        if self.transform is None:
            frame = load_source("csv-tables", source)
        else:
            from .frs_spine import uk_frs_spine_seed_frame

            frame = _invoke_transform(
                self.transform, uk_frs_spine_seed_frame(), context
            )
        if not isinstance(frame, Frame):
            raise TypeError(
                f"The UK root transform returned {type(frame).__name__}, not Frame."
            )
        return KernelResult(frame=_normalize_create_frame(frame, context))


class UKIdentityKernel(KernelBase):
    """Keep every person, creating a new ownership/population boundary."""

    ref = "uk.identity@1"
    capabilities = _FILTER

    def run(self, context: KernelContext) -> KernelResult:
        person = context.tables[UK_NATIONAL_SCHEMA.person_entity]
        id_column = UK_NATIONAL_SCHEMA.person_id_column
        ids = pd.Index(person[id_column].to_numpy(copy=True), name=id_column)
        return KernelResult(keep=pd.Series(True, index=ids, dtype="bool"))


class UKClaimKernel(KernelBase):
    """Claim executor-materialized incumbent cells in a new population."""

    ref = "uk.claim@1"
    capabilities = _COMPUTE

    def run(self, context: KernelContext) -> KernelResult:
        columns = {
            (owned.entity, owned.column): pd.Series(
                _cast_owned(
                    context.tables[owned.entity][owned.column].copy(deep=True),
                    owned.dtype,
                ).array,
                index=pd.Index(
                    context.tables[owned.entity][
                        UK_NATIONAL_SCHEMA.entity_id_column(owned.entity)
                    ].to_numpy(copy=True),
                    name=UK_NATIONAL_SCHEMA.entity_id_column(owned.entity),
                ),
                name=owned.column,
                dtype=dtype_for_token(owned.dtype),
            )
            for owned in context.node.outputs
        }
        return KernelResult(columns=MappingProxyType(columns))


class UKStageKernel(KernelBase):
    """One ordinary UK stage, parameterized by its manifest stage name."""

    capabilities = _COMPUTE

    def __init__(
        self,
        stage: str,
        transform: object | None = None,
        fixture_resolver: _FixtureTransformResolver | None = None,
    ) -> None:
        self.stage = stage
        self.transform = transform
        self.fixture_resolver = fixture_resolver
        self.ref = f"uk.stage.{stage}@1"

    def implementation_hash(self) -> str:
        return _implementation_hash(self, self.stage, self.transform)

    def run(self, context: KernelContext) -> KernelResult:
        transform = self.transform
        if transform is None and self.fixture_resolver is not None:
            transform = self.fixture_resolver.resolve(self.stage, context)
        if transform is None:
            raise RuntimeError(
                f"UK stage {self.stage!r} has no bound production transform; "
                "recorded fixture deltas are not accepted as parity evidence."
            )
        before = _minimal_frame(context)
        after = _invoke_transform(transform, before, context)
        if not isinstance(after, Frame):
            raise TypeError(
                f"UK stage {self.stage!r} returned {type(after).__name__}, not Frame."
            )
        columns = {
            (owned.entity, owned.column): _owned_series(
                after, owned.entity, owned.column, owned.dtype
            )
            for owned in context.node.outputs
        }
        return KernelResult(
            columns=MappingProxyType(columns),
            receipt={
                "stage": self.stage,
                "frame_mass_log_append": _mass_log_payload(before, after),
            },
        )


def _expand_cells(context: KernelContext) -> tuple[tuple[str, str, str], ...]:
    raw = context.params.get("expand_cells")
    if not isinstance(raw, tuple):
        raise ValueError(f"UK EXPAND stage {context.node.id!r} has no cell contract.")
    return tuple(tuple(str(part) for part in item) for item in raw)  # type: ignore[arg-type]


def _source_lineage(
    before: Frame,
    after: Frame,
    entity: str,
    *,
    id_offset: int | None,
) -> pd.Series:
    """Derive new target-to-immediate-source ids from a structural result."""

    id_column = before.schema.entity_id_column(entity)
    before_table = before.table(entity)
    after_table = after.table(entity)
    before_ids = pd.Index(before_table[id_column])
    targets: list[object] = []
    values: list[object] = []
    source_column = f"{entity}_source_id"
    for _, row in after_table.iterrows():
        target = row[id_column]
        if target in before_ids:
            continue

        # CGT stages retain long-lived source-id provenance from the SPI
        # support stage, so their immediate clone lineage is the stage's
        # reviewed ID offset, not that older provenance column.
        if id_offset is not None:
            candidate = target - id_offset
            if candidate in before_ids:
                targets.append(target)
                values.append(candidate)
                continue
            raise ValueError(
                f"UK stage produced {entity!r} target id {target!r} whose "
                f"offset lineage {candidate!r} is not an incumbent id."
            )

        if source_column in after_table and row[source_column] in before_ids:
            targets.append(target)
            values.append(row[source_column])
            continue

        raise ValueError(
            f"UK stage produced {entity!r} target id {target!r} without an "
            f"explicit {source_column!r} or a declared ID-offset lineage rule."
        )
    return pd.Series(
        values,
        index=pd.Index(targets, name=id_column, dtype=before_table[id_column].dtype),
        dtype=before_table[id_column].dtype,
        name=id_column,
    )


class UKExpandStageKernel(KernelBase):
    """A UK row-expanding stage returning lineage rather than a Frame."""

    capabilities = _EXPAND

    def __init__(
        self,
        stage: str,
        transform: object | None = None,
        fixture_resolver: _FixtureTransformResolver | None = None,
    ) -> None:
        self.stage = stage
        self.transform = transform
        self.fixture_resolver = fixture_resolver
        self.ref = f"uk.stage.expand.{stage}@1"

    def implementation_hash(self) -> str:
        return _implementation_hash(self, self.stage, self.transform)

    def run(self, context: KernelContext) -> KernelResult:
        transform = self.transform
        if transform is None and self.fixture_resolver is not None:
            transform = self.fixture_resolver.resolve(self.stage, context)
        if transform is None:
            raise RuntimeError(
                f"UK stage {self.stage!r} has no bound production transform; "
                "recorded fixture deltas are not accepted as parity evidence."
            )
        before = _minimal_frame(context)
        after = _invoke_transform(transform, before, context)
        if not isinstance(after, Frame):
            raise TypeError(
                f"UK stage {self.stage!r} returned {type(after).__name__}, not Frame."
            )
        cells = _expand_cells(context)
        id_offset = None
        if self.stage in {"cgt_incidence_clone", "cgt_band_donors"}:
            id_offset = id_multiplier_for_values(
                *(
                    before.table(entity)[before.schema.entity_id_column(entity)]
                    for entity in before.entities
                ),
                *(
                    before.table(before.schema.person_entity)[
                        before.schema.membership_column(group)
                    ]
                    for group in before.schema.group_entities
                ),
            )
        expand: dict[str, pd.Series] = {}
        for entity in before.entities:
            expand[entity] = _source_lineage(
                before,
                after,
                entity,
                id_offset=id_offset,
            )
        columns: dict[tuple[str, str], pd.Series] = {}
        for entity, column, dtype in cells:
            columns[(entity, column)] = _owned_series(after, entity, column, dtype)
        weight_entity = str(context.params["expand_weight_entity"])
        after_weights = after.weights_for(weight_entity)
        declared_kind = WeightKind(str(context.params["expand_weight_kind"]))
        if after_weights.kind is not declared_kind:
            raise ValueError(
                f"UK EXPAND stage {self.stage!r} returned weight kind "
                f"{after_weights.kind.value!r}, not declared "
                f"{declared_kind.value!r}."
            )
        receipt: dict[str, object] = {
            "stage": self.stage,
            "frame_mass_log_append": _mass_log_payload(before, after),
        }
        if context.node.mass == "declared":
            receipt["mass"] = _declared_mass_receipt(
                before, after, stage=self.stage, weight_entity=weight_entity
            )
        return KernelResult(
            columns=MappingProxyType(columns),
            expand=MappingProxyType(expand),
            weights=after_weights,
            receipt=receipt,
        )


#: Relative tolerance for the weight-entity mass invariant a ``declared`` UK
#: expansion must still hold; equals the executor's own ledger tolerance.
_WEIGHT_ENTITY_MASS_RTOL = 1e-9


def _declared_mass_receipt(
    before: Frame,
    after: Frame,
    *,
    stage: str,
    weight_entity: str,
) -> dict[str, object]:
    """State the person-mass ledger of a ``declared`` expansion.

    The executor's mass ledger is weighted person mass per stratum, which an
    expansion that changes household composition cannot conserve even when it
    conserves the mass of the entity it reweights.  A ``declared`` UK expansion
    therefore states the person-mass ledger for the executor to verify and
    asserts here the invariant that is actually its contract: the weight
    entity's total mass is unchanged.
    """

    before_entity = float(before.weights_for(weight_entity).total)
    after_entity = float(after.weights_for(weight_entity).total)
    if not np.isclose(
        after_entity, before_entity, rtol=_WEIGHT_ENTITY_MASS_RTOL, atol=0.0
    ):
        raise ValueError(
            f"UK EXPAND stage {stage!r} declares its person-mass change but must "
            f"conserve {weight_entity!r} mass: {before_entity!r} -> {after_entity!r}."
        )
    before_mass = before.stratum_mass()
    after_mass = after.stratum_mass()
    return {
        "policy": "declared",
        "before": float(before_mass.sum()),
        "after": float(after_mass.sum()),
        "stratum_before": {key: float(value) for key, value in before_mass.items()},
        "stratum_after": {key: float(value) for key, value in after_mass.items()},
        "weight_entity": weight_entity,
        "weight_entity_mass_before": before_entity,
        "weight_entity_mass_after": after_entity,
    }


def build_uk_registry(
    graph: Graph,
    implementations: Mapping[str, object],
) -> KernelRegistry:
    """Bind graph refs to supplied transforms or the hermetic H2 descriptor."""

    stage_names = {
        str(node.params["stage"]) for node in graph.nodes if "stage" in node.params
    }
    unknown = set(implementations) - (stage_names | {"frs_spine"})
    if unknown:
        raise ValueError(f"Unknown UK graph stage implementations: {sorted(unknown)}.")
    if implementations:
        missing = stage_names - implementations.keys()
        if missing:
            raise ValueError(
                f"Missing UK graph stage implementations: {sorted(missing)}."
            )

    registry = KernelRegistry()
    registry.register(UKCreateKernel(implementations.get("frs_spine")))
    registry.register(UKIdentityKernel())
    registry.register(UKClaimKernel())
    fixture_resolver = _FixtureTransformResolver()
    for stage in sorted(stage_names):
        transform = implementations.get(stage)
        if stage in {
            "spi_support_channel",
            "cgt_incidence_clone",
            "cgt_band_donors",
        }:
            registry.register(UKExpandStageKernel(stage, transform, fixture_resolver))
        else:
            registry.register(UKStageKernel(stage, transform, fixture_resolver))

    required = {node.kernel for node in graph.nodes}
    if set(registry.refs()) != required:
        missing = sorted(required - set(registry.refs()))
        extra = sorted(set(registry.refs()) - required)
        raise ValueError(
            f"UK registry does not exactly cover its graph (missing={missing}, "
            f"extra={extra})."
        )
    return registry
