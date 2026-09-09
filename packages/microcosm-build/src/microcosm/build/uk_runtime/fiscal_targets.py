"""UK fiscal target coverage declarations.

The UK national calibration surface is currently ingested from
policyengine-uk-data via :func:`microcosm.calibrate.specs_from_pe_surface`, whose
docstring calls that "the transitional path from harness-extracted surfaces to
declared facts". This module starts the declared-fact side for the UK, following
``us_runtime.fiscal_targets``.

The first facts declared here are capital gains, because their absence is
measurable. The published ``populace-uk`` release (``populace-uk-2023-dd68c73``)
carries 149 targets and none of them constrain capital gains, so the calibrated
weights leave the gains distribution unanchored: reading ``capital_gains``
weighted by ``household_weight`` straight out of ``populace_uk_2023.h5`` gives
1.47m CGT taxpayers holding £96.0bn of gains at a £65,374 mean, against HMRC's
378k, £65.9bn and ~£174,000. The taxpayer count is 3.9x the administrative
figure and the mean gain is 62% below it — the imputation spreads gains across
far too many households in amounts that are individually far too small.

That is distributional error, not a level shift, so it survives any
revenue-side correction. It matters for any CGT analysis: on an uncalibrated
baseline the share of people affected by a rate reform comes out roughly three
times too high.

The CGT values no longer live in this module. They compile from
``uk/target_references.json`` against HMRC ``cgt_statistics`` Ledger facts at
build time. The measures remain person-entity, following the UK convention
established by
``uk_runtime.hmrc_calibration``: the measures are person-level, so the
constraint rows live on the person table while the calibrated weights stay on
the household table via the frame's household ``Weights``. Both therefore need
prepared columns on the person frame — the registry refuses callables so that
it can serialize, and count-like facts are documented to use prepared
indicator/count columns. See ``UK_CGT_REQUIRED_COLUMNS``, and
``uk_runtime.cgt_calibration`` for the materialization that prepares them.
"""

from __future__ import annotations

from microcosm.build.gates import TargetCoverageRequirement
from microcosm.calibrate import TargetRegistry

__all__ = [
    "UK_CGT_REQUIRED_COLUMNS",
    "UK_CGT_TARGET_COVERAGE_REQUIREMENTS",
    "UK_CGT_TARGET_SPECS",
    "UK_FISCAL_TARGET_REGISTRY",
]

#: Prepared person columns the frame must expose for these facts to compile.
#:
#: ``uk_cgt_measure_gains_amount`` is each person's chargeable gains, zeroed on
#: people who are not CGT taxpayers.
#: ``uk_cgt_measure_taxpayer_count`` is the 0/1 indicator for people whose
#: gains exceed the annual exempt amount **in force for the period** — the AEA moved
#: £12,300 -> £6,000 -> £3,000 across 2022-23 to 2024-25, so a fixed threshold
#: would silently mean different things in different years.
UK_CGT_REQUIRED_COLUMNS: tuple[str, ...] = (
    "uk_cgt_measure_gains_amount",
    "uk_cgt_measure_taxpayer_count",
)

# Values are Ledger-owned and compile from UK target references.
UK_CGT_TARGET_SPECS: tuple = ()

UK_CGT_TARGET_COVERAGE_REQUIREMENTS: tuple[TargetCoverageRequirement, ...] = (
    TargetCoverageRequirement(
        requirement_id="uk_capital_gains",
        label="HMRC individual capital gains, taxpayer counts and liability",
        accepted_names=(
            "hmrc.cgt.gains_total",
            "hmrc.cgt.taxpayers_total",
            "hmrc.cgt.liability_total",
        ),
        min_matches=3,
        notes=(
            "All three facts describe individuals in observed FY2024-25. "
            "Liability alone cannot anchor the gains distribution or incidence. "
            "The separately retained OBR cash forecast is an unresolved "
            "diagnostic and cannot substitute for observed liability."
        ),
    ),
)

UK_FISCAL_TARGET_REGISTRY = TargetRegistry(UK_CGT_TARGET_SPECS, country="uk")
