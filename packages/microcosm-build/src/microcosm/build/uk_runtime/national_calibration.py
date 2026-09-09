"""Ledger-backed calibration stage for the UK national build."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from microcosm.build.plan import Stage
from microcosm.build.target_materialization import (
    MeasureResolution,
    resolve_target_measures,
)
from microcosm.build.uk_runtime.content_identity import uk_frame_content_identity
from microcosm.build.uk_runtime.ledger_targets import (
    UKFrameTargetAdapter,
    UKLedgerTargetCompilation,
    materialize_uk_ledger_targets,
)
from microcosm.build.uk_runtime.national_doctrine import (
    UK_NATIONAL_SOLVE_DOCTRINE,
    UKNationalSolveDoctrine,
    uk_national_target_loss_weights,
)

# This implementation is country-neutral despite its historical US runtime
# location. Reusing it keeps UK and US exact-count selection behavior aligned.
from microcosm.build.us_runtime.exact_k_ladder import (
    ExactKLadderCalibration,
    assert_exact_k_realized_count,
    calibrate_exact_k_ladder,
)
from microcosm.calibrate import (
    CalibrationResult,
    L0RefitResult,
    TargetRegistry,
    calibrate,
    effective_sample_size,
)
from microcosm.frame import Frame, WeightKind, Weights

__all__ = [
    "CalibrationFrameAdapter",
    "UKNationalCalibrationStage",
    "drop_injected_measure_inputs",
    "inject_measure_inputs",
    "national_calibration_mass_reason",
    "uk_national_calibration_stage",
]


UK_EXACT_K_L0_LAMBDA_SHARE = 0.8


class UKNationalCalibrationStage:
    """Fail-closed national calibration transform and its manifest evidence."""

    def __init__(
        self,
        registry: UKLedgerTargetCompilation | TargetRegistry,
        *,
        period: int,
        doctrine: UKNationalSolveDoctrine = UK_NATIONAL_SOLVE_DOCTRINE,
        measure_resolver: object | None = None,
        band_edge_registry: TargetRegistry,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        stage_callback: (
            Callable[[str, str, Mapping[str, object]], None] | None
        ) = None,
        exact_k: int | None = None,
        exact_k_pi_hi: float | None = None,
        exact_k_seed: int | None = None,
        exact_k_l0_lambda_share: float = UK_EXACT_K_L0_LAMBDA_SHARE,
    ) -> None:
        self.compilation = (
            registry
            if isinstance(registry, UKLedgerTargetCompilation)
            else UKLedgerTargetCompilation(registry=registry, unsupported=())
        )
        self.registry = self.compilation.registry
        # Required, never defaulted: the stage cannot tell a pruned registry
        # from a full one, so a fallback to self.registry would quietly
        # restate #792 for any caller holding an exclusion-pruned roster.
        # A caller whose registry is unpruned passes it explicitly.
        self.band_edge_registry = band_edge_registry
        # The materialization period is the declared calibration year the
        # registry was compiled at — never the input frame's base-year
        # time_period, which lags it (survey 2024, calibration 2025).
        if not isinstance(period, int) or isinstance(period, bool) or period <= 0:
            raise ValueError(
                f"period must be the declared calibration year, got {period!r}."
            )
        self.period = period
        self.doctrine = doctrine
        self.measure_resolver = measure_resolver
        self.progress_callback = progress_callback
        self.stage_callback = stage_callback
        exact_k_values = (exact_k, exact_k_pi_hi, exact_k_seed)
        if any(value is not None for value in exact_k_values) and any(
            value is None for value in exact_k_values
        ):
            raise ValueError(
                "exact_k, exact_k_pi_hi, and exact_k_seed must be provided together."
            )
        if exact_k is not None and (
            isinstance(exact_k, bool) or not isinstance(exact_k, int) or exact_k < 1
        ):
            raise ValueError("exact_k must be a positive integer.")
        if exact_k_pi_hi is not None and (
            not math.isfinite(exact_k_pi_hi) or not 0.0 <= exact_k_pi_hi <= 1.0
        ):
            raise ValueError("exact_k_pi_hi must be finite and in [0, 1].")
        if exact_k_seed is not None and (
            isinstance(exact_k_seed, bool)
            or not isinstance(exact_k_seed, int)
            or exact_k_seed < 0
        ):
            raise ValueError("exact_k_seed must be a non-negative integer.")
        if not math.isfinite(exact_k_l0_lambda_share) or exact_k_l0_lambda_share <= 0.0:
            raise ValueError("exact_k_l0_lambda_share must be positive and finite.")
        self.exact_k = exact_k
        self.exact_k_pi_hi = exact_k_pi_hi
        self.exact_k_seed = exact_k_seed
        self.exact_k_l0_lambda_share = float(exact_k_l0_lambda_share)
        self.manifest: dict[str, object] | None = None
        self.diagnostics: tuple[dict[str, object], ...] = ()
        self.solve_result: CalibrationResult | L0RefitResult | None = None
        self.output_content_identity: str | None = None

    def __call__(self, frame: Frame) -> Frame:
        declared = len(self.registry.specs) + len(self.compilation.unsupported)
        resolved = len(self.registry.specs)
        if self.compilation.unsupported:
            raise RuntimeError(
                "UK national calibration resolved "
                f"{resolved} of {declared} activated target references; "
                f"unsupported={self.compilation.unsupported!r}."
            )
        adapter = _CalibrationFrameAdapter(frame)
        original_columns = {
            entity: set(table.columns) for entity, table in adapter.tables.items()
        }
        self._notify_stage("measure_resolution", "started", {})
        measure_resolution = self._resolve_measures(frame)
        self._notify_stage(
            "measure_resolution",
            "completed",
            {
                "resolved_measure_count": (
                    0
                    if measure_resolution is None
                    else len(measure_resolution.measure_inputs)
                )
            },
        )
        if measure_resolution is not None:
            _inject_measure_inputs(adapter, measure_resolution.measure_inputs)
        materialized = materialize_uk_ledger_targets(
            adapter,
            self.registry,
            period=self.period,
            band_edge_registry=self.band_edge_registry,
        )
        if materialized.skipped:
            skipped = [skip.__dict__ for skip in materialized.skipped]
            raise RuntimeError(
                "UK national calibration could not materialize every activated "
                f"target reference: skipped={skipped}."
            )
        if measure_resolution is not None:
            _drop_injected_measure_inputs(
                adapter,
                measure_resolution.measure_inputs,
                original_columns,
            )
        prepared = adapter.prepared_frame()
        mass_reason = national_calibration_mass_reason(
            spec.family for spec in self.registry.specs
        )
        mass_log_records_before_calibration = len(frame.mass_log)
        # Target-set rows follow registry spec order, so the doctrine weight
        # vector aligns positionally; under the default "uniform" rule this
        # is None — the kernel's own equal weighting.
        target_loss_weights = uk_national_target_loss_weights(
            [spec.family for spec in self.registry.specs],
            rule=self.doctrine.target_weight_rule,
        )
        exact_k_outcome: ExactKLadderCalibration | None = None
        if self.exact_k is None:
            result: CalibrationResult | L0RefitResult = calibrate(
                prepared,
                self.registry.to_target_set(),
                weight_entity="household",
                epochs=self.doctrine.epochs,
                learning_rate=self.doctrine.learning_rate,
                mass=self.doctrine.mass_rule,
                mass_reason=mass_reason,
                max_weight_ratio=self.doctrine.max_weight_ratio,
                seed=self.doctrine.seed,
                l0_lambda=self.doctrine.l0_lambda,
                target_loss_cap=self.doctrine.target_loss_cap,
                target_loss_weights=target_loss_weights,
                progress_callback=self.progress_callback,
            )
        else:
            pool_households = int(prepared.n("household"))
            exact_k_outcome = calibrate_exact_k_ladder(
                prepared,
                self.registry.to_target_set(),
                k=self.exact_k,
                pi_hi=float(self.exact_k_pi_hi),
                seed=int(self.exact_k_seed),
                weight_entity="household",
                epochs=self.doctrine.epochs,
                refit_epochs=self.doctrine.epochs,
                learning_rate=self.doctrine.learning_rate,
                mass=self.doctrine.mass_rule,
                max_weight_ratio=self.doctrine.max_weight_ratio,
                l0_lambda=self.exact_k_l0_lambda_share / pool_households,
                target_loss_cap=self.doctrine.target_loss_cap,
                target_loss_weights=target_loss_weights,
                progress_callback=self.progress_callback,
            )
            assert_exact_k_realized_count(exact_k_outcome, self.exact_k)
            result = exact_k_outcome.result
        if result.skipped or len(result.problem.names) != declared:
            skipped = [item.name for item in result.skipped]
            raise RuntimeError(
                "UK national calibration matrix did not contain every activated "
                f"reference: declared={declared}, rows={len(result.problem.names)}, "
                f"skipped={skipped}."
            )
        clean_frame = adapter.restore(result.frame)
        self.solve_result = result
        calibration_record = _post_solve_calibration_record(
            frame,
            clean_frame,
            before_count=mass_log_records_before_calibration,
            exact_k_selected=(
                exact_k_outcome is not None and self.exact_k < prepared.n("household")
            ),
        )
        self.diagnostics = tuple(
            {
                "name": row.name,
                "estimate": row.final_estimate,
                "target": row.target,
                "relative_error": row.relative_error,
            }
            for row in result.diagnostics
        )
        ratios = result.weights / result.initial_weights
        old_total = float(calibration_record.old_total)
        new_total = float(calibration_record.new_total)
        before_kind = frame.weights_for("household").kind
        after_kind = clean_frame.weights_for("household").kind
        manifest = {
            "activated_reference_count": declared,
            "resolved_reference_count": resolved,
            "matrix_target_count": len(result.problem.names),
            "loss": result.final_loss,
            "effective_sample_size": effective_sample_size(result.weights),
            "max_weight_ratio": float(ratios.max()),
            "max_weight_ratio_bound": self.doctrine.max_weight_ratio,
            "target_materialization": materialized.report(),
            "weights": {
                "household_weight_kind": after_kind.value,
                "household_weight_kind_chain": [
                    {"stage": "staging", "kind": before_kind.value},
                    {"stage": "national_calibration", "kind": after_kind.value},
                ],
                "mass_log_records_before_calibration": (
                    mass_log_records_before_calibration
                ),
                "mass_log_records": len(clean_frame.mass_log),
                "calibration_mass_change": {
                    "entity": str(calibration_record.entity),
                    "old_total": old_total,
                    "new_total": new_total,
                    "relative_shift": (new_total - old_total) / old_total,
                    "declared_factor": calibration_record.declared_factor,
                    "reason": str(calibration_record.reason),
                },
            },
            "solve": {
                "n_targets": len(result.problem.names),
                "n_households": len(clean_frame.table("household")),
                "initial_loss": float(result.initial_loss),
                "final_loss": float(result.final_loss),
                "n_nonzero": int(np.count_nonzero(result.weights)),
            },
            "parameters": {"doctrine": _doctrine_bounds(self.doctrine)},
        }
        if exact_k_outcome is not None:
            manifest["exact_k_ladder"] = {
                "k": int(self.exact_k),
                "seed": int(self.exact_k_seed),
                "pool_households": int(prepared.n("household")),
                "realized_households": int(clean_frame.n("household")),
                "l0_lambda_share": self.exact_k_l0_lambda_share,
                "l0_lambda": self.exact_k_l0_lambda_share
                / int(prepared.n("household")),
                "selection_receipt": dict(exact_k_outcome.selection_receipt),
                "refit_baseline_diagnostics": dict(
                    exact_k_outcome.refit_baseline_diagnostics
                ),
            }
        if measure_resolution is not None:
            manifest["measure_resolution"] = dict(measure_resolution.receipt)
        self.manifest = manifest
        self.output_content_identity = uk_frame_content_identity(clean_frame)
        return clean_frame

    def _notify_stage(
        self, stage_id: str, status: str, details: Mapping[str, object]
    ) -> None:
        if self.stage_callback is not None:
            self.stage_callback(stage_id, status, details)

    def _resolve_measures(self, frame: Frame) -> MeasureResolution | None:
        if self.measure_resolver is None:
            return None
        resolve = getattr(self.measure_resolver, "resolve", None)
        if callable(resolve):
            return resolve(
                lambda: _CalibrationFrameAdapter(frame),
                self.registry,
                period=self.period,
                band_edge_registry=self.band_edge_registry,
            )
        return resolve_target_measures(
            lambda: _CalibrationFrameAdapter(frame),
            self.registry,
            self.measure_resolver,
            period=self.period,
            band_edge_registry=self.band_edge_registry,
        )

    def checkpoint_metadata(self) -> Mapping[str, object]:
        if self.manifest is None:
            raise RuntimeError("UK national calibration has not run.")
        return {
            "calibration": self.manifest,
            "diagnostics": self.diagnostics,
            "output_content_identity": self.output_content_identity,
        }

    def resume_from_checkpoint(
        self,
        metadata: Mapping[str, object],
        frame: Frame,
    ) -> None:
        """Rehydrate completed calibration evidence from its checkpoint record."""

        calibration = metadata.get("calibration")
        diagnostics = metadata.get("diagnostics")
        output_identity = metadata.get("output_content_identity")
        count_keys = (
            "activated_reference_count",
            "resolved_reference_count",
            "matrix_target_count",
        )
        if (
            not isinstance(calibration, Mapping)
            or not all(key in calibration for key in count_keys)
            or not isinstance(diagnostics, list)
            or not all(isinstance(row, Mapping) for row in diagnostics)
            or not isinstance(output_identity, str)
            or not output_identity
        ):
            raise RuntimeError(
                "UK national calibration resume requires the checkpoint record "
                "to carry calibration counts, diagnostics, and output content "
                "identity; a record without them cannot feed the calibration "
                "reference coverage gate or the drift check."
            )
        if uk_frame_content_identity(frame) != output_identity:
            raise RuntimeError(
                "UK national calibration checkpoint content does not match its "
                "recorded output identity; refusing to resume from a drifted "
                "record."
            )
        self.manifest = dict(calibration)
        self.diagnostics = tuple(dict(row) for row in diagnostics)
        self.output_content_identity = output_identity


def uk_national_calibration_stage(
    registry: UKLedgerTargetCompilation, **kwargs: Any
) -> Stage:
    """Return the named ordered-stage entry for national calibration."""

    transform = UKNationalCalibrationStage(registry, **kwargs)
    return Stage(name="national_calibration", transform=transform)


def national_calibration_mass_reason(bound_families: Sequence[str]) -> str:
    """The mass-record reason a national doctrine calibration declares."""

    families = sorted({str(name) for name in bound_families})
    if not families or any(not name.strip() for name in families):
        raise ValueError("bound_families must name at least one target family.")
    return (
        "National doctrine calibration to bound target family(ies) "
        f"{', '.join(families)}; total household mass moved with the targets."
    )


def _post_solve_calibration_record(
    before: Frame,
    after: Frame,
    *,
    before_count: int,
    exact_k_selected: bool = False,
):
    if after.weights_for("household").kind is not WeightKind.CALIBRATED:
        raise RuntimeError(
            "UK national calibration returned household weights whose kind is "
            f"{after.weights_for('household').kind.value!r}, not 'calibrated'."
        )
    expected_records = before_count + (2 if exact_k_selected else 1)
    if len(after.mass_log) != expected_records:
        expectation = "exactly two" if exact_k_selected else "exactly one"
        raise RuntimeError(
            f"UK national calibration must append {expectation} mass record(s); "
            f"before={before_count}, expected={expected_records}, "
            f"after={len(after.mass_log)}."
        )
    if before.mass_log != after.mass_log[:before_count]:
        raise RuntimeError(
            "UK national calibration changed pre-existing mass-log records."
        )
    record = after.mass_log[-1]
    calibration_reasons = ("calibration", "calibrated")
    if record.entity != "household" or not any(
        term in record.reason.casefold() for term in calibration_reasons
    ):
        raise RuntimeError(
            "UK national calibration latest mass record is not the calibration "
            f"record: entity={record.entity!r}, reason={record.reason!r}."
        )
    return record


def _doctrine_bounds(doctrine: UKNationalSolveDoctrine) -> dict[str, object]:
    return {
        "epochs": doctrine.epochs,
        "learning_rate": doctrine.learning_rate,
        "max_weight_ratio": doctrine.max_weight_ratio,
        "seed": doctrine.seed,
        "target_loss_cap": doctrine.target_loss_cap,
        "scale_rule": doctrine.scale_rule,
        "target_weight_rule": doctrine.target_weight_rule,
        "mass_rule": doctrine.mass_rule,
        "l0_lambda": doctrine.l0_lambda,
    }


def inject_measure_inputs(
    adapter: UKFrameTargetAdapter,
    measure_inputs: Mapping[tuple[str, str], np.ndarray],
) -> None:
    for (entity, variable), values in measure_inputs.items():
        adapter.tables[entity][variable] = values


def drop_injected_measure_inputs(
    adapter: UKFrameTargetAdapter,
    measure_inputs: Mapping[tuple[str, str], np.ndarray],
    original_columns: Mapping[str, set[str]],
) -> None:
    for entity, variable in measure_inputs:
        if variable not in original_columns[entity]:
            adapter.tables[entity].drop(columns=[variable], inplace=True)


class CalibrationFrameAdapter(UKFrameTargetAdapter):
    """The shared UK adapter plus the prepared-frame/restore lifecycle.

    Prepared measure columns are scratch state: they exist for constraint
    compilation only, and ``restore`` rebuilds pristine entity tables around
    the calibrated weights so the staged frame survives the HDFStore writer
    (slash-named scratch columns crash it).
    """

    def __init__(self, frame: Frame) -> None:
        super().__init__(frame)
        self._source_frame = frame
        self._original_tables = {
            name: table.copy() for name, table in self.tables.items()
        }

    def prepared_frame(self) -> Frame:
        return Frame(
            {**self.tables, **self.link_tables},
            self._source_frame.schema,
            {
                entity: self._source_frame.weights_for(entity)
                for entity in self._source_frame.weighted_entities
            },
            self._source_frame.strata,
            mass_log=self._source_frame.mass_log,
            metadata=self._source_frame.metadata,
        )

    def restore(self, calibrated: Frame) -> Frame:
        source = self._source_frame
        calibrated_households = calibrated.table("household")
        source_households = source.table("household")
        if len(calibrated_households) == len(source_households):
            tables = {
                name: table.copy() for name, table in self._original_tables.items()
            }
            tables.update(
                {name: table.copy() for name, table in self.link_tables.items()}
            )
        else:
            household_ids = calibrated_households[
                calibrated.schema.entity_id_column("household")
            ].to_numpy()
            membership = source.person[
                source.schema.membership_column("household")
            ].to_numpy()
            selected_source = source.select(np.isin(membership, household_ids))
            restored_household_ids = selected_source.table("household")[
                source.schema.entity_id_column("household")
            ].to_numpy()
            if not np.array_equal(restored_household_ids, household_ids):
                raise RuntimeError(
                    "Exact-k calibrated household order differs from the source "
                    "subset order; weights cannot be restored safely."
                )
            tables = {
                name: selected_source.table(name).copy()
                for name in selected_source.entities
            }
        return Frame(
            tables,
            calibrated.schema,
            {
                entity: Weights(
                    calibrated.weights_for(entity).values,
                    calibrated.weights_for(entity).kind,
                )
                for entity in calibrated.weighted_entities
            },
            calibrated.strata,
            mass_log=calibrated.mass_log,
            metadata=calibrated.metadata,
        )


def prepare_uk_target_frame(
    frame: Frame,
    registry: TargetRegistry,
    *,
    period: int | str,
    measure_resolver: object | None,
) -> tuple[Frame, Mapping[str, Any] | None]:
    """Materialize a registry's measures onto a frame, for scoring.

    The same resolve-inject-materialize route the calibration stage takes,
    without the solve: scoring a UK register against a raw exported H5 cannot
    work, because every packaged reference binds a slash-named prepared
    measure that calibration deliberately strips before export. A skipped
    target refuses rather than quietly shrinking the surface both sides are
    compared on.
    """

    resolution = None
    if measure_resolver is not None:
        resolution = resolve_target_measures(
            lambda: CalibrationFrameAdapter(frame),
            registry,
            measure_resolver,
            period=period,
        )
    adapter = CalibrationFrameAdapter(frame)
    if resolution is not None:
        inject_measure_inputs(adapter, resolution.measure_inputs)
    materialized = materialize_uk_ledger_targets(adapter, registry, period=period)
    if materialized.skipped:
        raise RuntimeError(
            "target measures did not materialize for scoring: "
            f"{[skip.__dict__ for skip in materialized.skipped]}."
        )
    return adapter.prepared_frame(), (
        None if resolution is None else dict(resolution.receipt)
    )


# Compatibility aliases for the established internal call sites.
_CalibrationFrameAdapter = CalibrationFrameAdapter
_inject_measure_inputs = inject_measure_inputs
_drop_injected_measure_inputs = drop_injected_measure_inputs
