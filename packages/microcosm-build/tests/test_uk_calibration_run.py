from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.chronicle_epoch import (
    CHRONICLE_CONSUMER_FACT_SCHEMA_VERSION,
    LEDGER_CONSUMER_FACT_SCHEMA_VERSION,
    PUBLISHED_CONSUMER_ARTIFACT_SCHEMA_VERSION,
)
from microcosm.build.country_spec import load_country_spec
from microcosm.build.gate_battery import (
    _canonical_json_bytes as canonical_json_bytes,
)
from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.uk_runtime import calibration_run
from microcosm.build.uk_runtime.calibration_run import (
    UK_CALIBRATION_GATE_SCOPE,
    UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS,
    UK_SPINE_GATE_SCOPE,
    UKCalibrationRunPaths,
    run_uk_calibration,
)
from microcosm.build.uk_runtime.etb_services import (
    UK_NHS_SPENDING_COMPONENT_COLUMNS,
)
from microcosm.build.uk_runtime.national_doctrine import UKNationalSolveDoctrine
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    uk_household_weight_kind,
    uk_national_frame,
    write_uk_national_frame,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import WeightKind

SIGNING_KEY = "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY="


@pytest.fixture(autouse=True)
def _signing_key(monkeypatch):
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", SIGNING_KEY)


def _frame():
    ids = np.arange(4, dtype="int64")
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": ids,
                "person_benunit_id": ids,
                "person_household_id": ids,
                "nhs_spending": [50.0, 50.0, 50.0, 50.0],
            }
        ),
        benunit=pd.DataFrame(
            {"benunit_id": ids, "universal_credit": [1.0, 1.0, 0.0, 0.0]}
        ),
        household=pd.DataFrame(
            {
                "household_id": ids,
                "region": "LONDON",
                "household_weight": [10.0, 10.0, 10.0, 10.0],
                "household_is_spi_synthetic": [False, False, False, False],
                "household_is_capital_gains_clone": [False, False, False, False],
                "electricity_consumption": [1.0, 1.0, 1.0, 1.0],
                "gas_consumption": [1.0, 1.0, 1.0, 1.0],
            }
        ),
        time_period="2023",
        weight_kind=WeightKind.DESIGN,
    )


def _registry():
    return TargetRegistry(
        [
            TargetSpec(
                name="dwp.uc.households",
                entity="benunit",
                measure="dwp/uc/households",
                value=20.0,
                source="test",
                family="dwp_universal_credit",
                metadata={"contract_target_id": "dwp.uc.households"},
            )
        ],
        country="uk",
    )


def _paths(tmp_path: Path) -> UKCalibrationRunPaths:
    return UKCalibrationRunPaths(
        input_h5=tmp_path / "input.h5",
        staging_h5=tmp_path / "staged.h5",
        diagnostics_json=tmp_path / "diagnostics.json",
        build_record_json=tmp_path / "build_record.json",
        terminal_gate_json=tmp_path / "terminal_gates.json",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_spine_sidecar(
    input_h5: Path,
    frame=None,
    **overrides,
) -> dict[str, object]:
    frame = _frame() if frame is None else frame
    sidecar = {
        "schema_version": 2,
        "pipeline": "uk-frs-spine",
        "stages": ["frs_spine", "was_wealth"],
        "stage_records": [
            {
                "stage": "was_wealth",
                "produced": ["property_wealth"],
                "nonzero_share": {"property_wealth": 1.0},
                "seconds": 0.1,
            }
        ],
        "stage_evidence": {
            "was_wealth": {
                "stage": "was_wealth",
                "support_clip": {"columns": {}},
            }
        },
        "artifact_pins": {"person": "a" * 64},
        "input_artifact_pins": {"was_qrf_donor": {"sha256": "b" * 64}},
        "resource_pins": {"wealth.json": "c" * 64},
        "stage_artifact_pins": {"was_wealth": {"was_qrf_donor": "d" * 64}},
        "declared_seeds": {"was_wealth": {"was_wealth": 0}},
        "rules_engine": {"package": "policyengine-uk", "version": "unavailable"},
        "source_vintages": {"frs": "2024_25"},
        "stochastic_contract_sha256": "e" * 64,
        "entity_row_counts": {
            entity: int(len(frame.table(entity))) for entity in frame.entities
        },
        "household_weight_kind": uk_household_weight_kind(frame).value,
        "household_weight_total": float(frame.weights_for("household").values.sum()),
    }
    sidecar.update(overrides)
    input_h5.with_suffix(".build.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    input_h5.with_suffix(".spine_gates.json").write_text(
        json.dumps(
            {
                "blocked_at_phase": None,
                "gates": {
                    gate_id: {
                        "criticality": "release_blocking",
                        "status": "passed",
                    }
                    for gate_id in UK_SPINE_GATE_SCOPE
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return sidecar


def _admin_anchor_values():
    values = {}
    for entry in load_country_spec("uk").gates.gates:
        if entry.id == "uk_aggregate_admin":
            for anchor in entry.parameters["anchors"]:
                values[str(anchor["name"])] = float(anchor["value"])
    return values


def test_gate_scope_classifies_every_uk_gate():
    all_ids = {entry.id for entry in load_country_spec("uk").gates.gates}
    assert (
        set(UK_CALIBRATION_GATE_SCOPE) | set(UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS)
        == all_ids
    )
    assert set(UK_CALIBRATION_GATE_SCOPE).isdisjoint(
        UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS
    )
    assert all(UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS.values())


def test_import_hygiene_does_not_load_national_build_in_fresh_subprocess():
    source = Path(calibration_run.__file__).read_text(encoding="utf-8")
    legacy_module = ".".join(("microcosm", "build", "uk_runtime", "national_build"))
    assert legacy_module not in source
    assert " ".join(("from", legacy_module, "import")) not in source


def test_run_uk_calibration_writes_cross_pinned_outputs(monkeypatch, tmp_path: Path):
    pytest.importorskip("tables")  # pandas HDF backend
    monkeypatch.setattr(
        calibration_run,
        "uk_aggregate_admin_totals",
        lambda frame, manifest: (_admin_anchor_values(), []),
    )
    input_h5 = tmp_path / "input.h5"
    frame = _frame()
    write_uk_national_frame(frame, input_h5)
    spine_sidecar = _write_spine_sidecar(input_h5, frame)
    paths = UKCalibrationRunPaths(
        input_h5=input_h5,
        staging_h5=tmp_path / "staged.h5",
        diagnostics_json=tmp_path / "diagnostics.json",
        build_record_json=tmp_path / "build_record.json",
        terminal_gate_json=tmp_path / "terminal_gates.json",
    )
    source_pins = {
        "input_h5": {"sha256": _sha(input_h5), "size_bytes": input_h5.stat().st_size},
        "ledger_facts": {"sha256": "a" * 64, "size_bytes": 1},
    }
    progress: list[dict[str, object]] = []
    events: list[tuple[str, str, dict[str, object]]] = []
    delivery = {
        "contract_version": 2,
        "enabled": False,
        "mode": "disabled",
        "run_id": None,
        "configured_repository": None,
        "upload_attempts": 0,
        "upload_successes": 0,
        "read_back": "not_requested",
        "last_error_code": None,
        "opt_out_reason": "test",
    }

    result = run_uk_calibration(
        paths=paths,
        input_sha256=_sha(input_h5),
        ledger_artifact=object(),
        register_registry=_registry(),
        band_edge_registry=_registry(),
        calibration_year=2025,
        exclusion_receipt={},
        doctrine=UKNationalSolveDoctrine(epochs=5),
        doctrine_overrides={},
        measure_resolver=None,
        source_pins=source_pins,
        run_config_extra={"calibration_year": 2025},
        release_id="test-run",
        progress_callback=progress.append,
        event_callback=lambda stage_id, status, details: events.append(
            (stage_id, status, dict(details))
        ),
        staging_delivery=delivery,
    )

    assert paths.staging_h5.exists()
    assert paths.diagnostics_json.exists()
    assert paths.build_record_json.exists()
    assert paths.terminal_gate_json.exists()
    assert progress and progress[0]["kind"] == "calibration_epoch"
    completed = {
        stage_id for stage_id, status, _details in events if status == "completed"
    }
    assert {
        "input_loading",
        "measure_resolution",
        "calibration",
        "diagnostics",
        "release_check_evaluation",
        "candidate_h5_creation",
        "build_record_creation",
    } <= completed
    assert result.build_record["staging_delivery"] == delivery
    assert result.build_record["artifacts"]["staging_h5"]["sha256"] == _sha(
        paths.staging_h5
    )
    assert result.build_record["artifacts"]["diagnostics_json"]["sha256"] == _sha(
        paths.diagnostics_json
    )
    assert result.build_record["artifacts"]["terminal_gate_json"]["sha256"] == _sha(
        paths.terminal_gate_json
    )
    # The record makes no shippability claim of its own — the hand-written
    # literal retired with the #757 release-cut audit — and instead points
    # at the certification artifact whose verdict is authoritative.
    assert "shippable" not in result.build_record
    assert "shippable_reason" not in result.build_record
    certification = result.build_record["certification"]
    assert certification["producer"] == "tools/certify_uk_release_cut.py"
    assert certification["expected_artifact"] == str(
        paths.staging_h5.with_suffix(".release_certification.json")
    )
    spine_provenance = result.build_record["spine_provenance"]
    assert spine_provenance["stages"] == spine_sidecar["stages"]
    assert spine_provenance["stage_records"] == spine_sidecar["stage_records"]
    assert spine_provenance["stage_evidence"] == spine_sidecar["stage_evidence"]
    assert spine_provenance["artifact_pins"] == spine_sidecar["artifact_pins"]
    assert (
        spine_provenance["input_artifact_pins"] == spine_sidecar["input_artifact_pins"]
    )
    assert spine_provenance["resource_pins"] == spine_sidecar["resource_pins"]
    assert (
        spine_provenance["stage_artifact_pins"] == spine_sidecar["stage_artifact_pins"]
    )
    assert spine_provenance["declared_seeds"] == spine_sidecar["declared_seeds"]
    assert spine_provenance["rules_engine"] == spine_sidecar["rules_engine"]
    assert spine_provenance["source_vintages"] == spine_sidecar["source_vintages"]
    assert (
        spine_provenance["stochastic_contract_sha256"]
        == spine_sidecar["stochastic_contract_sha256"]
    )
    diagnostics = json.loads(paths.diagnostics_json.read_text())
    assert diagnostics["build"]["spine_provenance"] == spine_provenance
    staged, _ = load_uk_national_frame(paths.staging_h5)
    assert staged.weights_for("household").kind is WeightKind.CALIBRATED
    report = json.loads(paths.terminal_gate_json.read_text())
    assert report["posture"] == "calibration_seam"
    assert set(report["scope_exclusions"]) == set(UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS)
    attestation = report["attestation"]
    signature = attestation["signature"]
    attestation["signature"] = None
    key = b"0123456789abcdef0123456789abcdef"
    assert (
        hmac.new(key, canonical_json_bytes(report), hashlib.sha256).hexdigest()
        == signature
    )
    assert result.logbook_spool.exists()


def test_run_uk_calibration_writes_exact_k_candidate_and_receipt(
    monkeypatch, tmp_path: Path
) -> None:
    pytest.importorskip("tables")
    monkeypatch.setattr(
        calibration_run,
        "uk_aggregate_admin_totals",
        lambda frame, manifest: (_admin_anchor_values(), []),
    )
    frame = _frame()
    paths = _paths(tmp_path)
    write_uk_national_frame(frame, paths.input_h5)
    _write_spine_sidecar(paths.input_h5, frame)

    result = run_uk_calibration(
        paths=paths,
        input_sha256=_sha(paths.input_h5),
        ledger_artifact=object(),
        register_registry=_registry(),
        band_edge_registry=_registry(),
        calibration_year=2025,
        exclusion_receipt={},
        doctrine=UKNationalSolveDoctrine(epochs=3),
        doctrine_overrides={},
        measure_resolver=None,
        source_pins={
            "input_h5": {
                "sha256": _sha(paths.input_h5),
                "size_bytes": paths.input_h5.stat().st_size,
            }
        },
        run_config_extra={"calibration_year": 2025},
        release_id="exact-k-integration",
        exact_k=2,
        exact_k_pi_hi=1.0,
        exact_k_seed=17,
    )

    staged, _ = load_uk_national_frame(paths.staging_h5)
    assert staged.n("household") == 2
    assert staged.n("benunit") == 2
    assert staged.n("person") == 2
    exact_k = result.build_record["calibration"]["exact_k_ladder"]
    assert exact_k["k"] == 2
    assert exact_k["realized_households"] == 2
    assert exact_k["pool_households"] == 4
    assert exact_k["selection_receipt"]["design"] == "sampford"
    diagnostics = json.loads(paths.diagnostics_json.read_text())
    assert diagnostics["build"]["exact_k_ladder"] == exact_k


def test_run_uk_calibration_requires_the_band_edge_register(
    tmp_path: Path,
):
    # Required, never defaulted: an empty receipt is a claim that nothing was
    # pruned, not permission to skip the reconciliation, so the seam takes no
    # register-without-edges path at all (#803 review findings 1 and 3).
    paths = _paths(tmp_path)

    with pytest.raises(TypeError, match="band_edge_registry"):
        run_uk_calibration(
            paths=paths,
            input_sha256="a" * 64,
            ledger_artifact=object(),
            register_registry=_registry(),
            calibration_year=2025,
            exclusion_receipt={"excluded.target": {"reason": "reviewed"}},
            doctrine=UKNationalSolveDoctrine(epochs=1),
            doctrine_overrides={},
            measure_resolver=None,
            source_pins={},
            run_config_extra={},
            release_id="pruned-without-edge-register",
        )

    assert not paths.staging_h5.exists()
    assert not paths.diagnostics_json.exists()
    assert not paths.build_record_json.exists()


def test_run_uk_calibration_reconciles_an_empty_receipt_as_no_prunes(
    tmp_path: Path,
):
    # A pruned register handed in with an empty receipt must refuse: with
    # nothing declared excluded, the two rosters have to be name-identical.
    paths = _paths(tmp_path)
    full = _registry()
    pruned = TargetRegistry([], country="uk")

    with pytest.raises(ValueError, match="exclusion receipt"):
        run_uk_calibration(
            paths=paths,
            input_sha256="a" * 64,
            ledger_artifact=object(),
            register_registry=pruned,
            band_edge_registry=full,
            calibration_year=2025,
            exclusion_receipt={},
            doctrine=UKNationalSolveDoctrine(epochs=1),
            doctrine_overrides={},
            measure_resolver=None,
            source_pins={},
            run_config_extra={},
            release_id="empty-receipt-pruned-register",
        )

    assert not paths.staging_h5.exists()
    assert not paths.diagnostics_json.exists()
    assert not paths.build_record_json.exists()


def test_run_uk_calibration_refuses_incoherent_band_edge_register(tmp_path: Path):
    paths = _paths(tmp_path)
    edge_registry = TargetRegistry(
        [
            *_registry().specs,
            TargetSpec(
                name="different.excluded",
                entity="benunit",
                measure="different/excluded",
                value=1.0,
                source="test",
                metadata={"contract_target_id": "different.excluded"},
            ),
        ],
        country="uk",
    )

    with pytest.raises(ValueError, match="exclusion receipt"):
        run_uk_calibration(
            paths=paths,
            input_sha256="a" * 64,
            ledger_artifact=object(),
            register_registry=_registry(),
            band_edge_registry=edge_registry,
            calibration_year=2025,
            exclusion_receipt={"other.excluded": {"reason": "reviewed"}},
            doctrine=UKNationalSolveDoctrine(epochs=1),
            doctrine_overrides={},
            measure_resolver=None,
            source_pins={},
            run_config_extra={},
            release_id="incoherent-edge-register",
        )

    assert not paths.staging_h5.exists()
    assert not paths.diagnostics_json.exists()
    assert not paths.build_record_json.exists()


def test_run_uk_calibration_records_band_edge_register_sha256(
    monkeypatch, tmp_path: Path
):
    pytest.importorskip("tables")  # pandas HDF backend
    monkeypatch.setattr(
        calibration_run,
        "uk_aggregate_admin_totals",
        lambda frame, manifest: (_admin_anchor_values(), []),
    )
    input_h5 = tmp_path / "input.h5"
    frame = _frame()
    write_uk_national_frame(frame, input_h5)
    _write_spine_sidecar(input_h5, frame)
    paths = _paths(tmp_path)
    register = _registry()
    edge_registry = TargetRegistry(
        [
            TargetSpec(
                name="dwp.uc.households",
                entity="benunit",
                measure="dwp/uc/households",
                value=99.0,
                source="test",
                family="dwp_universal_credit",
                metadata={"contract_target_id": "dwp.uc.households"},
            )
        ],
        country="uk",
    )

    result = run_uk_calibration(
        paths=paths,
        input_sha256=_sha(input_h5),
        ledger_artifact=object(),
        register_registry=register,
        band_edge_registry=edge_registry,
        calibration_year=2025,
        exclusion_receipt={},
        doctrine=UKNationalSolveDoctrine(epochs=5),
        doctrine_overrides={},
        measure_resolver=None,
        source_pins={
            "input_h5": {
                "sha256": _sha(input_h5),
                "size_bytes": input_h5.stat().st_size,
            }
        },
        run_config_extra={},
        release_id="band-edge-provenance",
    )

    assert (
        result.build_record["run_config"]["band_edge_register_sha256"]
        == edge_registry.version
    )


def test_run_uk_calibration_refuses_input_sha_before_outputs(tmp_path: Path):
    pytest.importorskip("tables")  # pandas HDF backend
    input_h5 = tmp_path / "input.h5"
    write_uk_national_frame(_frame(), input_h5)
    paths = UKCalibrationRunPaths(
        input_h5=input_h5,
        staging_h5=tmp_path / "staged.h5",
        diagnostics_json=tmp_path / "diagnostics.json",
        build_record_json=tmp_path / "build_record.json",
        terminal_gate_json=tmp_path / "terminal_gates.json",
    )
    with pytest.raises(ValueError, match="sha mismatch"):
        run_uk_calibration(
            paths=paths,
            input_sha256="0" * 64,
            ledger_artifact=object(),
            register_registry=_registry(),
            band_edge_registry=_registry(),
            calibration_year=2025,
            exclusion_receipt={},
            doctrine=UKNationalSolveDoctrine(epochs=1),
            doctrine_overrides={},
            measure_resolver=None,
            source_pins={
                "input_h5": {
                    "sha256": _sha(input_h5),
                    "size_bytes": input_h5.stat().st_size,
                }
            },
            run_config_extra={"calibration_year": 2025},
            release_id="bad-sha",
        )
    assert not paths.staging_h5.exists()
    assert not paths.diagnostics_json.exists()


def test_run_uk_calibration_refuses_absent_input_sidecar(tmp_path: Path):
    pytest.importorskip("tables")  # pandas HDF backend
    input_h5 = tmp_path / "input.h5"
    write_uk_national_frame(_frame(), input_h5)
    paths = UKCalibrationRunPaths(
        input_h5=input_h5,
        staging_h5=tmp_path / "staged.h5",
        diagnostics_json=tmp_path / "diagnostics.json",
        build_record_json=tmp_path / "build_record.json",
        terminal_gate_json=tmp_path / "terminal_gates.json",
    )

    with pytest.raises(ValueError, match="build sidecar absent"):
        run_uk_calibration(
            paths=paths,
            input_sha256=_sha(input_h5),
            ledger_artifact=object(),
            register_registry=_registry(),
            band_edge_registry=_registry(),
            calibration_year=2025,
            exclusion_receipt={},
            doctrine=UKNationalSolveDoctrine(epochs=1),
            doctrine_overrides={},
            measure_resolver=None,
            source_pins={
                "input_h5": {
                    "sha256": _sha(input_h5),
                    "size_bytes": input_h5.stat().st_size,
                }
            },
            run_config_extra={"calibration_year": 2025},
            release_id="missing-sidecar",
        )

    assert not paths.staging_h5.exists()
    assert not paths.diagnostics_json.exists()
    assert not paths.terminal_gate_json.exists()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            {"entity_row_counts": {"person": 999, "benunit": 4, "household": 4}},
            "row-count mismatch",
        ),
        ({"household_weight_total": 1.0}, "household_weight_total mismatch"),
    ],
)
def test_run_uk_calibration_refuses_unbound_input_sidecar(
    override, message, tmp_path: Path
):
    pytest.importorskip("tables")  # pandas HDF backend
    frame = _frame()
    input_h5 = tmp_path / "input.h5"
    write_uk_national_frame(frame, input_h5)
    _write_spine_sidecar(input_h5, frame, **override)
    paths = UKCalibrationRunPaths(
        input_h5=input_h5,
        staging_h5=tmp_path / "staged.h5",
        diagnostics_json=tmp_path / "diagnostics.json",
        build_record_json=tmp_path / "build_record.json",
        terminal_gate_json=tmp_path / "terminal_gates.json",
    )

    with pytest.raises(ValueError, match=message):
        run_uk_calibration(
            paths=paths,
            input_sha256=_sha(input_h5),
            ledger_artifact=object(),
            register_registry=_registry(),
            band_edge_registry=_registry(),
            calibration_year=2025,
            exclusion_receipt={},
            doctrine=UKNationalSolveDoctrine(epochs=1),
            doctrine_overrides={},
            measure_resolver=None,
            source_pins={
                "input_h5": {
                    "sha256": _sha(input_h5),
                    "size_bytes": input_h5.stat().st_size,
                }
            },
            run_config_extra={"calibration_year": 2025},
            release_id="unbound-sidecar",
        )

    assert not paths.staging_h5.exists()
    assert not paths.diagnostics_json.exists()
    assert not paths.terminal_gate_json.exists()


def test_seam_never_modifies_data_variables(monkeypatch, tmp_path: Path):
    """The seam's defining invariant: weights move, data never does.

    Every data column of every entity table in the staged H5 must be
    byte-identical to the input; only the household weights, the weight
    kind, and exactly one appended mass record may differ.
    """

    pytest.importorskip("tables")  # pandas HDF backend

    monkeypatch.setattr(
        calibration_run,
        "uk_aggregate_admin_totals",
        lambda frame, manifest: (_admin_anchor_values(), []),
    )
    input_h5 = tmp_path / "input.h5"
    frame = _frame()
    write_uk_national_frame(frame, input_h5)
    _write_spine_sidecar(input_h5, frame)
    paths = UKCalibrationRunPaths(
        input_h5=input_h5,
        staging_h5=tmp_path / "staged.h5",
        diagnostics_json=tmp_path / "diagnostics.json",
        build_record_json=tmp_path / "build_record.json",
        terminal_gate_json=tmp_path / "terminal_gates.json",
    )

    # Target 30 against an initial weighted UC count of 20, so the solve
    # genuinely has to move weights while the data stays untouched.
    pulling_registry = TargetRegistry(
        [
            TargetSpec(
                name="dwp.uc.households",
                entity="benunit",
                measure="dwp/uc/households",
                value=30.0,
                source="test",
                family="dwp_universal_credit",
                metadata={"contract_target_id": "dwp.uc.households"},
            )
        ],
        country="uk",
    )
    run_uk_calibration(
        paths=paths,
        input_sha256=_sha(input_h5),
        ledger_artifact=object(),
        register_registry=pulling_registry,
        band_edge_registry=pulling_registry,
        calibration_year=2025,
        exclusion_receipt={},
        doctrine=UKNationalSolveDoctrine(epochs=50),
        doctrine_overrides={},
        measure_resolver=None,
        source_pins={
            "input_h5": {
                "sha256": _sha(input_h5),
                "size_bytes": input_h5.stat().st_size,
            }
        },
        run_config_extra={},
        release_id="invariant-run",
    )

    source, _ = load_uk_national_frame(input_h5)
    staged, _ = load_uk_national_frame(paths.staging_h5)
    for entity in ("person", "benunit", "household"):
        left = source.table(entity)
        right = staged.table(entity)
        assert list(left.columns) == list(right.columns), entity
        for column in left.columns:
            pd.testing.assert_series_equal(left[column], right[column])
    assert staged.weights_for("household").kind is WeightKind.CALIBRATED
    assert not np.allclose(
        staged.weights_for("household").values,
        source.weights_for("household").values,
    )
    assert len(staged.mass_log) == len(source.mass_log) + 1


def test_aggregate_admin_measurement_convention_and_refusals():
    frame = _frame()
    manifest = calibration_run._calibration_gate_manifest()

    totals, receipt = calibration_run.uk_aggregate_admin_totals(frame, manifest)

    # Small anchors (NEED means) measure as the weighted mean over carriers;
    # the NHS total measures as the person total under mapped household
    # weights: 4 persons x 50.0 x weight 10.0.
    assert totals["need_electricity_mean_spending"] == pytest.approx(1.0)
    assert totals["need_gas_mean_spending"] == pytest.approx(1.0)
    assert totals["nhs_spending_total"] == pytest.approx(2000.0)
    by_anchor = {row["anchor"]: row for row in receipt}
    assert by_anchor["nhs_spending_total"]["entity"] == "person"
    assert (
        by_anchor["need_electricity_mean_spending"]["statistic_convention"]
        == "assessed_by_anchor_magnitude"
    )

    stripped = _frame()
    stripped.table("household").drop(columns=["electricity_consumption"], inplace=True)
    with pytest.raises(ValueError, match="household.electricity_consumption"):
        calibration_run.uk_aggregate_admin_totals(stripped, manifest)


def test_nhs_anchor_composes_from_the_columns_the_spine_actually_carries():
    """The anchor is published as one total; the spine carries it in three parts.

    Composing is the translation from the published concept to ours, and the
    receipt has to say so — the anchor measured a silent zero for as long as it
    named a column no stage produces.
    """

    frame = _frame()
    person = frame.table("person")
    person.drop(columns=["nhs_spending"], inplace=True)
    person["nhs_a_and_e_spending"] = [20.0, 20.0, 20.0, 20.0]
    person["nhs_admitted_patient_spending"] = [25.0, 25.0, 25.0, 25.0]
    person["nhs_outpatient_spending"] = [5.0, 5.0, 5.0, 5.0]
    manifest = calibration_run._calibration_gate_manifest()

    totals, receipt = calibration_run.uk_aggregate_admin_totals(frame, manifest)

    # Same 4 persons x 50.0 x weight 10.0 as the single-column fixture.
    assert totals["nhs_spending_total"] == pytest.approx(2000.0)
    by_anchor = {row["anchor"]: row for row in receipt}
    assert by_anchor["nhs_spending_total"]["composed_from"] == list(
        UK_NHS_SPENDING_COMPONENT_COLUMNS
    )
    assert by_anchor["need_gas_mean_spending"]["composed_from"] == []


def test_partly_carried_derived_anchor_refuses_and_names_the_missing_part():
    frame = _frame()
    person = frame.table("person")
    person.drop(columns=["nhs_spending"], inplace=True)
    person["nhs_a_and_e_spending"] = [20.0, 20.0, 20.0, 20.0]
    manifest = calibration_run._calibration_gate_manifest()

    with pytest.raises(ValueError, match="nhs_admitted_patient_spending"):
        calibration_run.uk_aggregate_admin_totals(frame, manifest)


def test_seam_pipeline_derives_a_ratified_logbook_scope():
    """The seam appends to the FRS line's chain, not a new unratified one."""

    logbook_tool = _load_logbook_tool()

    scope = logbook_tool._chain_scope(calibration_run._PIPELINE)

    assert scope == "uk/frs"
    assert scope in logbook_tool.DECLARED_SCOPES


def _load_logbook_tool():
    import importlib.util

    path = Path(__file__).resolve().parents[3] / "tools" / "logbook.py"
    spec = importlib.util.spec_from_file_location("_logbook_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_refusal_records_a_failed_attempt_and_stages_nothing(tmp_path: Path):
    pytest.importorskip("tables")  # pandas HDF backend
    input_h5 = tmp_path / "input.h5"
    write_uk_national_frame(_frame(), input_h5)
    paths = UKCalibrationRunPaths(
        input_h5=input_h5,
        staging_h5=tmp_path / "staged.h5",
        diagnostics_json=tmp_path / "diagnostics.json",
        build_record_json=tmp_path / "build_record.json",
        terminal_gate_json=tmp_path / "terminal_gates.json",
    )

    with pytest.raises(ValueError, match="sha mismatch"):
        run_uk_calibration(
            paths=paths,
            input_sha256="0" * 64,
            ledger_artifact=object(),
            register_registry=_registry(),
            band_edge_registry=_registry(),
            calibration_year=2025,
            exclusion_receipt={},
            doctrine=UKNationalSolveDoctrine(epochs=1),
            doctrine_overrides={},
            measure_resolver=None,
            source_pins={
                "input_h5": {
                    "sha256": _sha(input_h5),
                    "size_bytes": input_h5.stat().st_size,
                }
            },
            run_config_extra={"calibration_year": 2025},
            release_id="refused-run",
        )

    # Every terminal disposition is a row; a refusal that left the chain
    # silent would hide the attempt entirely.
    spooled = sorted((tmp_path / "logbook-spool").rglob("*.json"))
    assert spooled, "refusal recorded no Logbook row"
    rows = [json.loads(path.read_text()) for path in spooled]
    assert [row["disposition"] for row in rows] == ["failed"]
    assert rows[0]["pipeline"] == calibration_run._PIPELINE
    # The refusal is explained on disk, not only in the row.
    receipts = sorted((tmp_path / "logbook-receipts").rglob("error.json"))
    assert len(receipts) == 1
    assert not paths.staging_h5.exists()
    assert not paths.diagnostics_json.exists()
    assert not paths.terminal_gate_json.exists()


def test_attempt_ids_are_unique_across_reruns_of_one_release(
    monkeypatch, tmp_path: Path
):
    pytest.importorskip("tables")  # pandas HDF backend
    monkeypatch.setattr(
        calibration_run,
        "uk_aggregate_admin_totals",
        lambda frame, manifest: (_admin_anchor_values(), []),
    )
    input_h5 = tmp_path / "input.h5"
    frame = _frame()
    write_uk_national_frame(frame, input_h5)
    _write_spine_sidecar(input_h5, frame)
    source_pins = {
        "input_h5": {"sha256": _sha(input_h5), "size_bytes": input_h5.stat().st_size}
    }
    build_ids = []
    for attempt in ("a", "b"):
        run_dir = tmp_path / attempt
        run_dir.mkdir()
        result = run_uk_calibration(
            paths=UKCalibrationRunPaths(
                input_h5=input_h5,
                staging_h5=run_dir / "staged.h5",
                diagnostics_json=run_dir / "diagnostics.json",
                build_record_json=run_dir / "build_record.json",
                terminal_gate_json=run_dir / "terminal_gates.json",
            ),
            input_sha256=_sha(input_h5),
            ledger_artifact=object(),
            register_registry=_registry(),
            band_edge_registry=_registry(),
            calibration_year=2025,
            exclusion_receipt={},
            doctrine=UKNationalSolveDoctrine(epochs=5),
            doctrine_overrides={},
            measure_resolver=None,
            source_pins=source_pins,
            run_config_extra={"calibration_year": 2025},
            release_id="one-release-id",
        )
        build_ids.append(result.build_record["build_id"])

    # Both the local chain and the store reject a duplicate build id, so one
    # release re-run twice must not collide.
    assert build_ids[0] != build_ids[1]
    assert all(value.startswith("uk-frs-calibration-attempt-") for value in build_ids)


def test_verified_ledger_identity_reaches_the_run_evidence(monkeypatch, tmp_path: Path):
    pytest.importorskip("tables")  # pandas HDF backend
    monkeypatch.setattr(
        calibration_run,
        "uk_aggregate_admin_totals",
        lambda frame, manifest: (_admin_anchor_values(), []),
    )
    input_h5 = tmp_path / "input.h5"
    frame = _frame()
    write_uk_national_frame(frame, input_h5)
    _write_spine_sidecar(input_h5, frame)
    artifact = SimpleNamespace(
        facts_sha256="d" * 64,
        fact_row_count=107_550,
        manifest_sha256="e" * 64,
        manifest={
            "artifact_id": "chronicle-uk-artifact-1cab809",
            "profile": "uk-national",
            "schema_version": 1,
            "unrelated": "not carried",
        },
    )

    result = run_uk_calibration(
        paths=UKCalibrationRunPaths(
            input_h5=input_h5,
            staging_h5=tmp_path / "staged.h5",
            diagnostics_json=tmp_path / "diagnostics.json",
            build_record_json=tmp_path / "build_record.json",
            terminal_gate_json=tmp_path / "terminal_gates.json",
        ),
        input_sha256=_sha(input_h5),
        ledger_artifact=artifact,
        register_registry=_registry(),
        band_edge_registry=_registry(),
        calibration_year=2025,
        exclusion_receipt={},
        doctrine=UKNationalSolveDoctrine(epochs=5),
        doctrine_overrides={},
        measure_resolver=None,
        source_pins={
            "input_h5": {
                "sha256": _sha(input_h5),
                "size_bytes": input_h5.stat().st_size,
            }
        },
        run_config_extra={"calibration_year": 2025},
        release_id="ledger-identity",
    )

    ledger = result.build_record["run_config"]["ledger"]
    assert ledger["facts_sha256"] == "d" * 64
    assert ledger["manifest_sha256"] == "e" * 64
    assert ledger["fact_row_count"] == 107_550
    assert ledger["manifest"] == {
        "artifact_id": "chronicle-uk-artifact-1cab809",
        "profile": "uk-national",
        "schema_version": 1,
    }
    # A bare feed carries no manifest, and that absence is recorded rather
    # than invented.
    assert calibration_run._ledger_provenance(object())["manifest_sha256"] is None


def _consumer_fact_row(
    *,
    aggregate_fact_key: str,
    semantic_fact_key: str,
    schema_version: str,
    source_release_key: str | None = None,
) -> dict:
    row = {
        "aggregate_fact_key": aggregate_fact_key,
        "semantic_fact_key": semantic_fact_key,
        "schema_version": schema_version,
        "value": 1.0,
        "period": {"type": "tax_year", "value": 2025},
        "geography": {"level": "country", "id": "K02000001"},
        "entity": {"name": "household"},
        "aggregation": {"method": "sum"},
        "observed_measure": {"source_name": "ons", "unit": "gbp"},
        "source": {"source_name": "ons"},
        "lineage": {"source_record_id": "ons.2025.total"},
    }
    if source_release_key is not None:
        row["source_release_key"] = source_release_key
    return row


def _mixed_epoch_artifact_dir(tmp_path: Path) -> Path:
    """A cutover-window feed: ledger-era history, chronicle-era rows, and one
    Chronicle-namespace key spelling nothing declares.

    The manifest declares what Chronicle's ``main`` stamps today
    (``policyengine_ledger.consumer_artifact.v2``), so this is the shape a UK
    run is handed now, not a hypothetical one.
    """
    rows = [
        _consumer_fact_row(
            aggregate_fact_key="ledger.aggregate_fact.v2:aaa",
            semantic_fact_key="ledger.semantic_fact.v2:aaa",
            schema_version=LEDGER_CONSUMER_FACT_SCHEMA_VERSION,
        ),
        _consumer_fact_row(
            aggregate_fact_key="chronicle.aggregate_fact.v3:bbb",
            semantic_fact_key="chronicle.semantic_fact.v3:bbb",
            schema_version=CHRONICLE_CONSUMER_FACT_SCHEMA_VERSION,
        ),
        _consumer_fact_row(
            aggregate_fact_key="ledger.aggregate_fact.v2:ccc",
            semantic_fact_key="ledger.semantic_fact.v2:ccc",
            schema_version=LEDGER_CONSUMER_FACT_SCHEMA_VERSION,
            source_release_key="chronicle.source_release.v9:undeclared",
        ),
    ]
    artifact_dir = tmp_path / "chronicle-artifact"
    artifact_dir.mkdir()
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    (artifact_dir / "consumer_facts.jsonl").write_text(payload)
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": PUBLISHED_CONSUMER_ARTIFACT_SCHEMA_VERSION,
                "artifact_id": "chronicle-uk-artifact-mixed",
                "profile": "uk-national",
                "fact_row_count": len(rows),
                "facts_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return artifact_dir


def test_mixed_epoch_feed_epochs_reach_the_uk_release_evidence(monkeypatch, tmp_path):
    """A UK run says which Chronicle era resolved its targets.

    The run's own provenance block used to be assembled field by field from
    the artifact, which is how it came to carry the hashes but not the epoch
    witnesses the loader had already computed. It now delegates to the shared
    block, so a cutover-window feed is visible in the signed diagnostics and
    in the build record — including the one Chronicle-namespace spelling this
    build does not declare, named rather than folded into an era.
    """
    pytest.importorskip("tables")  # pandas HDF backend
    monkeypatch.setattr(
        calibration_run,
        "uk_aggregate_admin_totals",
        lambda frame, manifest: (_admin_anchor_values(), []),
    )
    input_h5 = tmp_path / "input.h5"
    frame = _frame()
    write_uk_national_frame(frame, input_h5)
    _write_spine_sidecar(input_h5, frame)
    artifact = load_ledger_consumer_artifact(_mixed_epoch_artifact_dir(tmp_path))
    diagnostics_json = tmp_path / "diagnostics.json"

    result = run_uk_calibration(
        paths=UKCalibrationRunPaths(
            input_h5=input_h5,
            staging_h5=tmp_path / "staged.h5",
            diagnostics_json=diagnostics_json,
            build_record_json=tmp_path / "build_record.json",
            terminal_gate_json=tmp_path / "terminal_gates.json",
        ),
        input_sha256=_sha(input_h5),
        ledger_artifact=artifact,
        register_registry=_registry(),
        band_edge_registry=_registry(),
        calibration_year=2025,
        exclusion_receipt={},
        doctrine=UKNationalSolveDoctrine(epochs=5),
        doctrine_overrides={},
        measure_resolver=None,
        source_pins={
            "input_h5": {
                "sha256": _sha(input_h5),
                "size_bytes": input_h5.stat().st_size,
            }
        },
        run_config_extra={"calibration_year": 2025},
        release_id="chronicle-mixed-epoch",
    )

    ledger = result.build_record["run_config"]["ledger"]
    # The observed manifest id, verbatim, and the era it belongs to.
    assert ledger["manifest"]["schema_version"] == (
        PUBLISHED_CONSUMER_ARTIFACT_SCHEMA_VERSION
    )
    assert ledger["schema_epoch"] == "ledger"
    # The feed straddles the cutover, and says so rather than reporting one era.
    assert ledger["fact_key_epochs"] == ["ledger", "chronicle", "undeclared"]
    assert ledger["undeclared_fact_key_domains"] == ["chronicle.source_release.v9"]
    assert ledger["fact_schema_versions"] == [
        CHRONICLE_CONSUMER_FACT_SCHEMA_VERSION,
        LEDGER_CONSUMER_FACT_SCHEMA_VERSION,
    ]
    # The hashes the block always carried are unchanged by the delegation.
    assert ledger["facts_sha256"] == artifact.facts_sha256
    assert ledger["fact_row_count"] == 3

    # The same block is what the signed diagnostics carry, so the evidence a
    # release assembler reads witnesses the era too.
    diagnostics = json.loads(diagnostics_json.read_text())
    assert diagnostics["build"]["ledger"] == ledger


def test_the_uk_block_delegates_rather_than_reassembling_the_shared_one(tmp_path):
    """Every field of the shared provenance block reaches the UK block.

    Pinned as a property, not as a list: the failure being prevented is a
    field the loader learns and this seam silently drops, and a test that
    enumerated today's fields would not catch tomorrow's.
    """
    artifact = load_ledger_consumer_artifact(_mixed_epoch_artifact_dir(tmp_path))
    shared = artifact.provenance()

    block = calibration_run._ledger_provenance(artifact)

    for field, value in shared.items():
        if field in {"path_name", "schema_version", "profiles"}:
            # Not identity of the feed: the directory name is local, and the
            # manifest id is carried in the narrower manifest sub-block.
            continue
        assert block[field] == value, field
    assert block["manifest"]["schema_version"] == shared["schema_version"]
