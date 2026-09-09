"""Synthetic end-to-end contract for the first UK rowwise candidate."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.logbook import LOGBOOK_ROW_FIELDS, load_spool_rows
from microcosm.build.uk_runtime import (
    assemble_uk_oa_ladder,
    ladder_target_provenance,
    load_uk_oa_ladder,
    read_uk_single_year_weight_metadata,
    write_uk_national_frame,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    validate_uk_national_frame,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import MassChangeRecord, WeightKind


@pytest.fixture(autouse=True)
def _empty_support_exclusions_for_synthetic_rosters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic candidates never carry the committed micro-LA exclusions.

    The committed ``local_area_support_exclusions.json`` names real local
    authorities measured on the licensed spine; a synthetic roster either
    lacks them (unknown) or meets the floor (stale), and the gate rightly
    fails either way. These tests exercise the machinery, so the support
    register is pinned empty here; the committed entries are covered by
    ``test_uk_battery_bindings.py``.
    """

    import microcosm.build.uk_runtime.battery_bindings as battery_bindings

    real_loader = battery_bindings.load_uk_reviewed_exclusion_register

    def _loader(path, *, resource, **kwargs):
        if resource == "local_area_support_exclusions.json":
            return {}
        return real_loader(path, resource=resource, **kwargs)

    monkeypatch.setattr(
        battery_bindings, "load_uk_reviewed_exclusion_register", _loader
    )


@pytest.fixture(autouse=True)
def _spool_only_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POPULACE_LEDGER_URL", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_API_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LOGBOOK_PREV_ROW_DIGEST", raising=False)
    monkeypatch.setenv(
        "MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY",
        base64.b64encode(b"\x07" * 32).decode("ascii"),
    )


def _spool_rows(output_dir: Path):
    rows = load_spool_rows(output_dir / "logbook-spool")
    for row in rows:
        assert frozenset(row.to_mapping()) == LOGBOOK_ROW_FIELDS
    return rows


def _local_ref(path: Path) -> str:
    return f"local://{path.resolve().as_posix().lstrip('/')}"


def _load_builder_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_uk_rowwise_candidate.py"
    spec = importlib.util.spec_from_file_location(
        "build_uk_rowwise_candidate",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _ladder_metadata() -> dict[str, object]:
    def layer(vintage: str) -> dict[str, object]:
        return {"vintage": vintage, "source": "synthetic test source"}

    return {
        "schema_version": 1,
        "kind": "uk_oa_ladder",
        "coverage": "uk",
        "oa_vintage": "synthetic",
        "constituency_sampling_basis": "synthetic household counts",
        "oa_sampling_basis": "synthetic population",
        "layers": {
            "constituency": layer("2024_pcon"),
            "lsoa": layer("synthetic"),
            "msoa": layer("synthetic"),
            "local_authority": layer("synthetic"),
            "ward": layer("synthetic"),
            "itl": layer("2021_itl"),
            "region": layer("synthetic"),
        },
    }


def _ladder_frame(
    household_counts: tuple[float, float, float, float] = (
        3.0,
        10.0,
        10.0,
        10.0,
    ),
) -> pd.DataFrame:
    rows = [
        (
            "E00000001",
            "E12000007",
            "E14000001",
            "E05014284",
            "E09000001",
            "TLI31",
        ),
        (
            "W00000001",
            "W99999999",
            "W07000041",
            "W05001517",
            "W06000001",
            "TLL11",
        ),
        (
            "S00000001",
            "S99999999",
            "S14000001",
            "S13002835",
            "S12000033",
            "TLM50",
        ),
        (
            "N20000001",
            "N99999999",
            "N05000001",
            "N10000104",
            "N09000001",
            "TLN0A",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "oa_code": oa,
                "population": 100.0,
                "households": households,
                "constituency_code": constituency,
                "region_code": region,
                "lsoa_code": oa,
                "msoa_code": oa,
                "local_authority_code": local_authority,
                "ward_code": ward,
                "itl3_code": itl3,
            }
            for (
                oa,
                region,
                constituency,
                ward,
                local_authority,
                itl3,
            ), households in zip(rows, household_counts, strict=True)
        ]
    )


def _write_ladder(
    path: Path,
    *,
    household_counts: tuple[float, float, float, float] = (
        3.0,
        10.0,
        10.0,
        10.0,
    ),
):
    payload = assemble_uk_oa_ladder(
        _ladder_frame(household_counts),
        _ladder_metadata(),
    )
    np.savez_compressed(path, **payload)
    return load_uk_oa_ladder(path)


def _write_staging_h5(
    path: Path,
    *,
    households_per_region: int = 3,
    region_masses: tuple[float, float, float, float] = (3.0, 10.0, 10.0, 10.0),
) -> None:
    if households_per_region < 3:
        raise ValueError("spine fixture needs one raw row and two derivatives")
    region_names = (
        "LONDON",
        "WALES",
        "SCOTLAND",
        "NORTHERN_IRELAND",
    )
    household_ids = list(range(1, 4 * households_per_region + 1))
    source_household_ids: list[int] = []
    support_clone_indices: list[int] = []
    spi_flags: list[bool] = []
    for region_index in range(4):
        first = region_index * households_per_region + 1
        raw_count = households_per_region - 2
        source_household_ids.extend(range(first, first + raw_count))
        source_household_ids.extend([first, first])
        support_clone_indices.extend([0] * raw_count + [0, 1])
        spi_flags.extend([False] * raw_count + [True, False])
    household = pd.DataFrame(
        {
            "household_id": household_ids,
            "household_weight": [
                mass / households_per_region
                for mass in region_masses
                for _ in range(households_per_region)
            ],
            "region": [
                region for region in region_names for _ in range(households_per_region)
            ],
            "source_household_id": source_household_ids,
            "source_household_key": [
                f"2023:{source_id}" for source_id in source_household_ids
            ],
            "household_source_id": source_household_ids,
            "household_support_clone_index": support_clone_indices,
            "household_is_spi_synthetic": spi_flags,
            "household_is_capital_gains_clone": [False] * len(household_ids),
            "household_is_cgt_band_donor": [False] * len(household_ids),
        }
    )
    person_ids = [10_000 + household_id for household_id in household_ids]
    benunit_ids = [20_000 + household_id for household_id in household_ids]
    person = pd.DataFrame(
        {
            "person_id": person_ids,
            "person_household_id": household_ids,
            "person_benunit_id": benunit_ids,
        }
    )
    benunit = pd.DataFrame({"benunit_id": benunit_ids})
    dataset = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
        mass_log=(
            MassChangeRecord(
                entity="household",
                old_total=33.0,
                new_total=33.0,
                declared_factor=1.0,
                reason="Synthetic staging mass record.",
            ),
        ),
    )
    write_uk_national_frame(dataset, path)


def test_candidate_build_writes_calibrated_h5_and_evidence(
    monkeypatch, tmp_path
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    output_dir = tmp_path / "candidate"
    _write_staging_h5(input_h5, households_per_region=52)
    ladder = _write_ladder(ladder_path)
    import microcosm.build.uk_runtime.battery_bindings as battery_bindings

    monkeypatch.setattr(
        battery_bindings,
        "_local_area_roster",
        lambda _resource, levels: {
            "constituency": tuple(sorted(set(ladder.constituency_code))),
            "local_authority": tuple(sorted(set(ladder.local_authority_code))),
        },
    )
    holdout = {
        "report_only": True,
        "method": "rotated_folds",
        "n_folds": 5,
        "seed": 20260529,
        "solve_seed": 7,
        "mean_holdout_loss": 0.1,
        "worst_holdout_loss": 0.2,
        "fold_losses": [0.1, 0.1, 0.2, 0.05, 0.05],
        "folds": [],
    }
    monkeypatch.setattr(
        builder,
        "rotated_uk_local_holdout",
        lambda *_args, **_kwargs: holdout,
    )

    assert (
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--ladder",
                str(ladder_path),
                "--out",
                str(output_dir),
                "--n-clones",
                "2",
                "--seed",
                "7",
                "--epochs",
                "2",
            ]
        )
        == 0
    )

    candidate_h5 = output_dir / builder.CANDIDATE_FILENAME_TEMPLATE.format(
        calibration_year=2025
    )
    expected_sidecars = {
        builder.MANIFEST_FILENAME,
        builder.SOLVE_DIAGNOSTICS_FILENAME,
        builder.AREA_SUPPORT_FILENAME,
        builder.PAST_CAP_FILENAME,
        builder.CALIBRATION_DIAGNOSTICS_FILENAME,
        builder.LOCAL_REGISTRY_FILENAME,
        builder.LOCAL_GATE_REPORT_FILENAME_TEMPLATE.format(calibration_year=2025),
    }
    assert candidate_h5.exists()
    assert expected_sidecars <= {path.name for path in output_dir.iterdir()}

    candidate_kind, candidate_mass_log = read_uk_single_year_weight_metadata(
        candidate_h5
    )
    with pd.HDFStore(candidate_h5, mode="r") as store:
        candidate_household = store["household"]
    assert candidate_kind is WeightKind.CALIBRATED
    assert candidate_household["source_year"].unique().tolist() == [2023]
    assert len(set(candidate_household["source_household_key"])) == 200
    assert {"2023:1", "2023:206"} <= set(candidate_household["source_household_key"])
    assert len(candidate_mass_log) == 3
    calibration_records = [
        record
        for record in candidate_mass_log
        if "census_households/constituency" in record.reason
    ]
    assert calibration_records == [candidate_mass_log[-1]]
    # The kernel-minted record declares the realized factor (the hand-minted
    # predecessor left it None) — declared-vs-realized is validated by the
    # kernel at with_weights time.
    record = candidate_mass_log[-1]
    assert record.declared_factor == pytest.approx(record.new_total / record.old_total)

    manifest = json.loads((output_dir / builder.MANIFEST_FILENAME).read_text())
    assert manifest["candidate_scope"] == "adjudicated_partial"
    assert manifest["bound_target_families"] == ["census_households/constituency"]
    adjudications = manifest["binding_adjudications"]
    assert adjudications["register_resource"] == "local_binding_adjudications.json"
    assert adjudications["bound_families"] == ["census_households/constituency"]
    assert adjudications["evaluated_on"]
    seed = adjudications["stood_on"]["census_households/constituency"][
        "census_disclosure_control_noise"
    ]
    assert seed["approved_by"] == "juaristi22"
    assert seed["adjudication"] == "microcosm#802"
    assert seed["approved_on"] == "2026-08-31"
    assert seed["expires_on"] == "2026-11-30"
    assert adjudications["dormant"] == [
        "full_frs_tei_band_unavailable",
        "hmrc_spi_frame_model_proxy",
        "population_universe_private_households",
        "uc_unit_vs_household_grain",
        "voa_dwellings_vs_household_frame",
    ]
    cross_grain = manifest["cross_grain"]
    assert cross_grain["bound_national_targets"] == []
    assert cross_grain["bound_higher_targets"] == []
    assert cross_grain["inconsistencies_in_force"] == []
    assert cross_grain["groups"] == []
    assert cross_grain["empty_legs_licensed"] == []
    assert cross_grain["controls_without_lower_rows"] == []
    assert cross_grain["absence"]
    assert manifest["ladder_target_provenance"] == ladder_target_provenance(ladder)
    assert manifest["gate"]["passed"] is True
    assert manifest["gate"]["phase"] == "post_calibration"
    assert manifest["gate"]["details"]
    assert (
        manifest["inputs"]["dataset"]["sha256"]
        == hashlib.sha256(input_h5.read_bytes()).hexdigest()
    )
    assert manifest["inputs"]["dataset"]["bytes"] == input_h5.stat().st_size
    assert (
        manifest["inputs"]["ladder"]["sha256"]
        == hashlib.sha256(ladder_path.read_bytes()).hexdigest()
    )
    assert manifest["inputs"]["ladder"]["bytes"] == ladder_path.stat().st_size
    assert manifest["parameters"]["n_clones"] == 2
    assert manifest["parameters"]["seed"] == 7
    assert manifest["parameters"]["source_year"] == 2023
    assert manifest["parameters"]["source_lineage_modulus"] is None
    assert manifest["parameters"]["epochs"] == 2
    assert manifest["parameters"]["learning_rate"] == pytest.approx(0.15)
    assert manifest["parameters"]["expected_constituency_vintage"] == "2024_pcon"
    assert [
        row["kind"] for row in manifest["weights"]["household_weight_kind_chain"]
    ] == ["importance", "importance", "calibrated"]
    assert manifest["weights"]["mass_log_records_before_calibration"] == 2
    assert manifest["weights"]["mass_log_records"] == 3
    mass_change = manifest["weights"]["calibration_mass_change"]
    assert mass_change["old_total"] == pytest.approx(33.0)
    assert mass_change["new_total"] == pytest.approx(
        candidate_household["household_weight"].sum()
    )
    assert mass_change["relative_shift"] == pytest.approx(
        (mass_change["new_total"] - 33.0) / 33.0
    )
    assert manifest["parameters"]["doctrine"] == {
        "target_loss_cap": 10.0,
        "max_weight_ratio": 10.0,
        "scale_rule": "default_target_loss_scales",
        "target_weight_rule": "grain_equal",
        "solve_epochs": 1500,
        "clone_count": 15,
    }
    assert manifest["ladder_household_uprating"]["applied"] is False
    assert manifest["solve"]["n_targets"] == 4
    assert manifest["solve"]["n_households"] == 416
    assert np.isfinite(manifest["solve"]["initial_loss"])
    assert np.isfinite(manifest["solve"]["final_loss"])
    assert np.isfinite(manifest["solve"]["max_abs_relative_error"])
    assert np.isfinite(manifest["solve"]["median_abs_relative_error"])
    assert manifest["solve"]["past_cap"]["n_targets"] == 4
    assert manifest["support"]["min_assigned_households"] == 104
    assert manifest["support"]["min_nonzero_households"] == 104
    assert manifest["support"]["min_effective_sample_size"] == pytest.approx(104.0)

    diagnostics = pd.read_csv(output_dir / builder.SOLVE_DIAGNOSTICS_FILENAME)
    support = pd.read_csv(output_dir / builder.AREA_SUPPORT_FILENAME)
    past_cap = json.loads((output_dir / builder.PAST_CAP_FILENAME).read_text())
    calibration_diagnostics = json.loads(
        (output_dir / builder.CALIBRATION_DIAGNOSTICS_FILENAME).read_text()
    )
    assert len(diagnostics) == 4
    assert diagnostics["metric"].unique().tolist() == ["households"]
    assert len(support) == 8
    assert past_cap["n_targets"] == 4
    assert calibration_diagnostics["schema_version"] == 6
    uk_diagnostics = calibration_diagnostics["uk_diagnostics"]
    assert len(uk_diagnostics["weakest_families"]) == 1
    assert len(uk_diagnostics["weakest_areas_by_fit"]["bottom_by_fit"]) == 4
    assert uk_diagnostics["weakest_areas_by_fit"]["n_areas_scored"] == 4
    assert {
        row["country"] for row in uk_diagnostics["weakest_areas_by_fit"]["countries"]
    } == {
        "England",
        "Northern Ireland",
        "Scotland",
        "Wales",
    }
    assert (
        manifest["diagnostics"]["weakest_families"]
        == uk_diagnostics["weakest_families"]
    )
    assert uk_diagnostics["rotated_holdout"] == holdout
    assert manifest["diagnostics"]["rotated_holdout"] == holdout
    assert "calibration_diagnostics" in manifest["outputs"]
    rows = _spool_rows(output_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row.pipeline == "uk-local-candidate"
    assert row.rung == "f100"
    assert row.seed == 7
    assert row.disposition == "iterating"
    assert row.artifact_location == _local_ref(candidate_h5)
    assert set(row.gate_verdicts) == set(builder.UK_LOCAL_GATE_SCOPE)
    assert {item["verdict"] for item in row.gate_verdicts.values()} == {"passed"}
    assert all(
        ".local_gates.json#/gates/" in item["receipt"]
        for item in row.gate_verdicts.values()
    )


def test_candidate_dry_run_plans_without_solve_or_write(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    output_dir = tmp_path / "dry-run-output"
    _write_staging_h5(input_h5)
    ladder = _write_ladder(ladder_path)

    def forbidden(*_args, **_kwargs):
        pytest.fail("dry run called a solve or dataset writer")

    monkeypatch.setattr(
        builder,
        "solve_uk_rowwise_weights_under_doctrine",
        forbidden,
    )
    monkeypatch.setattr(builder, "write_uk_rowwise_dataset", forbidden)

    assert (
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--ladder",
                str(ladder_path),
                "--out",
                str(output_dir),
                "--n-clones",
                "2",
                "--seed",
                "7",
                "--dry-run",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    plan = json.loads(captured.out)
    assert plan["dry_run"] is True
    assert plan["sampling"] == {
        "fraction": 1.0,
        "seed": 578,
        "rung_token": "f100",
        "sampled": False,
        "pre_household_count": 12,
        "post_household_count": 12,
    }
    assert plan["bound_target_families"] == ["census_households/constituency"]
    adjudications = plan["binding_adjudications"]
    assert adjudications["register_resource"] == "local_binding_adjudications.json"
    assert adjudications["bound_families"] == ["census_households/constituency"]
    assert adjudications["evaluated_on"]
    assert (
        "census_disclosure_control_noise"
        in adjudications["stood_on"]["census_households/constituency"]
    )
    assert adjudications["dormant"] == [
        "full_frs_tei_band_unavailable",
        "hmrc_spi_frame_model_proxy",
        "population_universe_private_households",
        "uc_unit_vs_household_grain",
        "voa_dwellings_vs_household_frame",
    ]
    cross_grain = plan["cross_grain"]
    assert cross_grain["bound_national_targets"] == []
    assert cross_grain["bound_higher_targets"] == []
    assert cross_grain["inconsistencies_in_force"] == []
    assert cross_grain["groups"] == []
    assert cross_grain["empty_legs_licensed"] == []
    assert cross_grain["controls_without_lower_rows"] == []
    assert cross_grain["absence"]
    assert plan["ladder_target_provenance"] == ladder_target_provenance(ladder)
    assert plan["shapes"]["person"][0] == 24
    assert plan["shapes"]["benunit"][0] == 24
    assert plan["shapes"]["household"][0] == 24
    assert plan["shapes"]["local_matrix"] == [4, 24]
    assert plan["target_count"] == 4
    assert not output_dir.exists()
    assert not (output_dir / "logbook-spool").exists()


def test_candidate_sampling_rung_receipt_and_engine_block_validation(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    output_dir = tmp_path / "dry-run-output"
    _write_staging_h5(input_h5)
    _write_ladder(ladder_path)

    def compact_sampler_forbidden(*_args, **_kwargs):
        pytest.fail("rowwise spine path called the certified-compact sampler")

    monkeypatch.setattr(
        builder,
        "sample_uk_national_frame",
        compact_sampler_forbidden,
        raising=False,
    )
    monkeypatch.setattr(
        builder,
        "sample_uk_spine_frame",
        lambda frame, **_kwargs: (
            frame,
            {
                "fraction": 0.01,
                "seed": 578,
                "rung_token": "f001",
                "pre_household_count": 12,
                "post_household_count": 12,
                "pre_family_count": 4,
                "post_family_count": 4,
                "normalization_factor": 1.0,
                "strata_count": 4,
                "receipt": {"synthetic_fixture": True},
            },
        ),
        raising=False,
    )

    assert (
        builder._parse_args(
            [
                "--input-h5",
                str(input_h5),
                "--ladder",
                str(ladder_path),
                "--out",
                str(output_dir),
            ]
        ).n_clones
        == builder.UK_LOCAL_CLONE_COUNT
    )
    with pytest.raises(ValueError, match="must equal --n-clones"):
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--ladder",
                str(ladder_path),
                "--out",
                str(output_dir),
                "--n-clones",
                "4",
                "--engine-blocks",
                "2",
                "--dry-run",
            ]
        )
    assert (
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--ladder",
                str(ladder_path),
                "--out",
                str(output_dir),
                "--n-clones",
                "2",
                "--sample-fraction",
                "0.01",
                "--sample-seed",
                "578",
                "--dry-run",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["sampling"]["fraction"] == 0.01
    assert plan["sampling"]["rung_token"] == "f001"
    assert plan["sampling"]["pre_household_count"] == 12
    assert plan["sampling"]["post_household_count"] >= 1
    assert plan["sampling"]["normalization_factor"] > 0


def test_candidate_f100_does_not_call_any_sampler(monkeypatch, tmp_path) -> None:
    pytest.importorskip("tables")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    _write_staging_h5(input_h5)
    frame, _ = builder.load_uk_national_frame(input_h5)

    def forbidden(*_args, **_kwargs):
        pytest.fail("f100 called a sampler")

    monkeypatch.setattr(builder, "sample_uk_national_frame", forbidden, raising=False)
    monkeypatch.setattr(builder, "sample_uk_spine_frame", forbidden, raising=False)

    sampled, receipt = builder._sample_candidate_frame(
        frame,
        fraction=1.0,
        seed=578,
    )

    assert sampled is frame
    assert receipt == {
        "fraction": 1.0,
        "seed": 578,
        "rung_token": "f100",
        "sampled": False,
        "pre_household_count": 12,
        "post_household_count": 12,
    }


def test_candidate_clone_count_planning_is_dry_run_only(tmp_path) -> None:
    builder = _load_builder_module()
    with pytest.raises(ValueError, match="only with --dry-run"):
        builder.main(
            [
                "--input-h5",
                str(tmp_path / "missing.h5"),
                "--ladder",
                str(tmp_path / "missing.npz"),
                "--out",
                str(tmp_path / "out"),
                "--candidate-clone-counts",
                "1,2,4",
            ]
        )


def test_candidate_engine_surface_reuses_one_resolver(
    monkeypatch,
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    _write_staging_h5(input_h5)
    frame, _ = builder.load_uk_national_frame(input_h5)
    constructions = []
    cgt_period_contract = {
        "version": "uk-cgt-measurement-v2",
        "input_period": "2024",
        "calibration_period": 2025,
        "bound_measurements": {
            "cgt_2024_gains": {
                "model_variable": "capital_gains",
                "measurement_period": 2024,
            }
        },
    }

    class StubResolver:
        def __init__(self, **kwargs):
            constructions.append(kwargs)
            self.simulation = object()
            self.contract_targets = {}

        def receipt(self):
            return {
                "mode": "stub",
                "policyengine_uk_version": "test",
                "cgt_period_contract": cgt_period_contract,
            }

    monkeypatch.setattr(
        builder,
        "compute_household_metrics",
        lambda _simulation, area_type, *, household_ids, **_kwargs: pd.DataFrame(
            {f"{area_type}_metric": np.ones(len(household_ids))},
            index=household_ids,
        ),
    )
    registry = TargetRegistry([], country="uk")
    prepared, restore, national, local_metrics, receipt = (
        builder._resolve_candidate_engine_surface(
            frame,
            registry,
            period=2025,
            scratch_dir=tmp_path / "scratch",
            resolver_factory=StubResolver,
        )
    )

    assert len(constructions) == 1
    assert receipt == {
        "mode": "stub",
        "engine_version": "test",
        "households": 12,
        "persons": 12,
        "benunits": 12,
        "national_inputs": 0,
        "local_metrics": {"constituency": 1, "la": 1},
        "blocks": 1,
        "cgt_period_contract": cgt_period_contract,
    }
    assert set(local_metrics) == {"constituency", "la"}
    assert len(national.targets) == 0
    assert restore(prepared).table("household").equals(frame.table("household"))


@pytest.mark.parametrize("second_cgt_period", [2024, 2025, None])
def test_candidate_engine_surface_resolves_real_per_clone_blocks(
    monkeypatch,
    tmp_path,
    second_cgt_period,
) -> None:
    pytest.importorskip("tables")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    _write_staging_h5(input_h5)
    frame, _ = builder.load_uk_national_frame(input_h5)
    ladder = _write_ladder(ladder_path)
    clone = builder._clone_with_ladder_binding(
        frame,
        ladder,
        n_clones=2,
        seed=7,
        source_year=2023,
        expected_constituency_vintage="2024_pcon",
        source_lineage_modulus=None,
    ).result
    constructions = []

    class StubResolver:
        def __init__(self, **kwargs):
            constructions.append(kwargs)
            self.simulation = object()
            self.contract_targets = {}
            self.cgt_period = 2024 if len(constructions) == 1 else second_cgt_period

        def receipt(self):
            receipt = {"mode": "stub", "policyengine_uk_version": "test"}
            if self.cgt_period is not None:
                receipt["cgt_period_contract"] = {
                    "version": "uk-cgt-measurement-v2",
                    "input_period": "2024",
                    "calibration_period": 2025,
                    "bound_measurements": {
                        "cgt_2024_gains": {
                            "model_variable": "capital_gains",
                            "measurement_period": self.cgt_period,
                        }
                    },
                }
            return receipt

    monkeypatch.setattr(
        builder,
        "compute_household_metrics",
        lambda _simulation, area_type, *, household_ids, **_kwargs: pd.DataFrame(
            {f"{area_type}_metric": np.arange(len(household_ids), dtype=float)},
            index=household_ids,
        ),
    )

    def resolve():
        return builder._resolve_candidate_engine_surface(
            clone.frame,
            TargetRegistry([], country="uk"),
            period=2025,
            scratch_dir=tmp_path / "block-scratch",
            resolver_factory=StubResolver,
            blocks=2,
        )

    if second_cgt_period != 2024:
        with pytest.raises(RuntimeError, match="CGT period contract is inconsistent"):
            resolve()
        return

    prepared, restore, _, metrics, receipt = resolve()

    assert len(constructions) == 2
    assert [len(call["frame"].table("household")) for call in constructions] == [
        12,
        12,
    ]
    assert receipt["blocks"] == 2
    assert receipt["cgt_period_contract"]["bound_measurements"] == {
        "cgt_2024_gains": {
            "model_variable": "capital_gains",
            "measurement_period": 2024,
        }
    }
    assert receipt["deviation"] == "per_clone_block_engine_resolution"
    sensitivity = receipt["block_sensitivity"]
    assert (
        "ons/corporate_land_value"
        in sensitivity["known_population_normalised_measures"]
    )
    assert set(sensitivity["present_in_this_run"]) <= set(
        sensitivity["known_population_normalised_measures"]
    )
    assert "not evidence for adjudication" in sensitivity["caveat"]
    assert (
        metrics["constituency"].index.tolist()
        == clone.frame.table("household")["household_id"].tolist()
    )
    assert restore(prepared).table("household").equals(clone.frame.table("household"))


def test_joint_candidate_f100_and_f001_end_to_end(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    """The driver solves one local/ladder/national matrix at both rung postures."""

    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    _write_staging_h5(
        input_h5,
        households_per_region=200,
        region_masses=(4.0, 10.0, 10.0, 9.0),
    )
    ladder_frame = _ladder_frame()
    ladder_frame.loc[0, "households"] = 1.0
    english = ladder_frame.iloc[0].copy()
    extra_english = []
    for suffix in (2, 3):
        row = english.copy()
        row["oa_code"] = f"E0000000{suffix}"
        row["lsoa_code"] = row["oa_code"]
        row["msoa_code"] = row["oa_code"]
        row["constituency_code"] = f"E1400000{suffix}"
        row["local_authority_code"] = f"E0900000{suffix}"
        row["households"] = 1.0
        extra_english.append(row)
    ladder_frame = pd.concat(
        [ladder_frame, pd.DataFrame(extra_english)], ignore_index=True
    )
    payload = assemble_uk_oa_ladder(ladder_frame, _ladder_metadata())
    np.savez_compressed(ladder_path, **payload)
    ladder = load_uk_oa_ladder(ladder_path)

    from microcosm.build.uk_runtime.ledger_targets import UK_CROSS_GRAIN_BRIDGES

    household_bridge = UK_CROSS_GRAIN_BRIDGES[0]
    reviewed_missing = {
        "ons.household_composition.unrelated_adult_households",
        "ons.household_composition.lone_parent_non_dependent_children_households",
        "ons.household_composition.multi_family_households",
    }
    selected_composition = tuple(
        target_id
        for target_id in household_bridge.higher_target_ids
        if target_id not in reviewed_missing
    )
    fanout_target_id = "dwp.uc.payment_distribution_single"
    fanout_names = tuple(f"payment-band-{index}" for index in range(3))
    national_registry = TargetRegistry(
        [
            *[
                TargetSpec(
                    name=target_id,
                    entity="household",
                    measure=f"national/composition_{index}",
                    value=33.0,
                    period=2025,
                    source="synthetic national fixture",
                    family="ons",
                    metadata={
                        "contract_target_id": target_id,
                        "ledger_geography_level": "country",
                        "ledger_geography_id": "K02000001",
                    },
                )
                for index, target_id in enumerate(selected_composition)
            ],
            *[
                TargetSpec(
                    name=name,
                    entity="household",
                    measure=f"national/payment_band_{index}",
                    value=33.0,
                    period=2025,
                    source="synthetic fan-out fixture",
                    family="dwp_uc",
                    metadata={
                        "contract_target_id": fanout_target_id,
                        "ledger_geography_level": "country",
                        "ledger_geography_id": "K03000001",
                    },
                )
                for index, name in enumerate(fanout_names)
            ],
        ],
        country="uk",
    )
    local_registry = TargetRegistry(
        [
            TargetSpec(
                name="ons.tenure.owned_outright@E09000001",
                entity="household",
                measure="tenure/owned_outright",
                value=1.0,
                period=2025,
                source="synthetic local fact fixture",
                family="ons",
                metadata={
                    "contract_target_id": "ons.tenure.owned_outright",
                    "geography_level": "local_authority",
                    "geography_id": "E09000001",
                    "ledger_fact_period": "2023",
                },
            )
        ],
        country="uk",
    )
    artifact = SimpleNamespace(
        provenance=lambda: {
            "facts_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "artifact_id": "synthetic-joint-fixture",
        }
    )
    joint_inputs = {
        "artifact": artifact,
        "calibration_year": 2025,
        "national_registry": national_registry,
        "band_edge_registry": national_registry,
        "local_registry": local_registry,
        "measure_exclusions": {
            f"compiled::{target_id}": {
                "tracking": "microcosm#791",
                "reason": "relationship-to-head is unavailable",
            }
            for target_id in reviewed_missing
        },
        "reviewed_unbound_higher_targets": {
            target_id: {
                "tracking": "microcosm#791",
                "reason": "relationship-to-head is unavailable",
            }
            for target_id in reviewed_missing
        },
    }
    monkeypatch.setattr(
        builder, "_load_joint_target_inputs", lambda _args: joint_inputs
    )

    constructions = []

    class StubResolver:
        def __init__(self, **kwargs):
            constructions.append(kwargs)
            self.frame = kwargs["frame"]
            # A live resolver writes the frame to a scratch H5 through the
            # national-frame writer, which validates the mass chain; a block
            # must therefore carry a record whose total equals its weights.
            validate_uk_national_frame(self.frame)
            self.simulation = object()
            self.contract_targets = {}

        def receipt(self):
            return {"mode": "stub", "policyengine_uk_version": "test"}

    monkeypatch.setattr(
        builder,
        "resolve_target_measures",
        # A live resolver injects ENGINE INPUTS (scratch columns the
        # materialization reads), some of which also exist on another entity
        # (region, esa_* on the spine); the driver must drop them before the
        # prepared frame or the flattening rule refuses the duplicate column.
        lambda _factory, _registry, provider, **_kwargs: SimpleNamespace(
            measure_inputs={
                ("household", "stub_engine_input"): np.ones(
                    len(provider.frame.table("household")), dtype=float
                ),
                ("person", "region"): np.zeros(
                    len(provider.frame.table("person")), dtype=float
                ),
            }
        ),
    )

    def _stub_materialize(adapter, registry, *, period, band_edge_registry=None):
        # Materialization is what mints the prepared measure columns the
        # national rows compile against; the stub writes them from the
        # injected input so the lifecycle matches the real stage.
        for spec in registry.specs:
            table = adapter.tables[spec.entity]
            table[spec.measure] = np.ones(len(table), dtype=float)
        return SimpleNamespace(skipped=())

    monkeypatch.setattr(builder, "materialize_uk_ledger_targets", _stub_materialize)
    monkeypatch.setattr(
        builder,
        "compute_household_metrics",
        lambda _simulation, area_type, *, household_ids, **_kwargs: pd.DataFrame(
            {
                "households": np.ones(len(household_ids), dtype=float),
                **(
                    {"tenure/owned_outright": np.ones(len(household_ids), dtype=float)}
                    if area_type == "la"
                    else {}
                ),
            },
            index=household_ids,
        ),
    )
    real_resolve = builder._resolve_candidate_engine_surface
    monkeypatch.setattr(
        builder,
        "_resolve_candidate_engine_surface",
        lambda *args, **kwargs: real_resolve(
            *args, resolver_factory=StubResolver, **kwargs
        ),
    )
    monkeypatch.setattr(
        builder,
        "rotated_uk_local_holdout",
        lambda *_args, **_kwargs: {"report_only": True, "folds": []},
    )
    import microcosm.build.uk_runtime.battery_bindings as battery_bindings

    monkeypatch.setattr(
        battery_bindings,
        "_local_area_roster",
        lambda _resource, levels: {
            "constituency": tuple(sorted(set(ladder.constituency_code))),
            "local_authority": tuple(sorted(set(ladder.local_authority_code))),
        },
    )
    real_support_summary = builder.uk_ladder_area_support_summary

    def support_summary(household, ladder_arg):
        if "household_weight" in household:
            return real_support_summary(household, ladder_arg)
        support = pd.DataFrame(
            {
                "nonzero_households": [len(household)],
                "effective_sample_size": [float(len(household))],
                "nonzero_source_households": [
                    household["source_household_id"].nunique()
                ],
            }
        )
        return {"constituency": support, "la": support}

    monkeypatch.setattr(builder, "uk_ladder_area_support_summary", support_summary)

    dry_out = tmp_path / "joint-dry"
    assert (
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--ladder",
                str(ladder_path),
                "--out",
                str(dry_out),
                "--n-clones",
                "2",
                "--dry-run",
            ]
        )
        == 0
    )
    dry_plan = json.loads(capsys.readouterr().out)
    dry_unbound = dry_plan["cross_grain"]["unbound_bridges"]
    assert dry_plan["cross_grain"]["empty_legs_licensed"] == []
    assert dry_plan["cross_grain"]["controls_without_lower_rows"] == []
    assert [entry["bridge_id"] for entry in dry_unbound] == [household_bridge.bridge_id]
    assert dry_unbound[0]["missing"] == sorted(reviewed_missing)
    dry_fanout = dry_plan["cross_grain"]["fanout_targets_not_controls"]
    assert dry_fanout == [
        {
            "target_id": fanout_target_id,
            "geography_id": "K03000001",
            "cells": 3,
            "cell_names": list(fanout_names),
            "activated_sum": 99.0,
            "reason": (
                "The activated cells are a band subset, so this distribution "
                "is not a cross-grain control."
            ),
        }
    ]
    assert "fanout_controls_summed" not in dry_plan["cross_grain"]
    assert not dry_out.exists()

    f100_out = tmp_path / "joint-f100"
    assert (
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--ladder",
                str(ladder_path),
                "--out",
                str(f100_out),
                "--n-clones",
                "2",
                "--epochs",
                "2",
                "--skip-holdout",
            ]
        )
        == 0
    )
    f100 = json.loads((f100_out / builder.MANIFEST_FILENAME).read_text())
    assert f100["schema_version"] == 2
    # The written rowwise artifact carries the shared ``clone_index`` name on
    # every table: the compact national loader must refuse it (flattening
    # rule) and the rowwise reader must undo the export rename.
    from microcosm.build.uk_runtime.rowwise_dataset import (
        ladder_clone_index_column,
        load_uk_rowwise_dataset,
    )

    dataset_path = Path(f100["outputs"]["dataset"]["path"])
    with pytest.raises(ValueError, match="globally unique"):
        builder.load_uk_national_frame(dataset_path)
    reloaded, provenance = load_uk_rowwise_dataset(dataset_path)
    assert provenance.source_h5 == dataset_path.resolve() or str(
        provenance.source_h5
    ).endswith(dataset_path.name)
    for entity in ("person", "benunit", "household"):
        assert ladder_clone_index_column(entity) in reloaded.table(entity).columns
        assert "clone_index" not in reloaded.table(entity).columns
    assert len(reloaded.table("household")) == f100["solve"]["n_households"]
    assert reloaded.weights_for("household").total == pytest.approx(
        f100["weights"]["calibration_mass_change"]["new_total"]
    )
    # Exact: the reader undoes the export rename and nothing else, so every
    # table equals the written one with clone_index renamed back.
    for entity in ("person", "benunit", "household"):
        written = pd.read_hdf(dataset_path, entity)
        expected = written.rename(
            columns={"clone_index": ladder_clone_index_column(entity)}
        )
        if entity == "household":
            # The frame carries the weight as its typed vector, not a column.
            np.testing.assert_array_equal(
                reloaded.weights_for("household").values,
                expected["household_weight"].to_numpy(dtype="float64"),
            )
            expected = expected.drop(columns=["household_weight"])
        got = reloaded.table(entity)
        assert sorted(got.columns) == sorted(expected.columns)
        pd.testing.assert_frame_equal(
            got[sorted(got.columns)].reset_index(drop=True),
            expected[sorted(expected.columns)].reset_index(drop=True),
            check_dtype=True,
        )
    assert f100["solve"]["n_targets_by_kind"] == {
        "local": 1,
        "ladder": 12,
        "national": len(national_registry.specs),
    }
    assert f100["solve"]["n_targets"] == 13 + len(national_registry.specs)
    assert f100["solve"]["measure_resolution"]["mode"] == "stub"
    assert f100["cross_grain"]["unbound_bridges"] == dry_unbound
    assert f100["solve"]["cross_grain"]["unbound_bridges"] == dry_unbound
    assert f100["cross_grain"]["empty_legs_licensed"] == []
    assert f100["solve"]["cross_grain"]["empty_legs_licensed"] == []
    assert f100["cross_grain"]["controls_without_lower_rows"] == []
    assert f100["solve"]["cross_grain"]["controls_without_lower_rows"] == []
    assert f100["cross_grain"]["fanout_targets_not_controls"] == dry_fanout
    assert f100["solve"]["cross_grain"]["fanout_targets_not_controls"] == dry_fanout
    assert "fanout_controls_summed" not in f100["cross_grain"]
    assert "fanout_controls_summed" not in f100["solve"]["cross_grain"]
    assert f100["releasable"] is True
    assert f100["measure_exclusions"] == joint_inputs["measure_exclusions"]
    assert f100["ladder_household_uprating"]["applied"] is False
    assert _spool_rows(f100_out)[0].rung == "f100"

    f001_out = tmp_path / "joint-f001"
    assert (
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--ladder",
                str(ladder_path),
                "--out",
                str(f001_out),
                "--n-clones",
                "1",
                "--sample-fraction",
                "0.01",
                "--epochs",
                "2",
                "--skip-holdout",
            ]
        )
        == 0
    )
    f001 = json.loads((f001_out / builder.MANIFEST_FILENAME).read_text())
    assert f001["rung_surface"]["dropped_cells"] > 0
    assert f001["rung_surface"]["dropped_unreachable_cells"] >= 0
    assert isinstance(f001["rung_surface"]["dropped_unreachable_by_grain"], dict)
    assert f001["rung_surface"]["dropped_by_grain"]["constituency"] >= 1
    assert f001["rung_surface"]["dropped_by_grain"]["la"] >= 1
    assert "uk_local_area_support" in f001["failing_gate_ids"]
    assert f001["releasable"] is False
    assert _spool_rows(f001_out)[0].rung == "f001"
    assert len(constructions) == 2


def test_candidate_refusal_records_receipt_and_reraises(
    monkeypatch,
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    output_dir = tmp_path / "candidate"
    _write_staging_h5(input_h5)
    _write_ladder(ladder_path)

    def failing_gate(*_args, **_kwargs):
        return builder.GateResult(
            name="spine_agreement",
            passed=False,
            failures=("post-calibration coverage failed",),
            details={"minimum": 0},
        )

    original = builder.UK_GATE_REGISTRY["spine_agreement"]
    monkeypatch.setattr(
        builder,
        "UK_GATE_REGISTRY",
        {
            **builder.UK_GATE_REGISTRY,
            "spine_agreement": replace(original, evaluator=failing_gate),
        },
    )

    with pytest.raises(
        builder.GateBatteryBlockedError, match="post-calibration coverage failed"
    ):
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--ladder",
                str(ladder_path),
                "--out",
                str(output_dir),
                "--n-clones",
                "2",
                "--seed",
                "7",
                "--epochs",
                "2",
            ]
        )

    rows = _spool_rows(output_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row.disposition == "failed"
    gate_report_path = output_dir / builder.LOCAL_GATE_REPORT_FILENAME_TEMPLATE.format(
        calibration_year=2025
    )
    assert gate_report_path.exists()
    assert row.gate_verdicts["uk_local_geography_ladder_post_calibration"] == {
        "verdict": "failed",
        "receipt": (
            f"{_local_ref(gate_report_path)}"
            "#/gates/uk_local_geography_ladder_post_calibration"
        ),
    }
    assert row.gate_verdicts["pipeline_error"]["verdict"] == "error"
    assert row.gate_verdicts["pipeline_error"]["receipt"].endswith("#/error_type")


def test_candidate_binding_adjudication_failure_records_failed_row(
    monkeypatch,
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    output_dir = tmp_path / "candidate"
    _write_staging_h5(input_h5)
    _write_ladder(ladder_path)

    import microcosm.build.uk_runtime.local_rowwise as local_rowwise

    monkeypatch.setattr(
        local_rowwise,
        "load_uk_reviewed_exclusion_register",
        lambda *_args, **_kwargs: {},
    )

    with pytest.raises(ValueError, match="census_disclosure_control_noise"):
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--ladder",
                str(ladder_path),
                "--out",
                str(output_dir),
                "--n-clones",
                "2",
                "--seed",
                "7",
                "--epochs",
                "2",
            ]
        )

    rows = _spool_rows(output_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row.disposition == "failed"
    assert "targets_bound" in row.phases_reached
    assert "solved" not in row.phases_reached
    assert row.gate_verdicts["pipeline_error"]["verdict"] == "error"
    assert row.gate_verdicts["pipeline_error"]["receipt"].endswith("#/error_type")


def test_candidate_setup_failure_records_failed_row(monkeypatch, tmp_path) -> None:
    """A pre-solve setup failure (ladder load) still spools a failed row.

    Adversarial-review finding on #666: input verification, frame/ladder
    loading, cloning, and target binding used to run before the recording
    envelope opened, so their failures escaped with no Logbook row.
    """

    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    output_dir = tmp_path / "candidate"
    _write_staging_h5(input_h5)
    _write_ladder(ladder_path)

    def failing_ladder_load(_path):
        raise RuntimeError("ladder artifact refused to parse")

    monkeypatch.setattr(builder, "load_uk_oa_ladder", failing_ladder_load)

    with pytest.raises(RuntimeError, match="ladder artifact refused to parse"):
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--ladder",
                str(ladder_path),
                "--out",
                str(output_dir),
                "--n-clones",
                "2",
                "--seed",
                "7",
                "--epochs",
                "2",
            ]
        )

    rows = _spool_rows(output_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row.disposition == "failed"
    assert row.gate_verdicts["pipeline_error"]["verdict"] == "error"
    assert row.gate_verdicts["pipeline_error"]["receipt"].endswith("#/error_type")
    assert "inputs_pinned" in row.phases_reached
    assert "cloned" not in row.phases_reached
    # Real input pins were promoted before the failure; the preflight
    # placeholder digest must not survive into the row.
    assert row.input_pins_digest != builder.preflight_digest(
        builder._UK_CANDIDATE_PIPELINE
    )


def test_candidate_refuses_separate_assignment_and_target_ladders(
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    first_path = tmp_path / "assignment_ladder.npz"
    second_path = tmp_path / "target_ladder.npz"
    _write_staging_h5(input_h5)
    assignment_ladder = _write_ladder(first_path)
    target_ladder = _write_ladder(
        second_path,
        household_counts=(4.0, 9.0, 10.0, 10.0),
    )
    assignment = builder._clone_with_ladder_binding(
        input_h5,
        assignment_ladder,
        n_clones=2,
        seed=7,
        source_year=2023,
        expected_constituency_vintage="2024_pcon",
        source_lineage_modulus=None,
    )

    with pytest.raises(ValueError, match="same loaded"):
        builder._build_bound_problem(
            assignment,
            target_ladder=target_ladder,
        )


def test_candidate_dry_run_refuses_ladder_sidecar_collision(
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    output_dir = tmp_path / "candidate"
    temporary_ladder = tmp_path / "ladder.npz"
    ladder_path = output_dir / builder.MANIFEST_FILENAME
    _write_staging_h5(input_h5)
    _write_ladder(temporary_ladder)
    output_dir.mkdir()
    temporary_ladder.replace(ladder_path)
    ladder_bytes = ladder_path.read_bytes()

    with pytest.raises(ValueError, match="differ"):
        builder.main(
            [
                "--input-h5",
                str(input_h5),
                "--ladder",
                str(ladder_path),
                "--out",
                str(output_dir),
                "--dry-run",
            ]
        )

    assert ladder_path.read_bytes() == ladder_bytes
    assert list(output_dir.iterdir()) == [ladder_path]


def test_candidate_publication_rolls_back_on_interrupt(
    monkeypatch,
    tmp_path,
) -> None:
    builder = _load_builder_module()
    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "candidate"
    staging_dir.mkdir()
    output_paths = builder._output_paths(
        output_dir,
        source_year=2023,
        calibration_year=2025,
    )
    staged = {key: staging_dir / path.name for key, path in output_paths.items()}
    for path in staged.values():
        path.write_text("complete staged artifact\n")

    original_replace = Path.replace

    def interrupt_support(self, target):
        if Path(target) == output_paths["support"]:
            raise KeyboardInterrupt
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", interrupt_support)
    with pytest.raises(KeyboardInterrupt):
        builder._publish_staged_files(staged, output_paths)

    assert not output_dir.exists()


def _failing_gate_evaluator(builder, name: str, message: str):
    def evaluator(*_args, **_kwargs):
        return builder.GateResult(
            name=name, passed=False, failures=(message,), details={"minimum": 0}
        )

    return evaluator


def _joint_f100_args(input_h5: Path, ladder_path: Path, output_dir: Path) -> list[str]:
    return [
        "--input-h5",
        str(input_h5),
        "--ladder",
        str(ladder_path),
        "--out",
        str(output_dir),
        "--n-clones",
        "2",
        "--seed",
        "7",
        "--epochs",
        "2",
        "--skip-holdout",
    ]


def test_candidate_weight_ratio_failure_is_reported_and_blocks(
    monkeypatch, tmp_path, capsys
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    output_dir = tmp_path / "candidate"
    _write_staging_h5(
        input_h5, households_per_region=200, region_masses=(4.0, 10.0, 10.0, 9.0)
    )
    _write_ladder(ladder_path)
    ladder = load_uk_oa_ladder(ladder_path)
    import microcosm.build.uk_runtime.battery_bindings as battery_bindings

    monkeypatch.setattr(
        battery_bindings,
        "_local_area_roster",
        lambda _resource, levels: {
            "constituency": tuple(sorted(set(ladder.constituency_code))),
            "local_authority": tuple(sorted(set(ladder.local_authority_code))),
        },
    )
    ratio = builder.UK_GATE_REGISTRY["weight_ratio"]
    monkeypatch.setattr(
        builder,
        "UK_GATE_REGISTRY",
        {
            **builder.UK_GATE_REGISTRY,
            "weight_ratio": replace(
                ratio,
                evaluator=_failing_gate_evaluator(
                    builder, "weight_ratio", "ratio 104.6 > 100"
                ),
            ),
        },
    )

    assert builder.main(_joint_f100_args(input_h5, ladder_path, output_dir)) == 1

    capsys.readouterr()
    manifest = json.loads((output_dir / builder.MANIFEST_FILENAME).read_text())
    assert manifest["failing_gate_ids"] == ["uk_local_weight_ratio"]
    assert manifest["blocked_at_f100"] is True
    assert manifest["diagnostic_failures"] == []
    assert manifest["blocking_failures"] == [
        "[uk_local_weight_ratio] ratio 104.6 > 100"
    ]
    assert manifest["releasable"] is False
    report = json.loads(
        Path(manifest["outputs"]["local_gate_report"]["path"]).read_text()
    )
    assert report["gates"]["uk_local_weight_ratio"]["criticality"] == "release_blocking"
    assert report["gates"]["uk_local_weight_ratio"]["status"] == "failed"
    assert _spool_rows(output_dir)[0].disposition == "failed"


def test_candidate_block_partitions_failures_by_criticality(
    monkeypatch, tmp_path, capsys
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    output_dir = tmp_path / "candidate"
    _write_staging_h5(
        input_h5, households_per_region=200, region_masses=(4.0, 10.0, 10.0, 9.0)
    )
    _write_ladder(ladder_path)
    registry = builder.UK_GATE_REGISTRY
    monkeypatch.setattr(
        builder,
        "UK_GATE_REGISTRY",
        {
            **registry,
            "area_support": replace(
                registry["area_support"],
                evaluator=_failing_gate_evaluator(
                    builder, "area_support", "ESS 42.3 < 50"
                ),
            ),
            "weight_ratio": replace(
                registry["weight_ratio"],
                evaluator=_failing_gate_evaluator(
                    builder, "weight_ratio", "ratio 578 > 100"
                ),
            ),
        },
    )

    assert builder.main(_joint_f100_args(input_h5, ladder_path, output_dir)) == 1

    captured = capsys.readouterr()
    assert "artifact unreleasable" in captured.err
    manifest = json.loads((output_dir / builder.MANIFEST_FILENAME).read_text())
    assert manifest["failing_gate_ids"] == [
        "uk_local_area_support",
        "uk_local_weight_ratio",
    ]
    assert manifest["blocked_at_f100"] is True
    assert manifest["blocking_failures"] == [
        "[uk_local_area_support] ESS 42.3 < 50",
        "[uk_local_weight_ratio] ratio 578 > 100",
    ]
    assert manifest["diagnostic_failures"] == []
    assert manifest["releasable"] is False
    assert _spool_rows(output_dir)[0].disposition == "failed"


def test_release_verdict_requires_single_block_engine() -> None:
    builder = _load_builder_module()
    releasable, posture = builder._release_verdict(
        sample_fraction=1.0, engine_blocks=1, release_blocking_gates_passed=True
    )
    assert releasable is True and all(posture.values())
    # A per-block engine resolution never writes a releasable artifact, even
    # with every release-blocking gate passed on the full rung (#736 erratum).
    releasable, posture = builder._release_verdict(
        sample_fraction=1.0, engine_blocks=15, release_blocking_gates_passed=True
    )
    assert releasable is False
    assert posture == {
        "full_rung": True,
        "single_block_engine": False,
        "release_blocking_gates_passed": True,
    }
    assert (
        builder._release_verdict(
            sample_fraction=0.1, engine_blocks=1, release_blocking_gates_passed=True
        )[0]
        is False
    )


def test_gate_criticality_reads_fail_closed() -> None:
    builder = _load_builder_module()
    assert builder._is_release_blocking({"criticality": "release_blocking"}) is True
    assert builder._is_release_blocking({"criticality": "diagnostic"}) is False
    # Missing or unknown criticality vetoes: schema drift on one entry cannot
    # drop a failed gate out of both the blocking list and all_gates_passed.
    assert builder._is_release_blocking({}) is True
    assert builder._is_release_blocking({"criticality": "advisory"}) is True
    blocking, diagnostic = builder._gate_failures_by_criticality(
        {
            "gates": {
                "uk_local_area_support": {
                    "status": "failed",
                    "failures": ["ESS 42.3 < 50"],
                },
                "uk_local_weight_ratio": {
                    "status": "failed",
                    "criticality": "diagnostic",
                    "failures": ["ratio 578 > 100"],
                },
                "uk_local_target_fit": {
                    "status": "passed",
                    "criticality": "diagnostic",
                },
            }
        }
    )
    assert blocking == ["[uk_local_area_support] ESS 42.3 < 50"]
    assert diagnostic == ["[uk_local_weight_ratio] ratio 578 > 100"]


def test_release_candidate_refuses_non_doctrine_solve_settings(tmp_path) -> None:
    builder = _load_builder_module()
    pin = "0" * 64
    base = [
        "--input-h5",
        str(tmp_path / "spine.h5"),
        "--input-sha256",
        pin,
        "--ladder",
        str(tmp_path / "ladder.npz"),
        "--ladder-sha256",
        pin,
        "--ledger-facts",
        str(tmp_path / "ledger"),
        "--ledger-facts-sha256",
        pin,
        "--ledger-manifest-sha256",
        pin,
        "--out",
        str(tmp_path / "out"),
        "--release-candidate",
    ]
    # The doctrine defaults are the release posture: nothing to refuse.
    args = builder._parse_args(base)
    builder._validate_cli_args(args)
    assert args.n_clones == builder.UK_LOCAL_CLONE_COUNT == 15
    assert args.epochs == builder.UK_LOCAL_SOLVE_EPOCHS == 1500
    assert args.target_weight_rule == "grain_equal"

    with pytest.raises(ValueError, match=r"--epochs != doctrine 1500"):
        builder._validate_cli_args(builder._parse_args([*base, "--epochs", "512"]))
    with pytest.raises(ValueError, match=r"--n-clones != doctrine 15"):
        builder._validate_cli_args(builder._parse_args([*base, "--n-clones", "10"]))
    with pytest.raises(ValueError, match=r"--target-weight-rule"):
        builder._validate_cli_args(
            builder._parse_args([*base, "--target-weight-rule", "uniform"])
        )


def test_candidate_multi_block_engine_run_is_never_releasable(
    monkeypatch, tmp_path, capsys
) -> None:
    """End to end: ``--engine-blocks K`` on f100 writes ``releasable: false``.

    Every release-blocking gate passes here; the posture alone withholds the
    verdict, and the manifest names the leg (``single_block_engine``). This is
    the assertion that catches a future caller bypassing ``_release_verdict``.
    """

    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    output_dir = tmp_path / "candidate"
    _write_staging_h5(
        input_h5, households_per_region=200, region_masses=(4.0, 10.0, 10.0, 9.0)
    )
    _write_ladder(ladder_path)
    ladder = load_uk_oa_ladder(ladder_path)
    import microcosm.build.uk_runtime.battery_bindings as battery_bindings

    monkeypatch.setattr(
        battery_bindings,
        "_local_area_roster",
        lambda _resource, levels: {
            "constituency": tuple(sorted(set(ladder.constituency_code))),
            "local_authority": tuple(sorted(set(ladder.local_authority_code))),
        },
    )

    args = [
        *_joint_f100_args(input_h5, ladder_path, output_dir),
        "--engine-blocks",
        "2",
    ]
    assert builder.main(args) == 0

    capsys.readouterr()
    manifest = json.loads((output_dir / builder.MANIFEST_FILENAME).read_text())
    assert manifest["parameters"]["engine_blocks"] == 2
    assert manifest["blocking_failures"] == []
    assert manifest["releasable"] is False
    assert manifest["release_posture"] == {
        "full_rung": True,
        "single_block_engine": False,
        "release_blocking_gates_passed": True,
    }
